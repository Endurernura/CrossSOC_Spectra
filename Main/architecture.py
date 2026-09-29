"""CrossSOC tokenizers, position encoding, attention, encoders and prediction heads."""
import os
import warnings
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

def apply_rotary_emb(x, cos, sin, interleaved=False, inplace=False,
                            seqlen_offsets=0, cu_seqlens=None, max_seqlen=None):
    """Pure-PyTorch drop-in for rotary.apply_rotary_emb (autograd supported).

    Mirrors the Triton kernel's math exactly: fp32 compute, cast back to the
    input dtype, first rotary_dim dims rotated, the rest passed through.
    Supports the interleaved (GPT-J) pairing used by CrossSOC."""
    if cu_seqlens is not None or seqlen_offsets != 0:
        raise ValueError("portable CrossSOC RoPE requires plain batches with zero offsets")
    d_half = cos.shape[-1]
    cos = cos.float()[None, :, None, :]
    sin = sin.float()[None, :, None, :]
    if interleaved:
        x0 = x[..., :2 * d_half:2].float()
        x1 = x[..., 1:2 * d_half:2].float()
        out = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1)
        out = out.flatten(-2)
    else:
        x0 = x[..., :d_half].float()
        x1 = x[..., d_half:2 * d_half].float()
        out = torch.cat((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1)
    if 2 * d_half < x.shape[-1]:
        out = torch.cat((out, x[..., 2 * d_half:].float()), dim=-1)
    out = out.to(x.dtype)
    return x.copy_(out) if inplace else out



if os.environ.get("CROSSSOC_ROTARY", "torch") == "triton":
    from rotary_triton import apply_rotary_emb


# ---------------------------------------------------------------------------
# Attention backend selection (tiered): flash-attn -> xformers -> PyTorch SDPA.
# q/k/v layout is [batch, seqlen, heads, head_dim] for all three backends.
# Note: xformers probes flash-attn at import time (unguarded), so a *broken*
# flash-attn install breaks xformers too — keep flash-attn either healthy or
# uninstalled; when absent, xformers falls back to its torch-native flash path.
# ---------------------------------------------------------------------------
try:
    from flash_attn import flash_attn_func as _flash_attn_func
except Exception:
    _flash_attn_func = None

try:
    import xformers.ops as _xops
    from xformers.ops import LowerTriangularMask as _LowerTriangularMask

    _xma = _xops.memory_efficient_attention
except Exception:
    _xma = None

_active_backend = None  # resolved on first CUDA call, then cached


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        is_causal=causal, enable_gqa=k.shape[2] != q.shape[2],
    )
    return out.transpose(1, 2)


def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    """Tiered attention dispatch; falls back permanently after a runtime failure."""
    global _active_backend
    if not q.is_cuda:
        return _sdpa(q, k, v, causal)
    candidates = ["flash", "xformers", "sdpa"] if _active_backend is None else [_active_backend]
    last_err = None
    for name in candidates:
        if name == "flash" and _flash_attn_func is None:
            continue
        if name == "xformers" and _xma is None:
            continue
        if name == "xformers" and k.shape[2] != q.shape[2]:
            # xformers GQA/MQA (BMGHK) is forward-only -> unusable for training;
            # leave mismatched-head calls to SDPA's native enable_gqa.
            continue
        try:
            if name == "flash":
                out = _flash_attn_func(q, k, v, causal=causal)
            elif name == "xformers":
                bias = _LowerTriangularMask() if causal else None
                out = _xma(q, k, v, attn_bias=bias)
            else:
                out = _sdpa(q, k, v, causal)
        except Exception as e:  # noqa: BLE001 - any backend failure falls through
            last_err = e
            warnings.warn(f"[architecture] attention backend '{name}' failed: {e}; trying next")
            continue
        if _active_backend != name:
            _active_backend = name
            print(f"[architecture] attention backend: {name}")
        return out
    raise RuntimeError(f"no usable attention backend (last error: {last_err})")


def diff_func(attn1: torch.Tensor, attn2: torch.Tensor, lambda_val: torch.Tensor) -> torch.Tensor:
    return attn1 - torch.sigmoid(lambda_val).unsqueeze(-1) * attn2


class MultiheadFlashDiffV2(nn.Module):
    """
    Differential Attention Version 2 (DiffAttnV2) implementation using Flash Attention.
    """
    def __init__(
        self,
        use_diff_v2: bool, # If False, acts as a baseline Transformer attention
        d_model: int,      # Model dimension
        num_heads: int,    # Number of output heads
        num_kv_heads: Optional[int], # Number of KV heads for GQA
        head_dim: int,     # Dimension per head
    ):
        super().__init__()
        self.use_diff_v2 = use_diff_v2
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = head_dim

        self.num_q_heads = 2 * self.num_heads if self.use_diff_v2 else self.num_heads
        self.q_proj = nn.Linear(self.d_model, self.num_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.d_model, bias=False)
        self.lambda_proj = nn.Linear(self.d_model, self.num_heads, bias=False) if self.use_diff_v2 else None

    def forward(
        self,
        x: torch.Tensor,               # Input tensor [bsz, seq_len, d_model]
        rel_pos: Tuple[torch.Tensor, torch.Tensor], # Rotary embedding (cos, sin)
        causal: bool = True,           # BERT-style bidirectional encoders pass False
    ) -> torch.Tensor:
        """
        Forward pass for MultiheadFlashDiffV2.

        Args:
            x: Input hidden states of shape [batch, length, d_model]
            rel_pos: Tuple of (cos, sin) tensors for rotary positional embeddings
            causal: Whether to apply causal masking (default True, original behavior)

        Returns:
            Output tensor of shape [batch, length, d_model]
        """
        bsz, tgt_len, _ = x.size()
        src_len = tgt_len

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, tgt_len, self.num_q_heads, self.head_dim)
        k = k.view(bsz, src_len, self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, self.head_dim)

        # The rotary kernel requires x.dtype == cos/sin.dtype exactly; under
        # bf16 autocast the projections output bf16 while the tables stay fp32.
        rel_pos = (rel_pos[0].to(q.dtype), rel_pos[1].to(q.dtype))
        q = apply_rotary_emb(q, *rel_pos, interleaved=True)
        k = apply_rotary_emb(k, *rel_pos, interleaved=True)

        # Single shared KV head: expand so every backend sees matched head counts
        if self.num_kv_heads == 1 and self.num_q_heads > 1:
            k = k.expand(bsz, src_len, self.num_q_heads, self.head_dim)
            v = v.expand(bsz, src_len, self.num_q_heads, self.head_dim)

        attn = _attention(q, k, v, causal=causal)
        if self.use_diff_v2:
            lambda_val = self.lambda_proj(x)
            attn1, attn2 = attn[:, :, 0::2], attn[:, :, 1::2]
            attn = diff_func(attn1, attn2, lambda_val)

        attn = attn.reshape(bsz, tgt_len, self.num_heads * self.head_dim)
        output = self.o_proj(attn)
        return output

# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------
N_BANDS = 4200
BANDS_PER_TOKEN = 10
N_TOKENS = N_BANDS // BANDS_PER_TOKEN  # 420

# LUCAS 2009 inner-processed column for each task name. SOC uses OC_gkg
# (organic carbon, g/kg) — the lab measurement SOC is derived from.
TASK_COLUMNS: Dict[str, str] = {
    "soc": "OC_gkg",
    "oc": "OC_gkg",
    "cec": "CEC_cmolc",
    "caco3": "CaCO3_gkg",
    "ph": "pH_H2O",
    "clay": "clay_pct",
    "silt": "silt_pct",
    "sand": "sand_pct",
}
DEFAULT_AUX_TASKS: Tuple[str, ...] = ("cec", "caco3", "ph", "clay", "silt", "sand")


# ---------------------------------------------------------------------------
# pre_encoder: packs of 10 raw bands -> one token, shared 3-layer MLP
# ---------------------------------------------------------------------------
class PreEncoder(nn.Module):
    """Shared-parameter three-layer MLP.

    The 4200 spectral bands are split into ``n_bands // bands_per_token``
    consecutive packs; every pack passes through the *same* MLP, producing one
    token per pack (no positional mixing inside the MLP — position is added by
    the encoder's RoPE).
    """

    def __init__(
        self,
        n_bands: int = N_BANDS,
        bands_per_token: int = BANDS_PER_TOKEN,
        d_model: int = 1024,
        hidden_dims: Tuple[int, int] = (256, 512),
    ):
        super().__init__()
        if n_bands % bands_per_token != 0:
            raise ValueError(f"n_bands ({n_bands}) must be divisible by bands_per_token")
        self.bands_per_token = bands_per_token
        self.n_tokens = n_bands // bands_per_token
        d1, d2 = hidden_dims
        self.mlp = nn.Sequential(
            nn.Linear(bands_per_token, d1),
            nn.GELU(),
            nn.Linear(d1, d2),
            nn.GELU(),
            nn.Linear(d2, d_model),
        )

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        """(B, n_bands) -> (B, n_tokens, d_model)"""
        bsz = spectra.shape[0]
        packs = spectra.reshape(bsz, self.n_tokens, self.bands_per_token)
        return self.mlp(packs)


class CNNPreEncoder(nn.Module):
    """Band normalization + 1D-CNN tokenizer (variant B4).

    BatchNorm1d normalizes every band across the batch (removes the
    brightness/albedo shortcut that saturates on dark organic soils); the conv
    stack maps (B, 1, n_bands) -> (B, n_tokens, d_model) with a ~43-band
    (~21 nm) receptive field per token instead of the MLP's 10 bands.
    """

    def __init__(
        self,
        n_bands: int = N_BANDS,
        bands_per_token: int = BANDS_PER_TOKEN,
        d_model: int = 1024,
        hidden_dims: Tuple[int, int] = (32, 64),
        kernels: Tuple[int, int, int] = (31, 7, 5),
        band_norm: bool = True,
    ):
        super().__init__()
        self.n_tokens = n_bands // bands_per_token
        c1, c2 = hidden_dims
        k1, k2, k3 = kernels
        p1, p2, p3 = k1 // 2, k2 // 2, k3 // 2
        got = (n_bands + 2 * p1 - k1) // bands_per_token + 1
        if got != self.n_tokens:
            raise ValueError(f"cnn tokenizer yields {got} tokens, expected {self.n_tokens}")
        self.bands_per_token = bands_per_token
        self.band_norm = nn.BatchNorm1d(n_bands) if band_norm else nn.Identity()
        self.conv = nn.Sequential(
            nn.Conv1d(1, c1, k1, stride=bands_per_token, padding=p1),
            nn.GELU(),
            nn.Conv1d(c1, c2, k2, stride=1, padding=p2),
            nn.GELU(),
            nn.Conv1d(c2, d_model, k3, stride=1, padding=p3),
        )

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        x = self.band_norm(spectra)                # (B, n_bands): per-band normalization
        x = x.unsqueeze(1)                         # (B, 1, n_bands)
        return self.conv(x).transpose(1, 2)        # (B, n_tokens, d_model)


def build_pre_encoder(kind: str, **kwargs) -> nn.Module:
    if kind == "mlp":
        kwargs.pop("band_norm", None)   # MLP path has no band normalization
        return PreEncoder(**kwargs)
    if kind == "cnn":
        return CNNPreEncoder(
            n_bands=kwargs["n_bands"], bands_per_token=kwargs["bands_per_token"],
            d_model=kwargs["d_model"], band_norm=kwargs.get("band_norm", True))
    raise ValueError(f"unknown pre-encoder kind: {kind!r}")


class VanillaMHA(nn.Module):
    """torch built-in nn.MultiheadAttention (variants B1M/B2).

    The "no advanced operators" backbone: standard multi-head self-attention
    with torch's internal kernels, no RoPE (the encoder adds learned positional
    embeddings instead), no differential attention, no xformers/flash-attn.
    The SDPA dispatch inside nn.MultiheadAttention is pinned to the math
    backend and the fused eval fastpath is disabled, so attention runs as
    plain matmul + softmax with no fused/flash kernel.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        if hasattr(torch.backends, "mha"):
            torch.backends.mha.set_fastpath_enabled(False)

    def forward(self, x: torch.Tensor,
                rel_pos: Tuple[torch.Tensor, torch.Tensor],
                causal: bool = False) -> torch.Tensor:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel(SDPBackend.MATH):
            out, _ = self.mha(x, x, x, need_weights=False)
        return out


class MoEFFN(nn.Module):
    """Top-1 token-routing MoE with a switch-style load-balancing loss."""

    def __init__(self, d_model: int, hidden: int, n_experts: int):
        super().__init__()
        self.n_experts = n_experts
        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model))
            for _ in range(n_experts)
        ])

    def route(self, h: torch.Tensor):
        """Returns (assign (B,L,) long, load-balancing aux loss)."""
        gate = self.router(h)
        assign = gate.argmax(dim=-1)
        probs = torch.softmax(gate.float(), dim=-1)
        onehot = F.one_hot(assign, self.n_experts).to(probs.dtype)
        frac = onehot.mean(dim=(0, 1))          # f_e: fraction of tokens per expert
        mean_p = probs.mean(dim=(0, 1))         # P_e: mean router probability
        aux = (frac * mean_p).sum() * self.n_experts
        return assign, aux

    def apply(self, h: torch.Tensor, assign: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(h)
        flat_h = h.reshape(-1, h.shape[-1])
        flat_a = assign.reshape(-1)
        flat_out = out.reshape(-1, h.shape[-1])
        for e, expert in enumerate(self.experts):
            sel = flat_a == e
            if sel.any():
                # index_put does not promote: match the destination dtype
                # (h is fp32 under autocast — LayerNorm — while Linear is bf16)
                flat_out[sel] = expert(flat_h[sel]).to(flat_out.dtype)
        return out


# ---------------------------------------------------------------------------
# diff_encoder: BERT-style stack of differential-transformer layers
# ---------------------------------------------------------------------------
def _build_rotary_tables(
    max_seqlen: int, head_dim: int, base: float = 10000.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compact (cos, sin) tables of shape (max_seqlen, head_dim // 2) for the
    interleaved Triton rotary kernel in rotary.py."""
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim // 2, dtype=torch.float32) / (head_dim // 2))
    )
    t = torch.arange(max_seqlen, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


class DiffEncoderLayer(nn.Module):
    """Pre-norm block: bidirectional differential attention + feed-forward.

    ffn_type="dense": standard MLP. ffn_type="moe": top-1 4-expert MoE whose
    routing runs outside the checkpointed segment (the load-balancing loss is
    exposed as ``self.aux_loss``); only the expert application is recomputed.
    gradient_checkpointing=True recomputes the attention and FFN segments
    separately during backward, halving the recompute peak.
    """

    def __init__(
        self,
        d_model: int = 1024,
        num_heads: int = 16,
        head_dim: int = 64,
        num_kv_heads: Optional[int] = None,
        ffn_mult: float = 4.0,
        gradient_checkpointing: bool = False,
        ffn_type: str = "dense",
        n_experts: int = 4,
        use_diff: bool = True,
        encoder: str = "diff",
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.ffn_type = ffn_type
        self.aux_loss = None
        self.attn_norm = nn.LayerNorm(d_model)
        if encoder == "vanilla_mha":
            self.attn = VanillaMHA(d_model, num_heads)
        else:
            self.attn = MultiheadFlashDiffV2(
                use_diff_v2=use_diff,
                d_model=d_model,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            )
        self.ffn_norm = nn.LayerNorm(d_model)
        hidden = int(d_model * ffn_mult)
        if ffn_type == "moe":
            self.moe = MoEFFN(d_model, hidden, n_experts)
        elif ffn_type == "dense":
            self.ffn = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Linear(hidden, d_model),
            )
        else:
            raise ValueError(f"unknown ffn_type: {ffn_type!r}")

    def _attn_seg(self, x: torch.Tensor, rel_pos: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return x + self.attn(self.attn_norm(x), rel_pos, causal=False)

    def _dense_ffn_seg(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.ffn_norm(x))

    def _moe_apply_seg(self, h: torch.Tensor, assign: torch.Tensor) -> torch.Tensor:
        return h + self.moe.apply(h, assign)

    def forward(self, x: torch.Tensor, rel_pos: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        ckpt = self.gradient_checkpointing and self.training
        if ckpt:
            x = grad_checkpoint(self._attn_seg, x, rel_pos, use_reentrant=False)
        else:
            x = self._attn_seg(x, rel_pos)
        if self.ffn_type == "moe":
            h = self.ffn_norm(x)
            assign, aux = self.moe.route(h)
            self.aux_loss = aux
            if ckpt:
                out = grad_checkpoint(self._moe_apply_seg, h, assign, use_reentrant=False)
            else:
                out = self._moe_apply_seg(h, assign)
            x = x + out
        else:
            if ckpt:
                x = grad_checkpoint(self._dense_ffn_seg, x, use_reentrant=False)
            else:
                x = self._dense_ffn_seg(x)
        return x


class DiffEncoder(nn.Module):
    """Stack of differential-transformer layers over spectral tokens.

    pooling="mean": average the token outputs -> (B, d_model)
    pooling="cls":  prepend a learnable [CLS] token, use its final state
    """

    def __init__(
        self,
        d_model: int = 1024,
        num_heads: int = 16,
        head_dim: int = 64,
        num_kv_heads: Optional[int] = None,
        n_layers: int = 3,
        ffn_mult: float = 4.0,
        n_tokens: int = N_TOKENS,
        pooling: str = "mean",
        gradient_checkpointing: bool = False,
        ffn_type: str = "dense",
        n_experts: int = 4,
        use_diff: bool = True,
        encoder: str = "diff",
    ):
        super().__init__()
        if pooling not in ("mean", "cls"):
            raise ValueError(f"unknown pooling: {pooling!r}")
        self.pooling = pooling
        self.gradient_checkpointing = gradient_checkpointing
        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, d_model)) if pooling == "cls" else None
        )
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.encoder = encoder
        if encoder == "vanilla_mha":
            # learned positional embeddings replace RoPE in the B1M backbone
            # (+1 slot covers the optional [CLS] token)
            self.pos_emb = nn.Parameter(torch.zeros(1, n_tokens + 1, d_model))
            nn.init.trunc_normal_(self.pos_emb, std=0.02)
        else:
            # Compact (cos, sin) tables for the interleaved Triton rotary
            # kernel in rotary.py; tables stay fp32 and are cast to the
            # activations' dtype at call time.
            self.register_buffer(
                "rotary_cos", _build_rotary_tables(n_tokens + 1, head_dim)[0], persistent=False
            )
            self.register_buffer(
                "rotary_sin", _build_rotary_tables(n_tokens + 1, head_dim)[1], persistent=False
            )
        self.layers = nn.ModuleList(
            DiffEncoderLayer(d_model, num_heads, head_dim, num_kv_heads, ffn_mult,
                             gradient_checkpointing=gradient_checkpointing,
                             ffn_type=ffn_type, n_experts=n_experts,
                             use_diff=use_diff, encoder=encoder)
            for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)

    def pop_aux_loss(self):
        """Aggregated MoE load-balancing loss (None for dense FFN); call after
        each forward — layers overwrite their caches on the next pass."""
        aux = None
        for layer in self.layers:
            if layer.ffn_type == "moe" and layer.aux_loss is not None:
                aux = layer.aux_loss if aux is None else aux + layer.aux_loss
        return aux

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, L, d_model) -> (B, d_model) joint representation"""
        h = tokens
        if self.cls_token is not None:
            cls = self.cls_token.expand(h.shape[0], 1, -1)
            h = torch.cat([cls, h], dim=1)
        if self.encoder == "vanilla_mha":
            rel_pos = None  # learned pos-emb already added; VanillaMHA ignores it
            h = h + self.pos_emb[:, : h.shape[1]]
        else:
            rel_pos = (
                self.rotary_cos[: h.shape[1]].to(h.dtype),
                self.rotary_sin[: h.shape[1]].to(h.dtype),
            )
        for layer in self.layers:
            h = layer(h, rel_pos)
        h = self.norm(h)
        if self.pooling == "cls":
            return h[:, 0]
        if self.cls_token is not None:
            return h[:, 1:].mean(dim=1)
        return h.mean(dim=1)


# ---------------------------------------------------------------------------
# predictor and full model
# ---------------------------------------------------------------------------
def build_predictor(
    d_model: int = 1024, n_tasks: int = 1, hidden_dims: Tuple[int, int] = (512, 256)
) -> nn.Sequential:
    """Three-layer MLP prediction head."""
    h1, h2 = hidden_dims
    return nn.Sequential(
        nn.Linear(d_model, h1),
        nn.GELU(),
        nn.Linear(h1, h2),
        nn.GELU(),
        nn.Linear(h2, n_tasks),
    )


class CrossSOCModel(nn.Module):
    """Spectral regression model: pre_encoder -> diff_encoder -> predictor.

    Single-task mode: ``aux_tasks=()`` — only the main task is predicted and
    the loss is the main-task MSE.
    Auxiliary-loss mode: pass aux task names; loss = main MSE + ``aux_weight``
    * mean(aux MSE). Targets are compared in normalized space; NaN targets
    (sparse labels) are masked out per task.
    """

    def __init__(
        self,
        main_task: str = "soc",
        aux_tasks: Iterable[str] = (),
        pooling: str = "mean",
        d_model: int = 1024,
        num_heads: int = 16,
        head_dim: int = 64,
        num_kv_heads: Optional[int] = None,
        n_layers: int = 3,
        ffn_mult: float = 4.0,
        pre_hidden: Tuple[int, int] = (256, 512),
        predictor_hidden: Tuple[int, int] = (512, 256),
        gradient_checkpointing: bool = False,
        head_type: str = "regression",
        bin_edges=None,
        bin_soft_sigma: float = 0.0,
        bin_soft_rel: float = 0.0,
        bin_class_weights=None,
        bands_per_token: int = BANDS_PER_TOKEN,
        pre_encoder: str = "mlp",
        band_norm: bool = True,
        ffn_type: str = "dense",
        n_experts: int = 4,
        moe_aux_weight: float = 0.01,
        use_diff: bool = True,
        encoder: str = "diff",
    ):
        super().__init__()
        for t in (main_task, *aux_tasks):
            if t not in TASK_COLUMNS:
                raise KeyError(f"unknown task {t!r}; known: {sorted(TASK_COLUMNS)}")
        self.main_task = main_task
        self.aux_tasks = tuple(aux_tasks)
        self.task_names = [main_task] + list(self.aux_tasks)
        self.pooling = pooling
        self.head_type = head_type
        self.moe_aux_weight = moe_aux_weight
        self.moe_aux = None
        self.pre_encoder = build_pre_encoder(
            pre_encoder, n_bands=N_BANDS, bands_per_token=bands_per_token,
            d_model=d_model, hidden_dims=pre_hidden, band_norm=band_norm)
        self.diff_encoder = DiffEncoder(
            d_model=d_model,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            n_layers=n_layers,
            ffn_mult=ffn_mult,
            pooling=pooling,
            gradient_checkpointing=gradient_checkpointing,
            n_tokens=self.pre_encoder.n_tokens,
            ffn_type=ffn_type,
            n_experts=n_experts,
            use_diff=use_diff,
            encoder=encoder,
        )
        if head_type == "binned":
            # Distributional head: predictor emits K class logits over SOC bins
            # (edges in physical g/kg, quantile-spaced on the train fold so the
            # classes are balanced); prediction = expectation over bin centers.
            if self.aux_tasks:
                raise NotImplementedError("binned head supports the main task only")
            if bin_edges is None:
                raise ValueError("head_type='binned' requires bin_edges (physical units)")
            edges = torch.as_tensor(bin_edges, dtype=torch.float32)
            if edges.dim() != 1 or len(edges) < 3 or not bool((edges[1:] > edges[:-1]).all()):
                raise ValueError(f"invalid bin_edges: {bin_edges}")
            self.register_buffer("bin_edges", edges)
            self.register_buffer("bin_centers", (edges[:-1] + edges[1:]) / 2)
            # soft-label CE: target distribution = Gaussian kernel around the
            # true value with heteroscedastic width max(sigma, rel * |y|); the
            # expectation readout can then interpolate *within* a bin
            self.bin_soft_sigma = float(bin_soft_sigma)
            self.bin_soft_rel = float(bin_soft_rel)
            if bin_class_weights is not None:
                w = torch.as_tensor(bin_class_weights, dtype=torch.float32)
                if w.shape != (len(edges) - 1,):
                    raise ValueError(
                        f"bin_class_weights has shape {tuple(w.shape)}, "
                        f"expected ({len(edges) - 1},)"
                    )
                self.register_buffer("bin_class_weights", w)
            else:
                self.bin_class_weights = None
            self.predictor = build_predictor(d_model, len(edges) - 1, predictor_hidden)
        elif head_type == "regression":
            self.predictor = build_predictor(d_model, len(self.task_names), predictor_hidden)
        else:
            raise ValueError(f"unknown head_type: {head_type!r}")

    def forward(self, spectra: torch.Tensor) -> Dict[str, torch.Tensor]:
        """(B, n_bands) raw reflectance -> {task: (B,) prediction} (regression)
        or {main_task: (B, K) bin logits} (binned head)."""
        tokens = self.pre_encoder(spectra)
        joint = self.diff_encoder(tokens)
        self.moe_aux = self.diff_encoder.pop_aux_loss()
        out = self.predictor(joint)
        if self.head_type == "binned":
            return {self.main_task: out}
        return {name: out[:, i] for i, name in enumerate(self.task_names)}

    def predict_expectation(self, logits: torch.Tensor) -> torch.Tensor:
        """(B, K) bin logits -> (B,) expected value in physical units."""
        probs = torch.softmax(logits.float(), dim=-1)
        return (probs * self.bin_centers).sum(dim=-1)

    def compute_loss(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        aux_weight: float = 0.5,
    ) -> torch.Tensor:
        if self.head_type == "binned":
            t = targets[self.main_task]
            logits = preds[self.main_task].float()
            if self.bin_soft_sigma > 0 or self.bin_soft_rel > 0:
                # soft-label CE: normalized Gaussian kernel over bin centers
                # (loss in fp32 — computed outside the autocast region)
                s = torch.clamp(t.abs() * self.bin_soft_rel,
                                min=max(self.bin_soft_sigma, 1e-3))
                kern = -0.5 * ((self.bin_centers[None, :] - t[:, None]) / s[:, None]) ** 2
                q = torch.softmax(kern, dim=-1)
                loss = -(q * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            else:
                # out-of-range targets snap to the first/last bin
                idx = torch.bucketize(t, self.bin_edges[1:-1])
                loss = F.cross_entropy(logits, idx, weight=self.bin_class_weights)
        else:
            loss = F.mse_loss(preds[self.main_task], targets[self.main_task])
            if self.aux_tasks and aux_weight > 0:
                aux_losses = []
                for name in self.aux_tasks:
                    t, p = targets[name], preds[name]
                    mask = ~torch.isnan(t)
                    if mask.any():
                        aux_losses.append(F.mse_loss(p[mask], t[mask]))
                if aux_losses:
                    loss = loss + aux_weight * torch.stack(aux_losses).mean()
        if self.moe_aux is not None:
            loss = loss + self.moe_aux_weight * self.moe_aux
        return loss
