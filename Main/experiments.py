"""Serial CrossSOC training, tokenizer sweep and checkpoint analysis."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from architecture import CrossSOCModel
from data_io import load_data, load_folds
from training import measure_flops
import training
import data_io
from architecture import apply_rotary_emb
from data_io import LUCASSpectrumDataset, metrics
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
# X1 protocol: L1, K25, AdamW, clip 1, 3% warmup, cosine, patience 20.
BASE = dict(n_layers=1, head_type='binned', n_bins=25, d_model=1024,
            num_heads=16, head_dim=64, ffn_mult=4., pre_encoder='mlp',
            bands_per_token=10, band_norm=False, encoder='diff', use_diff=True)
SPECS = {
    'X1': {}, 'B1': {'use_diff': False},
    'X2': dict(num_heads=4, head_dim=256),
    'X2A': dict(num_heads=4, head_dim=256, pre_encoder='cnn', bands_per_token=20, band_norm=True),
    'X2B': dict(num_heads=4, head_dim=128, pre_encoder='cnn', bands_per_token=20, band_norm=True, d_model=512, ffn_mult=2.),
    'B1M': dict(encoder='vanilla_mha', use_diff=False),
    'B2': dict(num_heads=4, head_dim=128, pre_encoder='cnn', bands_per_token=20, band_norm=True, d_model=512, ffn_mult=2., encoder='vanilla_mha', use_diff=False),
}


# Distributional uncertainty calculations.
PI_LEVELS = [0.5, 0.8, 0.9, 0.95]
SOC_BIN_EDGES = [0.,5.,10.,15.,20.,30.,50.,np.inf]
PIT_BINS = 20
RISK_FRACS = [i / 10 for i in range(1,11)]

def discrete_quantile(probs: np.ndarray, edges: np.ndarray, qs: np.ndarray) -> np.ndarray:
    """Quantiles of the piecewise-uniform CDF built from bin masses.

    probs (n, K), edges (K+1), qs (Q,) -> (n, Q). Within each bin the mass is
    spread uniformly, so quantiles interpolate inside bins instead of snapping
    to centers."""
    cum = np.cumsum(probs, axis=-1)
    cum = np.clip(cum, 0.0, 1.0)
    # first bin whose cumulative mass reaches q (broadcast: n x Q x K bools)
    idx = (cum[:, None, :] < qs[None, :, None]).sum(-1)
    idx = np.minimum(idx, probs.shape[-1] - 1)
    lo = edges[:-1][idx]
    width = np.diff(edges)[idx]
    prev = np.where(idx > 0, np.take_along_axis(cum, np.maximum(idx - 1, 0), axis=-1), 0.0)
    mass = np.take_along_axis(probs, idx, axis=-1)
    frac = np.where(mass > 1e-12, (qs[None, :] - prev) / np.maximum(mass, 1e-12), 0.5)
    return lo + np.clip(frac, 0.0, 1.0) * width


def discrete_pit(probs: np.ndarray, edges: np.ndarray, y: np.ndarray) -> np.ndarray:
    """F(y) under the piecewise-uniform CDF; clips to [0, 1] outside the support."""
    cum = np.cumsum(probs, axis=-1)
    k = np.clip(np.searchsorted(edges, y, side="right") - 1, 0, len(edges) - 2)
    prev = np.where(k > 0, cum[np.arange(len(y)), k - 1], 0.0)
    mass = probs[np.arange(len(y)), k]
    width = np.diff(edges)[k]
    frac = np.clip((y - edges[k]) / np.maximum(width, 1e-12), 0.0, 1.0)
    return np.clip(prev + mass * frac, 0.0, 1.0)


@torch.no_grad()
def infer_fold(model: CrossSOCModel, spectra, y_soc: np.ndarray, val_idx: np.ndarray,
               device: torch.device, batch_size: int = 256) -> dict:
    """Forward the validation fold; return per-sample distribution summaries."""
    loader = DataLoader(
        LUCASSpectrumDataset(spectra, np.zeros((len(val_idx), 1), dtype=np.float32),
                             indices=val_idx),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    logits_all = []
    amp = device.type == "cuda"
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(x)
        logits_all.append(out["soc"].float().cpu())
    logits = torch.cat(logits_all)
    probs = torch.softmax(logits, dim=-1).numpy()
    edges = model.bin_edges.cpu().numpy().astype(np.float64)
    centers = model.bin_centers.cpu().numpy().astype(np.float64)

    mean = probs @ centers
    sq_dev = (centers[None, :] - mean[:, None]) ** 2
    var = (probs * sq_dev).sum(-1)
    q = np.array([[0.5 - l / 2 for l in PI_LEVELS],
                  [0.5 + l / 2 for l in PI_LEVELS]]).T.flatten()
    bounds = discrete_quantile(probs, edges, np.sort(q))
    # bounds columns follow sorted qs: lo95, lo90, lo80, lo50, hi50, hi80, hi90, hi95
    K = probs.shape[-1]
    entropy = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(-1) / np.log(K)
    y_true = y_soc[val_idx]
    return {
        "point_id": None,  # filled by caller
        "true": y_true,
        "pred": mean,
        "unc_std": np.sqrt(np.maximum(var, 0.0)),
        "entropy": entropy,
        "pi95_lo": bounds[:, 0], "pi95_hi": bounds[:, 7],
        "pi90_lo": bounds[:, 1], "pi90_hi": bounds[:, 6],
        "pi80_lo": bounds[:, 2], "pi80_hi": bounds[:, 5],
        "pi50_lo": bounds[:, 3], "pi50_hi": bounds[:, 4],
        "pit": discrete_pit(probs, edges, y_true),
    }


def stratify(df: pd.DataFrame, meta: pd.DataFrame) -> list:
    """Stratum summaries across SOC range / mineral / land cover / country."""
    d = df.merge(meta, on="point_id", how="left")
    d["abs_err"] = (d.pred - d.true).abs()
    soc_range = pd.cut(d.true, SOC_BIN_EDGES, right=False)
    soc_range = soc_range.cat.rename_categories(
        {c: f"{c.left:g}-{c.right:g}" for c in soc_range.cat.categories})
    groups = [
        ("soc_range", soc_range.astype(str)),
        ("mineral", d.mineral.astype("Int64").astype(str)),
        ("land_cover", d.LC1_2009.astype(str)),
        ("country", d.country.astype(str)),
    ]
    rows = []
    for gname, gvals in groups:
        g = d.assign(_g=gvals).groupby("_g", observed=True)
        for key, sub in g:
            rows.append({
                "model": sub.model.iloc[0], "stratum_type": gname, "stratum": key,
                "n": len(sub),
                "r2": metrics(sub.true.to_numpy(), sub.pred.to_numpy())["r2"],
                "rmse": float(np.sqrt(np.mean((sub.pred - sub.true) ** 2))),
                "bias": float((sub.pred - sub.true).mean()),
                "mean_unc_std": float(sub.unc_std.mean()),
                "median_unc_std": float(sub.unc_std.median()),
                "mean_entropy": float(sub.entropy.mean()),
                "coverage90": float(((sub.true >= sub.pi90_lo) & (sub.true <= sub.pi90_hi)).mean()),
                "mean_width90": float((sub.pi90_hi - sub.pi90_lo).mean()),
            })
    return rows


def risk_coverage(df: pd.DataFrame) -> list:
    """RMSE/R2 of the most-confident x% (ascending predictive SD)."""
    d = df.sort_values("unc_std").reset_index(drop=True)
    rows = []
    for frac in RISK_FRACS:
        sub = d.iloc[: max(1, int(round(frac * len(d))))]
        rows.append({
            "model": d.model.iloc[0], "coverage_frac": frac, "n": len(sub),
            "rmse": float(np.sqrt(np.mean((sub.pred - sub.true) ** 2))),
            "r2": metrics(sub.true.to_numpy(), sub.pred.to_numpy())["r2"],
            "mae": float((sub.pred - sub.true).abs().mean()),
        })
    return rows


def summarize(model: str, df: pd.DataFrame, meta: pd.DataFrame) -> dict:
    df = df.merge(meta, on="point_id", how="left")
    df["abs_err"] = (df.pred - df.true).abs()
    m = metrics(df.true.to_numpy(), df.pred.to_numpy())
    cov, width = {}, {}
    for l in PI_LEVELS:
        lo, hi = df[f"pi{int(l * 100)}_lo"], df[f"pi{int(l * 100)}_hi"]
        cov[l] = float(((df.true >= lo) & (df.true <= hi)).mean())
        width[l] = float((hi - lo).mean())
    pit = df.pit.to_numpy()
    # KS distance of PIT from uniform = max |ECDF - 0.5 line| style statistic
    hist, _ = np.histogram(pit, bins=PIT_BINS, range=(0, 1))
    pit_dev = float(np.abs(hist / hist.sum() - 1 / PIT_BINS).sum() / 2)
    rho = df[["unc_std", "abs_err"]].corr(method="spearman").iloc[0, 1]
    return {
        "model": model, "n": m["n"], "r2": m["r2"], "rmse": m["rmse"], "mae": m["mae"],
        "spearman": m["spearman"],
        **{f"coverage{int(l * 100)}": cov[l] for l in PI_LEVELS},
        **{f"width{int(l * 100)}": width[l] for l in PI_LEVELS},
        "pit_hist_total_variation": pit_dev,
        "spearman_unc_abserr": float(rho),
        "rmse_most_confident_50pct": float(np.sqrt(
            np.mean((df.sort_values("unc_std").iloc[: len(df) // 2].pred
                     - df.sort_values("unc_std").iloc[: len(df) // 2].true) ** 2))),
    }



# Spectral importance and perturbation calculations.
DEVICE = torch.device("cpu")

def predict(model, x_np, bs=256):
    """(n, 4200) spectra -> (n,) physical-unit predictions."""
    preds = []
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=DEVICE.type == "cuda"):
        for i in range(0, len(x_np), bs):
            x = torch.from_numpy(x_np[i:i + bs]).to(DEVICE)
            out = model(x)
            preds.append(model.predict_expectation(out["soc"]).float().cpu().numpy())
    return np.concatenate(preds)


def eval_perturbed(model, y_true, x, perturb_fn):
    x_p = perturb_fn(x)
    p = predict(model, x_p)
    m = metrics(y_true, p)
    return {"rmse": m["rmse"], "mae": m["mae"], "r2": m["r2"],
            "spearman": m["spearman"]}


# ---------------------------------------------------------------------------
# importance methods (population-level ranking per model+fold)
# ---------------------------------------------------------------------------
def grad_x_input_importance(model, x, bs=64):
    """mean |d(expectation)/dx| * |x| over val samples -> (4200,) ranking."""
    total = torch.zeros(x.shape[1], device=DEVICE)
    n = 0
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(x[i:i + bs]).to(DEVICE).requires_grad_(True)
        with torch.autocast("cuda", torch.bfloat16, enabled=DEVICE.type == "cuda"):
            out = model(xb)["soc"]
        pred = model.predict_expectation(out).sum()
        g = torch.autograd.grad(pred, xb)[0].float()
        total += (g.abs() * xb.detach().abs()).sum(dim=0)
        n += len(xb)
    return (total / n).detach().cpu().numpy()


def attention_importance(model, x, bs=32):
    """Token importance from explicit attention weights (instrumented forward).

    Replaces the fused kernel pass for one eval sweep: vanilla A = mean_h
    softmax(q k^T / sqrt(d)); diff A_eff = mean_h (softmax(q_2h k) -
    sigmoid(lambda_h) softmax(q_2h+1 k)). importance[token] = mean_query |A|;
    computed in small batches, averaged over samples.
    """
    attn = model.diff_encoder.layers[0].attn
    enc = model.diff_encoder
    cos, sin = enc.rotary_cos, enc.rotary_sin
    n_tokens = enc.rotary_cos.shape[0] - 1  # table has +1 for the optional [CLS]
    from architecture import apply_rotary_emb
    total, n = torch.zeros(n_tokens, device=DEVICE), 0
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(x[i:i + bs]).to(DEVICE)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=DEVICE.type == "cuda"):
            h = model.pre_encoder(xb)
            xn = model.diff_encoder.layers[0].attn_norm(h)
            q = attn.q_proj(xn).view(xb.shape[0], xn.shape[1],
                                     attn.num_q_heads, attn.head_dim)
            k = attn.k_proj(xn).view(xb.shape[0], xn.shape[1],
                                     attn.num_kv_heads, attn.head_dim)
            q = apply_rotary_emb(q, cos[:q.shape[1]].to(q.dtype),
                                 sin[:q.shape[1]].to(q.dtype), interleaved=True)
            k = apply_rotary_emb(k, cos[:k.shape[1]].to(k.dtype),
                                 sin[:k.shape[1]].to(k.dtype), interleaved=True)
            if attn.num_kv_heads == 1 and attn.num_q_heads > 1:
                k = k.expand(xb.shape[0], k.shape[1], attn.num_q_heads, attn.head_dim)
            elif attn.num_q_heads > attn.num_kv_heads:  # GQA: kv head h//rep
                rep = attn.num_q_heads // attn.num_kv_heads
                k = k.repeat_interleave(rep, dim=2)
            q, k = q.float().transpose(1, 2), k.float().transpose(1, 2)  # B,H,L,D
            A = torch.softmax(q @ k.transpose(-1, -2) / (attn.head_dim ** 0.5),
                              dim=-1)                                     # B,H,Lq,Lk
            if attn.use_diff_v2:
                lam = torch.sigmoid(attn.lambda_proj(xn).float())         # B,L,H
                A1, A2 = A[:, 0::2], A[:, 1::2]
                A_eff = A1 - lam.transpose(1, 2).unsqueeze(-2) * A2
                a = A_eff.abs().mean(dim=1)                               # B,Lq,Lk
            else:
                a = A.mean(dim=1)
            total += a.abs().mean(dim=1).sum(dim=0).float()               # (L,)
            n += len(xb)
    return (total / n).cpu().numpy()  # token-level, length 420


def attention_importance_dispatch(model, x):
    """Token importance for either backbone: diff/RoPE models use the explicit
    softmax path; vanilla_mha backbones (B2) recompute attention
    from the MHA in_proj weights (no RoPE — learned pos-emb is already part of
    the hidden state)."""
    if model.diff_encoder.encoder == "vanilla_mha":
        return attention_importance_mha(model, x)
    return attention_importance(model, x)


def attention_importance_mha(model, x, bs=32):
    """Token importance for the torch nn.MultiheadAttention backbone (B2).

    A = mean_h softmax(q k^T / sqrt(d_h)); importance[token] = mean_query |A|."""
    import torch.nn.functional as F
    layer = model.diff_encoder.layers[0]
    attn = layer.attn.mha
    enc = model.diff_encoder
    d, H = attn.embed_dim, attn.num_heads
    dh = d // H
    Wq, Wk, _ = attn.in_proj_weight.chunk(3, dim=0)
    bq, bk, _ = attn.in_proj_bias.chunk(3, dim=0)
    total, n = torch.zeros(model.pre_encoder.n_tokens, device=DEVICE), 0
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(x[i:i + bs]).to(DEVICE)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=DEVICE.type == "cuda"):
            h = model.pre_encoder(xb)
            h = h + enc.pos_emb[:, : h.shape[1]]
            xn = layer.attn_norm(h)
            bsz, L = xn.shape[0], xn.shape[1]
            q = F.linear(xn, Wq, bq).view(bsz, L, H, dh).transpose(1, 2)
            k = F.linear(xn, Wk, bk).view(bsz, L, H, dh).transpose(1, 2)
            A = torch.softmax(q @ k.transpose(-1, -2) / dh ** 0.5, dim=-1)
            a = A.abs().mean(dim=1)                    # (B, Lq, Lk)
        total += a.abs().mean(dim=1).sum(dim=0).float()
        n += len(xb)
    return (total / n).cpu().numpy()


def expand_token_importance(tok_imp, n_bands=4200, bpt=10):
    """(n_tokens,) -> (n_bands,) by repeating each token's score over its bands."""
    return np.repeat(tok_imp, bpt)[:n_bands]






def build_model(ckpt, device):
    c = ckpt['config']
    names = ['main_task','aux_tasks','pooling','n_layers','d_model','num_heads','head_dim',
             'num_kv_heads','ffn_mult','encoder','head_type','bin_edges','bands_per_token',
             'pre_encoder','band_norm','use_diff','ffn_type','n_experts','moe_aux_weight']
    m = CrossSOCModel(**{k:c[k] for k in names if k in c}, gradient_checkpointing=False)
    m.load_state_dict(ckpt['model'])
    return m.to(device).eval()


# Experiment orchestration: import-based, serial execution.
def train(a, configs):
    rows = []
    for name, config in configs.items():
        for fold in a.folds:
            cmd = ['--version', a.version,
                   '--data-dir', a.data_dir, '--folds-file', a.folds_file,
                   '--run-dir', str(Path(a.run_dir)/name), '--fold', str(fold),
                   '--epochs', str(a.epochs), '--patience', '20', '--device', a.device,
                   '--batch-size', str(a.batch_size), '--num-workers', str(a.num_workers)]
            for k,v in config.items():
                if k == 'band_norm': cmd += ['--band-norm' if v else '--no-band-norm']
                elif k == 'use_diff':
                    if not v: cmd += ['--no-diff-attention']
                else: cmd += ['--'+k.replace('_','-'),str(v)]
            training.main(cmd)
            path = Path(a.run_dir)/name/a.version/f'fold{fold}'/'metrics.json'
            m = json.loads(path.read_text())
            rows.append({**m['report'], **m['metrics']['soc'], 'model':name, 'fold':fold})
    df = pd.DataFrame(rows)
    data_io.write_csv(Path(a.results_dir)/'performance_folds.csv',df)
    cols = ['rmse','mae','r2','pearson','spearman']
    summaries = []
    for name, sub in df.groupby('model'):
        for agg in ['mean','sd']:
            vals = sub[cols].mean() if agg == 'mean' else sub[cols].std(ddof=0)
            summaries.append(dict(model=name, agg=agg, n_folds=len(sub), **vals.to_dict()))
    data_io.write_csv(Path(a.results_dir)/'performance_summary.csv',pd.DataFrame(summaries))


def analyze(a, spectra, ids, features, folds):
    device = torch.device(a.device if a.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    global DEVICE
    DEVICE = device
    tables = {}
    def add(name, rows): tables.setdefault(name, []).extend(rows)
    for name in a.models:
        samples = []
        for fold in a.folds:
            d = Path(a.run_dir)/name/a.version/f'fold{fold}'
            ckpt = data_io.load_checkpoint(d/'best.pt',device)
            if ckpt['config']['fold'] != fold or not np.array_equal(ckpt['point_ids'],ids) or not np.array_equal(ckpt['val_indices'],folds[f'fold_val_{fold}']):
                raise ValueError('checkpoint fold mismatch')
            predictions = np.load(d/'predictions.npz')
            idx = folds[f'fold_val_{fold}']
            if not np.array_equal(predictions['point_ids'],ids[idx]) or not np.array_equal(predictions['y_true_soc'], features.OC_gkg.to_numpy(dtype=np.float32)[idx]):
                raise ValueError('checkpoint predictions do not match requested validation data/split')
            m = build_model(ckpt,device)
            x = np.array(spectra[idx],dtype=np.float32)
            y = features.OC_gkg.to_numpy()[idx]
            tag = dict(model=name, fold=fold)
            if a.mode == 'uncertainty':
                if m.head_type != 'binned': raise ValueError('uncertainty requires binned head')
                res = infer_fold(m,spectra,features.OC_gkg.to_numpy(),idx,device,a.batch_size)
                res['point_id'] = ids[idx]
                samples.append(pd.DataFrame(dict(**tag,**res)))
            elif a.mode == 'efficiency':
                xb = torch.from_numpy(x[:a.batch_size]).to(device)
                def sync():
                    if device.type == 'cuda': torch.cuda.synchronize(device)
                with torch.no_grad(), torch.autocast('cuda',torch.bfloat16,enabled=device.type=='cuda'):
                    for _ in range(a.warmup): m(xb)
                    sync()
                    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
                    start=time.perf_counter()
                    for _ in range(a.repeats): m(xb)
                    sync(); elapsed=time.perf_counter()-start
                add('efficiency',[dict(**tag,device=str(device),batch_size=len(xb),repeats=a.repeats,
                    samples_per_second=len(xb)*a.repeats/elapsed, params=sum(p.numel() for p in m.parameters()),
                    forward_flops=measure_flops(m,device),peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)])
            else:
                baseline=eval_perturbed(m,y,x,lambda z:z)
                add('gaussian_noise',[dict(**tag,sigma=0.,**baseline)])
                add('band_removal',[dict(**tag,del_pct=0,pos_seed=0,**baseline)])
                add('importance_removal',[dict(**tag,method='none',del_pct=0,**baseline)])
                for li,sig in enumerate(a.sigmas):
                    eps=np.random.default_rng(1000*fold+100*li+7).standard_normal(x.shape).astype(np.float32)
                    add('gaussian_noise',[dict(**tag,sigma=sig,**eval_perturbed(m,y,x,lambda z:z+sig*eps))])
                for li,pct in enumerate(a.del_pcts):
                    w=round(x.shape[1]*pct/100)
                    for seed in range(a.position_seeds):
                        start=int(np.random.default_rng(2000*fold+100*li+seed).integers(0,x.shape[1]-w+1))
                        xp=x.copy(); xp[:,start:start+w]=0
                        add('band_removal',[dict(**tag,del_pct=pct,pos_seed=seed,**eval_perturbed(m,y,xp,lambda z:z))])
                for method, imp in [('grad_x_input',grad_x_input_importance(m,x)),('attention',expand_token_importance(attention_importance_dispatch(m,x),bpt=m.pre_encoder.bands_per_token))]:
                    order=np.argsort(-imp)
                    for pct in a.del_pcts:
                        xp=x.copy(); xp[:,order[:round(x.shape[1]*pct/100)]]=0
                        add('importance_removal',[dict(**tag,method=method,del_pct=pct,**eval_perturbed(m,y,xp,lambda z:z))])
        if samples:
            df=pd.concat(samples,ignore_index=True)
            meta=features[['POINT_ID','mineral','LC1_2009','country']].rename(columns={'POINT_ID':'point_id'})
            add('uncertainty_per_sample',df.to_dict('records'))
            add('uncertainty_summary',[summarize(name,df,meta)])
            add('stratified_uncertainty',stratify(df,meta))
            add('risk_coverage',risk_coverage(df))
            hist,edges=np.histogram(df.pit,bins=20,range=(0,1))
            add('pit_hist',[dict(model=name,lo=edges[i],hi=edges[i+1],count=int(v)) for i,v in enumerate(hist)])
            add('calibration_by_level',[dict(model=name,nominal=l,empirical=((df.true>=df[f'pi{int(l*100)}_lo'])&(df.true<=df[f'pi{int(l*100)}_hi'])).mean(),mean_width=(df[f'pi{int(l*100)}_hi']-df[f'pi{int(l*100)}_lo']).mean()) for l in PI_LEVELS])
    for name,rows in tables.items(): data_io.write_csv(Path(a.results_dir)/(name+'.csv'),pd.DataFrame(rows))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['split','cv','tokenizer-sweep','faithfulness','uncertainty','efficiency'])
    p.add_argument('--data-dir',required=True)
    p.add_argument('--folds-file',required=True)
    p.add_argument('--split-kind',choices=['random','spatial'],default='random')
    p.add_argument('--run-dir',default=str(ROOT/'runs'))
    p.add_argument('--results-dir',default=str(ROOT/'results'))
    p.add_argument('--models',nargs='+',choices=list(SPECS),default=['X2B'])
    p.add_argument('--folds',nargs='+',type=int,default=list(range(5)))
    p.add_argument('--version',choices=['soc_only','multi_task'],default='soc_only')
    p.add_argument('--epochs',type=int,default=150)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--num-workers',type=int,default=4)
    p.add_argument('--device',default='auto')
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--repeats',type=int,default=100)
    p.add_argument('--sigmas',nargs='+',type=float,default=[round(.005*i,3) for i in range(1,11)])
    p.add_argument('--del-pcts',nargs='+',type=int,default=list(range(2,41,2)))
    p.add_argument('--position-seeds',type=int,default=5)
    a=p.parse_args()
    if a.epochs<1 or a.batch_size<1 or a.repeats<1 or a.warmup<0 or a.position_seeds<1 or any(not 0<=v<=100 for v in a.del_pcts) or any(not np.isfinite(v) or v<0 for v in a.sigmas): p.error('invalid training/profiling/perturbation counts')
    if a.mode == 'split':
        data_io.main(['--data-dir',a.data_dir,'--out',a.folds_file,'--kind',a.split_kind])
        return
    spectra,_,ids,features=load_data(a.data_dir)
    folds=load_folds(a.folds_file,ids)
    if len(set(a.folds)) != len(a.folds) or any(f'fold_val_{f}' not in folds for f in a.folds): p.error('invalid or duplicate fold selection')
    Path(a.results_dir).mkdir(parents=True,exist_ok=True)
    if a.mode=='cv': train(a,{n:dict(BASE,**SPECS[n]) for n in a.models})
    elif a.mode=='tokenizer-sweep':
        train(a,{f'{pre}_tok{4200//bpt}':dict(BASE,pre_encoder=pre,bands_per_token=bpt,band_norm=pre=='cnn') for pre in ['mlp','cnn'] for bpt in [40,20,10,5]})
    else: analyze(a,spectra,ids,features,folds)


if __name__=='__main__': main()
