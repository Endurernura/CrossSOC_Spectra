"""Single-fold CrossSOC training, early stopping and checkpoint generation."""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from architecture import DEFAULT_AUX_TASKS, CrossSOCModel
from data_io import (
    write_json, write_predictions, save_checkpoint, load_checkpoint, write_history,
    LUCASSpectrumDataset,
    TargetScaler,
    format_task_metrics,
    load_data,
    load_folds,
    metrics,
    target_matrix,
)

VERSION_AUX_TASKS = {"soc_only": (), "multi_task": DEFAULT_AUX_TASKS}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CrossSOC fold training")
    p.add_argument("--version", choices=["soc_only", "multi_task"], required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--folds-file", required=True)
    p.add_argument("--run-dir", default=str(Path(__file__).resolve().parents[1] / "runs"))
    p.add_argument("--run-subdir", default="",
                   help="optional leaf appended to fold<k>/ (e.g. seed142 for "
                        "multi-seed repeats); empty keeps the standard layout")
    p.add_argument("--main-task", default="soc")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    # Optimization stability protocol.
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="max global grad norm, 0 disables clipping")
    p.add_argument("--warmup-frac", type=float, default=0.03,
                   help="fraction of total steps for linear LR warmup (0 = none)")
    p.add_argument("--lr-schedule", choices=["cosine", "constant"], default="cosine")
    # Distributional-head options
    p.add_argument("--target-transform", choices=["none", "log1p"], default="none",
                   help="log1p = train on z-scored log1p(SOC) (soc_only regression)")
    p.add_argument("--bin-soft-sigma", type=float, default=0.0,
                   help="binned CE soft-label kernel floor in g/kg (0 = hard labels)")
    p.add_argument("--bin-soft-rel", type=float, default=0.0,
                   help="binned CE soft-label kernel width as a fraction of |y|")
    p.add_argument("--aux-weight", type=float, default=0.5)
    p.add_argument("--n-layers", type=int, default=1,
                   help="number of differential-transformer layers (depth ablation)")
    p.add_argument("--head-type", choices=["regression", "binned"], default="binned",
                   help="binned = distributional head (CE over SOC bins, expectation at inference)")
    p.add_argument("--n-bins", type=int, default=25,
                   help="number of SOC bins for --head-type binned (train-fold quantiles)")
    p.add_argument("--bin-weights", choices=["none", "inv_freq"], default="none",
                   help="binned CE class weights (inv_freq ~no-op for quantile bins, kept for reference)")
    p.add_argument("--bands-per-token", type=int, default=20,
                   help="spectral bands per token (4200 must be divisible by this value)")
    p.add_argument("--pre-encoder", choices=["mlp", "cnn"], default="cnn",
                   help="shared MLP or convolutional tokenizer")
    p.add_argument("--band-norm", action=argparse.BooleanOptionalAction, default=True,
                   help="per-band BatchNorm before CNN tokenization")
    p.add_argument("--ffn-type", choices=["dense", "moe"], default="dense",
                   help="dense or mixture-of-experts feed-forward network")
    p.add_argument("--no-diff-attention", action="store_true",
                   help="vanilla attention baseline (use_diff_v2=False), e.g. B1 backbone")
    p.add_argument("--encoder", choices=["diff", "vanilla_mha"], default="diff",
                   help="vanilla_mha = torch built-in nn.MultiheadAttention + "
                        "learned pos-emb, no RoPE/xformers (B1M)")
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-heads", type=int, default=4,
                   help="output attention heads (diff attn doubles the q heads)")
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--num-kv-heads", type=int, default=None,
                   help="GQA kv heads; default = num_heads (no GQA)")
    p.add_argument("--ffn-mult", type=float, default=2.0,
                   help="dense FFN hidden dim = ffn_mult * d_model")
    p.add_argument("--n-experts", type=int, default=4)
    p.add_argument("--moe-aux-weight", type=float, default=0.01)
    p.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader workers per process (12-core host: 2 concurrent "
                        "runs x (1 main + 4 workers) = 10)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--no-amp", action="store_true", help="disable bf16 autocast")
    p.add_argument("--no-grad-checkpoint", action="store_true")
    p.add_argument("--log-every", type=int, default=20)
    return p.parse_args(argv)


def measure_flops(model: torch.nn.Module, device: torch.device):
    """Per-sample forward FLOPs via torch's FlopCounterMode (None if unavailable)."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None
    was_training = model.training
    model.eval()
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        model(torch.zeros(1, 4200, device=device))
    if was_training:
        model.train()
    return int(counter.get_total_flops())


def autocast_ctx(enabled: bool):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled)


def prediction_for_eval(model, out):
    """(B,) physical-unit prediction from either head type."""
    if model.head_type == "binned":
        return model.predict_expectation(out)
    return out


def soc_mse(model, preds, targets, main_task):
    """SOC MSE of the point prediction (expectation for the binned head) —
    the same quantity across head types, used for tracking/early stopping."""
    return torch.nn.functional.mse_loss(
        prediction_for_eval(model, preds[main_task]).float(), targets[main_task])


def train_one_epoch(model, loader, optimizer, task_names, main_task, aux_weight,
                    amp, device, log_every, scheduler=None, grad_clip=0.0):
    model.train()
    sums = {"total": 0.0, "soc": 0.0}
    gnorms = []
    n = 0
    for step, (x, y) in enumerate(loader, 1):
        x = x.to(device, non_blocking=True)
        targets = {name: y[:, i].to(device, non_blocking=True) for i, name in enumerate(task_names)}
        with autocast_ctx(amp):
            preds = model(x)
        loss = model.compute_loss(preds, targets, aux_weight=aux_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip) \
            if grad_clip > 0 else None
        if gnorm is None:  # no clipping: still log the raw norm (divergence forensics)
            gnorm = torch.linalg.vector_norm(torch.stack([
                p.grad.norm() for p in model.parameters() if p.grad is not None]))
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        sums["total"] += float(loss.detach())
        sums["soc"] += float(soc_mse(model, preds, targets, main_task).detach())
        gnorms.append(float(gnorm))
        n += 1
        if step % log_every == 0:
            print(f"    step {step:4d}/{len(loader)}  loss {sums['total'] / n:.4f}", flush=True)
    return (sums["total"] / n, sums["soc"] / n,
            float(np.mean(gnorms)), float(np.max(gnorms)))


@torch.no_grad()
def validate(model, loader, task_names, main_task, aux_weight, amp, device):
    model.eval()
    soc_sum, total_sum, n = 0.0, 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        targets = {name: y[:, i].to(device, non_blocking=True) for i, name in enumerate(task_names)}
        with autocast_ctx(amp):
            preds = model(x)
        soc_sum += float(soc_mse(model, preds, targets, main_task))
        total_sum += float(model.compute_loss(preds, targets, aux_weight=aux_weight))
        n += 1
    return soc_sum / n, total_sum / n


@torch.no_grad()
def evaluate_fold(model, loader, task_names, amp, device, to_phys):
    """Timed pure-forward pass (throughput) + prediction collection (metrics).

    ``to_phys(z, col)`` maps model-space values to physical units
    (scaler inverse-transform, plus the expm1 link for log1p targets)."""
    model.eval()
    # throughput: pure forward, first batch is warmup
    for x, _ in loader:
        with autocast_ctx(amp):
            model(x.to(device, non_blocking=True))
        break
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for x, _ in loader:
        with autocast_ctx(amp):
            model(x.to(device, non_blocking=True))
    if device.type == "cuda":
        torch.cuda.synchronize()
    infer_s = time.time() - t0
    n_infer = len(loader.dataset)

    # prediction collection for metrics (inverse-transform to physical units)
    raw_preds = {t: [] for t in task_names}
    raw_trues = {t: [] for t in task_names}
    for x, y in loader:
        with autocast_ctx(amp):
            out = model(x.to(device, non_blocking=True))
        for i, t in enumerate(task_names):
            raw_preds[t].append(
                prediction_for_eval(model, out[t]).float().cpu().numpy())
            raw_trues[t].append(y[:, i].numpy())
    results, arrays = {}, {}
    for i, t in enumerate(task_names):
        p = np.concatenate(raw_preds[t])
        y = np.concatenate(raw_trues[t])
        y_phys = to_phys(y, col=i)
        p_phys = to_phys(p, col=i)
        results[t] = metrics(y_phys, p_phys)
        arrays[f"y_true_{t}"] = y_phys   # physical units, matching metrics.json
        arrays[f"y_pred_{t}"] = p_phys
    return results, arrays, infer_s, n_infer


def main(argv=None):
    args = parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.log_every < 1 or args.patience < 1:
        raise ValueError("epochs, batch-size, log-every and patience must be positive")
    torch.manual_seed(args.seed + args.fold)
    np.random.seed(args.seed + args.fold)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    amp = (device.type == "cuda") and not args.no_amp

    data = load_data(args.data_dir)
    spectra, wavelengths, point_ids, features = data
    folds = load_folds(args.folds_file, point_ids)
    val_idx = folds[f"fold_val_{args.fold}"]
    val_mask = np.zeros(len(point_ids), dtype=bool)
    val_mask[val_idx] = True
    train_idx = np.where(~val_mask)[0]

    aux_tasks = VERSION_AUX_TASKS[args.version]
    task_names = [args.main_task] + list(aux_tasks)
    if args.target_transform == "log1p" and (
            args.version != "soc_only" or args.head_type == "binned"):
        raise ValueError("--target-transform log1p only supports soc_only + regression head")
    y_all = target_matrix(features, task_names)
    if not np.isfinite(y_all).all():
        raise ValueError("selected target columns must be finite (complete-case protocol)")
    bin_class_weights = None
    if args.head_type == "binned":
        # physical-unit targets: the CE head works on bins, the MSE monitor on
        # the expectation; no z-score needed
        scaler = TargetScaler.identity(len(task_names))
        quantiles = np.linspace(0.0, 1.0, args.n_bins + 1)
        bin_edges = np.quantile(y_all[train_idx, 0], quantiles)
        bin_edges[0] = min(0.0, float(bin_edges[0]))   # SOC floor at 0 g/kg
        bin_edges = np.unique(bin_edges)               # guard against duplicate quantiles
        if args.bin_weights == "inv_freq":
            idx = torch.bucketize(
                torch.as_tensor(y_all[train_idx, 0]),
                torch.as_tensor(bin_edges[1:-1], dtype=torch.float32))
            counts = torch.bincount(idx, minlength=len(bin_edges) - 1).float()
            w = 1.0 / torch.clamp(counts / counts.sum(), min=1e-6)
            bin_class_weights = (w / w.mean()).numpy()
            print(f"[{args.version} fold{args.fold}] bin weights (inv_freq): "
                  f"{np.round(bin_class_weights, 2).tolist()}")
        print(f"[{args.version} fold{args.fold}] binned head: {len(bin_edges)-1} bins, "
              f"edges = {np.round(bin_edges, 1).tolist()} g/kg (train quantiles)")
    else:
        scaler = None
        bin_edges = None
    if args.target_transform == "log1p":
        y_model = np.log1p(np.maximum(y_all, 0.0))
        scaler = TargetScaler().fit(y_model[train_idx])
        def to_phys(z, col, _s=scaler):
            return np.expm1(_s.inverse_transform(z, col=col))
    else:
        if scaler is None:
            scaler = TargetScaler().fit(y_all[train_idx])  # fold-train statistics only
        def to_phys(z, col, _s=scaler):
            return _s.inverse_transform(z, col=col)
    y_norm = scaler.transform(
        y_model if args.target_transform == "log1p" else y_all)

    run_dir = Path(args.run_dir) / args.version / f"fold{args.fold}"
    if args.run_subdir:
        run_dir = run_dir / args.run_subdir
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ld = DataLoader(
        LUCASSpectrumDataset(spectra, y_norm[train_idx], indices=train_idx),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
        generator=torch.Generator().manual_seed(args.seed + args.fold),
    )
    val_ld = DataLoader(
        LUCASSpectrumDataset(spectra, y_norm[val_idx], indices=val_idx),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model = CrossSOCModel(
        main_task=args.main_task, aux_tasks=aux_tasks, pooling=args.pooling,
        n_layers=args.n_layers,
        d_model=args.d_model, num_heads=args.num_heads, head_dim=args.head_dim,
        num_kv_heads=args.num_kv_heads, ffn_mult=args.ffn_mult,
        encoder=args.encoder,
        gradient_checkpointing=not args.no_grad_checkpoint,
        head_type=args.head_type, bin_edges=bin_edges,
        bin_soft_sigma=args.bin_soft_sigma, bin_soft_rel=args.bin_soft_rel,
        bin_class_weights=bin_class_weights,
        bands_per_token=args.bands_per_token, pre_encoder=args.pre_encoder,
        band_norm=args.band_norm,
        ffn_type=args.ffn_type, n_experts=args.n_experts,
        moe_aux_weight=args.moe_aux_weight,
        use_diff=not args.no_diff_attention,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_flops = measure_flops(model, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = None
    if args.lr_schedule == "cosine":
        total_steps = max(1, args.epochs * len(train_ld))
        warmup_steps = int(args.warmup_frac * total_steps)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f"[{args.version} fold{args.fold}] device={device} bf16_amp={amp} "
          f"grad_ckpt={not args.no_grad_checkpoint}")
    print(f"[{args.version} fold{args.fold}] model={type(model).__name__} "
          f"(Main/architecture.py: pre_encoder -> diff_encoder -> predictor) "
          f"layers={args.n_layers} head={args.head_type} tokens={model.pre_encoder.n_tokens} "
          f"pre={args.pre_encoder}{' +bandnorm' if args.band_norm else ''} ffn={args.ffn_type} "
          f"tasks={task_names} params={n_params:,} "
          f"train={len(train_idx)} val={len(val_idx)}")

    best_val, best_epoch, since_best = float("inf"), -1, 0
    train_seconds, epochs_run = 0.0, 0
    history_path = run_dir / "history.csv"
    write_history(history_path, ["epoch", "train_loss", "train_soc_loss", "val_soc_loss", "val_total_loss",
             "is_best", "epoch_seconds", "grad_norm_mean", "grad_norm_max"])
    t_start = time.time()
    early_stopped = False
    for epoch in range(args.epochs):
        t_ep = time.time()
        train_loss, train_soc, gnorm_mean, gnorm_max = train_one_epoch(
            model, train_ld, optimizer, task_names, args.main_task,
            args.aux_weight, amp, device, args.log_every,
            scheduler=scheduler, grad_clip=args.grad_clip)
        val_soc, val_total = validate(
            model, val_ld, task_names, args.main_task, args.aux_weight, amp, device)
        epoch_s = time.time() - t_ep
        train_seconds += epoch_s
        epochs_run += 1

        is_best = val_soc < best_val - 1e-6
        if is_best:
            best_val, best_epoch, since_best = val_soc, epoch, 0
            save_checkpoint({"model": model.state_dict(), "epoch": epoch,
                        "val_soc_loss": val_soc,
                        "config": {"main_task": args.main_task, "aux_tasks": list(aux_tasks),
                                   "pooling": args.pooling, "version": args.version,
                                   "fold": args.fold, "head_type": args.head_type,
                                   "bin_edges": bin_edges.tolist() if bin_edges is not None else None,
                                   "n_layers": args.n_layers, "d_model": args.d_model,
                                   "num_heads": args.num_heads, "head_dim": args.head_dim,
                                   "num_kv_heads": args.num_kv_heads,
                                   "ffn_mult": args.ffn_mult, "encoder": args.encoder,
                                   "bands_per_token": args.bands_per_token,
                                   "pre_encoder": args.pre_encoder,
                                   "band_norm": args.band_norm,
                                   "n_bins": args.n_bins, "use_diff": not args.no_diff_attention,
                                   "ffn_type": args.ffn_type, "n_experts": args.n_experts,
                                   "moe_aux_weight": args.moe_aux_weight,
                                   "bin_soft_sigma": args.bin_soft_sigma,
                                   "bin_soft_rel": args.bin_soft_rel},
                        "target_scaler": {"mean": scaler.mean_.tolist(), "std": scaler.std_.tolist()},
                        "target_transform": args.target_transform,
                        "protocol": vars(args), "point_ids": point_ids,
                        "train_indices": train_idx, "val_indices": val_idx},
                       run_dir / "best.pt")
        else:
            since_best += 1
        write_history(history_path, [epoch, f"{train_loss:.6f}", f"{train_soc:.6f}",
                                    f"{val_soc:.6f}", f"{val_total:.6f}",
                                    int(is_best), f"{epoch_s:.2f}",
                                    f"{gnorm_mean:.4f}", f"{gnorm_max:.4f}"], append=True)
        print(f"[{args.version} fold{args.fold}] epoch {epoch:3d}  "
              f"train {train_loss:.4f} (soc {train_soc:.4f})  "
              f"val_soc {val_soc:.4f}  {'*best*' if is_best else f'({since_best}/{args.patience})'}  "
              f"gn {gnorm_mean:.2f}/{gnorm_max:.2f}  {epoch_s:.1f}s", flush=True)
        if since_best >= args.patience:
            early_stopped = True
            print(f"[{args.version} fold{args.fold}] early stopping at epoch {epoch} "
                  f"(best epoch {best_epoch}, val_soc {best_val:.4f})", flush=True)
            break
    total_runtime = time.time() - t_start

    # final evaluation with the best checkpoint
    ckpt = load_checkpoint(run_dir / "best.pt", device)
    model.load_state_dict(ckpt["model"])
    results, arrays, infer_s, n_infer = evaluate_fold(
        model, val_ld, task_names, amp, device, to_phys)
    infer_throughput = n_infer / infer_s if infer_s > 0 else float("nan")

    peak_allocated = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else float("nan")
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30 if device.type == "cuda" else float("nan")

    report = {
        "model": type(model).__name__,
        "head_type": args.head_type,
        "n_bins": int(len(bin_edges) - 1) if bin_edges is not None else None,
        "n_layers": args.n_layers,
        "tokens": int(model.pre_encoder.n_tokens),
        "pre_encoder": args.pre_encoder,
        "ffn_type": args.ffn_type,
        "diff_attention": not args.no_diff_attention,
        "encoder": args.encoder,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "head_dim": args.head_dim,
        "num_kv_heads": args.num_kv_heads,
        "ffn_mult": args.ffn_mult,
        "version": args.version,
        "fold": args.fold,
        "params": n_params,
        "gflops_per_sample_fwd": round(total_flops / 1e9, 3) if total_flops else None,
        "peak_mem_allocated_gb": round(peak_allocated, 3),
        "peak_mem_reserved_gb": round(peak_reserved, 3),
        "avg_epoch_s": round(train_seconds / max(epochs_run, 1), 2),
        "train_seconds": round(train_seconds, 1),
        "total_runtime_s": round(total_runtime, 1),
        "infer_throughput_samples_per_s": round(infer_throughput, 1),
        "infer_seconds": round(infer_s, 3),
        "infer_n": n_infer,
        "bf16_amp": amp,
        "grad_checkpointing": not args.no_grad_checkpoint,
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "early_stopped": early_stopped,
        "best_val_soc_loss": best_val,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "grad_clip": args.grad_clip,
        "warmup_frac": args.warmup_frac,
        "lr_schedule": args.lr_schedule,
        "target_transform": args.target_transform,
        "bin_soft_sigma": args.bin_soft_sigma,
        "bin_soft_rel": args.bin_soft_rel,
        "bin_weights": args.bin_weights,
        "seed": args.seed + args.fold,
    }
    write_json(run_dir / "metrics.json", {"report": report, "metrics": results})
    write_predictions(run_dir / "predictions.npz", dict(point_ids=point_ids[val_idx], **arrays))

    print(f"[{args.version} fold{args.fold}] validation metrics (physical units):")
    for t in task_names:
        print(format_task_metrics(t, results[t]))
    print(f"[{args.version} fold{args.fold}] REPORT " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
