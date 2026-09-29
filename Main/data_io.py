"""Data loading, splitting, target scaling and metrics for CrossSOC."""

import argparse
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans

from architecture import TASK_COLUMNS

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(data_dir: str):
    """Load the LUCAS 09 inner-processed dataset.

    Returns (spectra, wavelengths, point_ids, features) where spectra is a
    read-only float32 memmap (n, 4200) and features rows align with spectra rows.
    """
    spectra = np.load(f"{data_dir}/spectra.npy", mmap_mode="r")
    wavelengths = np.load(f"{data_dir}/wavelengths.npy")
    point_ids = np.load(f"{data_dir}/point_ids.npy")
    features = pd.read_csv(f"{data_dir}/features.csv")
    n = len(point_ids)
    if spectra.shape != (n, 4200) or wavelengths.shape != (4200,) or point_ids.ndim != 1:
        raise ValueError("expected spectra (N,4200), wavelengths (4200,), point_ids (N,)")
    if point_ids.dtype.kind not in "iu" or np.any(point_ids[1:] <= point_ids[:-1]):
        raise ValueError("POINT_ID must be unique integers sorted in ascending order")
    if len(features) != n or "POINT_ID" not in features or not np.array_equal(features.POINT_ID.to_numpy(), point_ids):
        raise ValueError("features POINT_ID rows must exactly match point_ids.npy order")
    required = ["OC_gkg"]
    if any(c not in features for c in required) or not np.isfinite(features[required].to_numpy(dtype=float)).all():
        raise ValueError("finite OC_gkg targets are required")
    if not np.isfinite(wavelengths).all() or np.any(np.diff(wavelengths) <= 0):
        raise ValueError("wavelengths must be finite and increasing")
    for start in range(0, n, 1024):
        x = spectra[start:start + 1024]
        if not np.isfinite(x).all() or (x < 0).any() or (x.std(axis=1) == 0).any():
            raise ValueError("spectra must be finite, nonnegative and nonconstant")
    return spectra, wavelengths, point_ids, features


def target_matrix(features: pd.DataFrame, task_names: Sequence[str]) -> np.ndarray:
    """(n, len(task_names)) float32 target array; NaN = missing label."""
    cols = [TASK_COLUMNS[t] for t in task_names]
    return features[cols].to_numpy(dtype=np.float32)


def load_folds(path: str, point_ids=None) -> dict:
    """Validate the persisted sample order and complete fold partition."""
    with np.load(path, allow_pickle=False) as data:
        folds = {k: data[k] for k in data.files}
    ids = folds.get("point_ids")
    if ids is None or ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError("fold point_ids must be unique and one-dimensional")
    if point_ids is not None and not np.array_equal(ids, point_ids):
        raise ValueError("fold POINT_ID order differs from data")
    keys = sorted(k for k in folds if k.startswith("fold_val_"))
    if set(keys) != {f"fold_val_{k}" for k in range(len(keys))} or len(keys) < 2:
        raise ValueError("fold keys must be consecutive fold_val_0..K-1")
    for key in keys:
        v = folds[key]
        if v.ndim != 1 or v.dtype.kind not in "iu" or not 0 < len(v) < len(ids):
            raise ValueError(f"invalid validation indices: {key}")
    all_val = np.concatenate([folds[k] for k in keys])
    if not np.array_equal(np.sort(all_val), np.arange(len(ids))):
        raise ValueError("folds must be disjoint and cover every sample exactly once")
    return folds


class TargetScaler:
    """NaN-aware z-score scaler fitted on train targets only."""

    def fit(self, y: np.ndarray) -> "TargetScaler":
        self.mean_ = np.nanmean(y, axis=0).astype(np.float32)
        std = np.nanstd(y, axis=0).astype(np.float32)
        self.std_ = np.where(std < 1e-8, np.float32(1.0), std)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return (y - self.mean_) / self.std_

    def inverse_transform(self, y: np.ndarray, col: int = None) -> np.ndarray:
        """Inverse-transform; ``col`` selects a single task column (a plain
        (n,) vs (n, n_tasks) broadcast would apply the wrong statistics)."""
        if col is None:
            return y * self.std_ + self.mean_
        return y * self.std_[col] + self.mean_[col]

    @classmethod
    def identity(cls, n_cols: int) -> "TargetScaler":
        """No-op scaler (mean 0, std 1) for heads that work in physical units,
        e.g. the binned distributional head."""
        s = cls()
        s.mean_ = np.zeros(n_cols, dtype=np.float32)
        s.std_ = np.ones(n_cols, dtype=np.float32)
        return s


class LUCASSpectrumDataset(Dataset):
    """(spectrum, target-vector) pairs backed by a float32 memmap.

    ``indices`` maps dataset position i -> row in the FULL spectra array.
    It must be passed whenever ``targets`` was built by indexing
    (e.g. ``y_norm[train_idx]``); otherwise position i would pair
    ``spectra[i]`` with the target of a *different* sample.
    """

    def __init__(self, spectra: np.ndarray, targets: np.ndarray,
                 indices: np.ndarray = None):
        self.spectra = spectra
        self.targets = np.ascontiguousarray(targets, dtype=np.float32)
        if self.targets.ndim == 1:
            self.targets = self.targets[:, None]
        self.indices = (np.arange(len(self.targets)) if indices is None
                        else np.asarray(indices))

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = np.array(self.spectra[self.indices[i]], dtype=np.float32)  # copy out of the memmap
        return torch.from_numpy(x), torch.from_numpy(self.targets[i])


# ---------------------------------------------------------------------------
# Metrics (computed in physical units, after inverse-transform)
# ---------------------------------------------------------------------------
def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ok = ~np.isnan(y_true)
    y_true, y_pred = y_true[ok], y_pred[ok]
    if len(y_true) == 0:
        nan = float("nan")
        return {"r2": nan, "rmse": nan, "mae": nan, "pearson": nan, "spearman": nan, "n": 0}
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    pearson = spearman = float("nan")
    if len(y_true) > 1 and y_true.std() > 0 and y_pred.std() > 0:
        try:
            pearson = float(pearsonr(y_true, y_pred)[0])
            spearman = float(spearmanr(y_true, y_pred)[0])
        except ValueError:
            pass
    return {
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "pearson": pearson,
        "spearman": spearman,
        "n": int(ok.sum()),
    }


def format_task_metrics(task: str, m: Dict[str, float]) -> str:
    return (f"    {task:<6s} R2={m['r2']:8.4f}  RMSE={m['rmse']:9.4f}  MAE={m['mae']:9.4f}  "
            f"Pearson={m['pearson']:7.4f}  Spearman={m['spearman']:7.4f}  (n={m['n']})")

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--kind', choices=['random', 'spatial'], default='random')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-splits', type=int, default=5)
    a = p.parse_args(argv)
    _, _, ids, features = load_data(a.data_dir)
    if not 2 <= a.n_splits <= len(ids):
        raise ValueError('n-splits must be between 2 and N')
    if a.kind == 'random':
        vals = [v for _, v in KFold(a.n_splits, shuffle=True, random_state=a.seed).split(ids)]
    else:
        xy = features[['x_laea', 'y_laea']].to_numpy(dtype=float)
        if not np.isfinite(xy).all() or (xy.std(axis=0) == 0).any():
            raise ValueError('finite LAEA coordinates with nonzero variance are required')
        labels = KMeans(n_clusters=a.n_splits, n_init=10, random_state=a.seed).fit_predict((xy-xy.mean(axis=0))/xy.std(axis=0))
        vals = [np.flatnonzero(labels == k) for k in range(a.n_splits)]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, point_ids=ids, **{f'fold_val_{k}': v.astype(np.int64) for k,v in enumerate(vals)})
    load_folds(out, ids)
    print(f'Saved {a.kind} folds to {out}; validation sizes: {[len(v) for v in vals]}')


def write_json(path, value):
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_predictions(path, arrays):
    np.savez_compressed(path, **arrays)


def save_checkpoint(value, path):
    torch.save(value, path)


def load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=False)


def write_history(path, row, append=False):
    import csv
    with open(path, "a" if append else "w", newline="") as handle:
        csv.writer(handle).writerow(row)


if __name__ == '__main__':
    main()
