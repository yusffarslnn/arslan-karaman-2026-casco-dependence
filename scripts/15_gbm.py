"""
================================================================================
SCRIPT 15 — GBM Baseline (Pure-NumPy Histogram Gradient Boosting, Gini)
================================================================================
PURPOSE
    Gradient boosting baseline that captures non-linear interactions;
    compared with the GLM (Module 14) on the same data/split/metric.
    Pure-numpy histogram-based implementation (no external ML dependency →
    full reproducibility).

METHOD
    • Gradient/hessian boosting for the Poisson deviance loss.
    • Histogram binning (fast split search); shallow regression trees.
    • Training on a TREE_TRAIN_CAP sub-sample for large n (speed); TEST on
      the full data (fair Gini/deviance).

INPUT : data/proc/analytical_dataset.parquet
OUTPUT: outputs/15_gbm.csv ; data/proc/gbm_preds.parquet
================================================================================
"""
from __future__ import annotations
import sys
import json
import logging
import platform
from pathlib import Path
from datetime import datetime

# --- Spyder-compatible path setup: look for config.py both in the same
# folder and in the parent folder (flat scripts/ layout AND src/ layout supported) ---
try:
    _HERE = Path(__file__).resolve().parent
except NameError:
    _HERE = Path.cwd()  # working directory if __file__ is undefined in Spyder
for _cand in (_HERE, _HERE.parent):
    if (_cand / "config.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

import numpy as np
import pandas as pd

from config import (
    DATA_PROC_DIR, OUTPUTS_DIR, LOGS_DIR, GLOBAL_SEED, TRAIN_YEARS, TEST_YEARS,
    NUMERIC_COVARS, CATEG_COVARS, TREE_TRAIN_CAP,
    GBM_N_TREES, GBM_MAX_DEPTH, GBM_LR, GBM_N_BINS, GINI_BOOTSTRAP_B,
)

SCRIPT_NAME = "15_gbm"
# Hyperparameters are read from config.py (hardcoding removed)
N_TREES = GBM_N_TREES
MAX_DEPTH = GBM_MAX_DEPTH
LR = GBM_LR
N_BINS = GBM_N_BINS


def setup_logger() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(SCRIPT_NAME)
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOGS_DIR / f"{SCRIPT_NAME}.log", mode="w",
                             encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger


log = logging.getLogger(SCRIPT_NAME)


def load_proc(stem: str) -> pd.DataFrame:
    pq = DATA_PROC_DIR / f"{stem}.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    return pd.read_pickle(DATA_PROC_DIR / f"{stem}.pkl")


def build_features(df):
    parts, names = [], []
    for c in NUMERIC_COVARS:
        if c in df.columns:
            parts.append(pd.to_numeric(df[c], errors="coerce").fillna(0).values
                         .astype(float).reshape(-1, 1))
            names.append(c)
    for c in CATEG_COVARS:
        if c in df.columns:
            codes = pd.Categorical(df[c].astype(str)).codes.astype(float)
            parts.append(codes.reshape(-1, 1)); names.append(c)
    return (np.hstack(parts) if parts else np.zeros((len(df), 1))), names


def bin_features(X, n_bins=N_BINS):
    """Quantile binning per column; returns bin indices."""
    Xb = np.zeros_like(X, dtype=np.int16)
    edges = []
    for j in range(X.shape[1]):
        q = np.quantile(X[:, j], np.linspace(0, 1, n_bins + 1)[1:-1])
        q = np.unique(q)
        Xb[:, j] = np.digitize(X[:, j], q)
        edges.append(q)
    return Xb, edges


def apply_bins(X, edges):
    Xb = np.zeros_like(X, dtype=np.int16)
    for j in range(X.shape[1]):
        Xb[:, j] = np.digitize(X[:, j], edges[j])
    return Xb


class TreeNode:
    __slots__ = ("feat", "thr", "left", "right", "val")
    def __init__(self):
        self.feat = -1; self.thr = 0; self.left = None; self.right = None
        self.val = 0.0


def fit_tree(Xb, grad, hess, depth, max_depth, n_bins, lam=1.0):
    node = TreeNode()
    G, H = grad.sum(), hess.sum()
    node.val = -G / (H + lam)
    if depth >= max_depth or len(grad) < 50:
        return node
    best_gain, best_f, best_t = 0.0, -1, 0
    for f in range(Xb.shape[1]):
        for t in range(n_bins):
            mask = Xb[:, f] <= t
            if mask.sum() < 20 or (~mask).sum() < 20:
                continue
            GL, HL = grad[mask].sum(), hess[mask].sum()
            GR, HR = G - GL, H - HL
            gain = GL**2/(HL+lam) + GR**2/(HR+lam) - G**2/(H+lam)
            if gain > best_gain:
                best_gain, best_f, best_t = gain, f, t
    if best_f < 0:
        return node
    node.feat, node.thr = best_f, best_t
    mask = Xb[:, best_f] <= best_t
    node.left = fit_tree(Xb[mask], grad[mask], hess[mask], depth+1, max_depth, n_bins, lam)
    node.right = fit_tree(Xb[~mask], grad[~mask], hess[~mask], depth+1, max_depth, n_bins, lam)
    return node


def predict_tree(node, Xb):
    out = np.empty(len(Xb))
    def rec(n, idx):
        if n.feat < 0 or n.left is None:
            out[idx] = n.val; return
        m = Xb[idx, n.feat] <= n.thr
        rec(n.left, idx[m]); rec(n.right, idx[~m])
    rec(node, np.arange(len(Xb)))
    return out


def gini(y_true, y_pred):
    order = np.argsort(y_pred); ys = np.asarray(y_true)[order]
    cum = np.cumsum(ys) / (np.sum(ys) + 1e-12); n = len(y_true)
    g = 1 - 2 * np.sum(cum) / n
    op = np.argsort(y_true); yp = np.asarray(y_true)[op]
    cp = np.cumsum(yp) / (np.sum(yp) + 1e-12)
    gp = 1 - 2 * np.sum(cp) / n
    return float(g / gp) if gp != 0 else 0.0


def gini_bootstrap(yt, yp, B=GINI_BOOTSTRAP_B, seed=GLOBAL_SEED):
    rng = np.random.default_rng(seed); n = len(yt); v = []
    for _ in range(B):
        idx = rng.integers(0, n, n); v.append(gini(yt[idx], yp[idx]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main() -> None:
    global log
    log = setup_logger()
    log.info("=" * 78)
    log.info(f"SCRIPT {SCRIPT_NAME} — GBM Baseline (pure-numpy)")
    log.info("=" * 78)

    outfiles: list = []
    ana = load_proc("analytical_dataset")
    ana["exposure"] = pd.to_numeric(ana["exposure"], errors="coerce").fillna(1.0)
    tr = ana[ana["policy_year"].isin(TRAIN_YEARS)].copy()
    te = ana[ana["policy_year"].isin(TEST_YEARS)].copy()

    # training cap (speed); test on full data
    rng = np.random.default_rng(GLOBAL_SEED)
    if TREE_TRAIN_CAP is not None and len(tr) > TREE_TRAIN_CAP:  # R2: None = no cap
        tr = tr.iloc[rng.choice(len(tr), TREE_TRAIN_CAP, replace=False)]
    log.info(f"\n[1/3] train={len(tr):,} (cap), test={len(te):,}")

    Xtr, names = build_features(tr)
    Xte, _ = build_features(te)
    Xb_tr, edges = bin_features(Xtr)
    Xb_te = apply_bins(Xte, edges)

    log.info("\n[2/3] Poisson GBM training")
    ytr = tr["n_claims"].values.astype(float)
    yte = te["n_claims"].values.astype(float)
    base = np.log(max(ytr.mean(), 1e-3))
    F_tr = np.full(len(ytr), base); F_te = np.full(len(yte), base)
    trees = []
    for t in range(N_TREES):
        mu = np.exp(np.clip(F_tr, -20, 20))
        grad = mu - ytr        # Poisson gradient
        hess = mu              # Poisson hessian
        node = fit_tree(Xb_tr, grad, hess, 0, MAX_DEPTH, N_BINS)
        F_tr += LR * predict_tree(node, Xb_tr)
        F_te += LR * predict_tree(node, Xb_te)
        trees.append(node)
    mu_te = np.exp(np.clip(F_te, -20, 20))
    mu_te_c = np.clip(mu_te, 1e-9, None)
    dev = float(2 * np.sum(np.where(yte > 0, yte*np.log(yte/mu_te_c), 0.0) - (yte - mu_te_c)))
    log.info(f"  Test deviance = {dev:,.0f}")

    log.info("\n[3/3] Pure-premium + Gini")
    sev_mean = (tr.loc[tr["n_claims"] > 0, "ultimate_rbns"]
                / tr.loc[tr["n_claims"] > 0, "n_claims"]).mean()
    pp_pred = mu_te * sev_mean
    pp_true = te["ultimate_rbns"].values
    g = gini(pp_true, pp_pred); lo, hi = gini_bootstrap(pp_true, pp_pred)
    log.info(f"  Pure-premium Gini = {g:.3f} [{lo:.3f}, {hi:.3f}]")

    pd.DataFrame({"policy_ref": te["policy_ref"].values, "freq_pred": mu_te,
                  "pp_pred": pp_pred, "pp_true": pp_true}).to_parquet(
        DATA_PROC_DIR / "gbm_preds.parquet", index=False)
    outfiles.append(str((DATA_PROC_DIR / "gbm_preds.parquet").resolve()))
    pd.DataFrame([{"model": "GBM", "test_deviance": dev, "pp_gini": g,
                   "gini_lo": lo, "gini_hi": hi, "n_test": len(te)}]).to_csv(
        OUTPUTS_DIR / "15_gbm.csv", index=False)
    outfiles.append(str((OUTPUTS_DIR / "15_gbm.csv").resolve()))

    manifest = {
        "script": SCRIPT_NAME,
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "test_deviance": dev, "pp_gini": g, "gini_ci": [lo, hi],
        "hyperparams": {"n_trees": N_TREES, "max_depth": MAX_DEPTH, "lr": LR},
        "output_files": outfiles,
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__},
    }
    with open(OUTPUTS_DIR / "15_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log.info("\n" + "=" * 78)
    log.info("SCRIPT 15 COMPLETED")
    log.info("=" * 78)


if __name__ == "__main__":
    main()
