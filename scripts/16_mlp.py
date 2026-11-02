"""
================================================================================
SCRIPT 16 — MLP Baseline (Pure-NumPy Neural Network, Gini)
================================================================================
PURPOSE
    Flexible non-linear baseline (multi-layer perceptron); compared with the GLM
    and GBM on the same data/split/metric. Pure-numpy forward/back-propagation
    (no external ML dependency). The gradient has been verified by finite
    differences.

METHOD
    • Architecture: input → hidden (ReLU) → output (exp, Poisson mean).
    • Loss: Poisson deviance; full-batch gradient descent + momentum.
    • Training capped for large n; test on the full data.

INPUT : data/proc/analytical_dataset.parquet
OUTPUT: outputs/16_mlp.csv ; data/proc/mlp_preds.parquet
================================================================================
"""
from __future__ import annotations
import sys
import json
import logging
import platform
from pathlib import Path
from datetime import datetime

# --- Spyder-compatible path setup: look for config.py both in the same folder
# and in the parent folder (flat scripts/ layout AND src/ layout supported) ---
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
    MLP_HIDDEN, MLP_EPOCHS, MLP_LR, MLP_MOMENTUM, GINI_BOOTSTRAP_B,
)

SCRIPT_NAME = "16_mlp"
# Hyperparameters are read from config.py (hard-coding removed)
HIDDEN = MLP_HIDDEN
EPOCHS = MLP_EPOCHS
LR = MLP_LR
MOMENTUM = MLP_MOMENTUM


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


def build_features(df, edges=None):
    parts, names = [], []
    for c in NUMERIC_COVARS:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce").fillna(0).values.astype(float)
            parts.append(x.reshape(-1, 1)); names.append(c)
    for c in CATEG_COVARS:
        if c in df.columns:
            codes = pd.Categorical(df[c].astype(str)).codes.astype(float)
            parts.append(codes.reshape(-1, 1)); names.append(c)
    X = np.hstack(parts) if parts else np.zeros((len(df), 1))
    return X, names


def standardize(Xtr, Xte):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-9
    return (Xtr - mu) / sd, (Xte - mu) / sd


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
    log.info(f"SCRIPT {SCRIPT_NAME} — MLP Baseline (pure-numpy)")
    log.info("=" * 78)

    outfiles: list = []
    ana = load_proc("analytical_dataset")
    tr = ana[ana["policy_year"].isin(TRAIN_YEARS)].copy()
    te = ana[ana["policy_year"].isin(TEST_YEARS)].copy()
    rng = np.random.default_rng(GLOBAL_SEED)
    if TREE_TRAIN_CAP is not None and len(tr) > TREE_TRAIN_CAP:  # R2: None = no cap
        tr = tr.iloc[rng.choice(len(tr), TREE_TRAIN_CAP, replace=False)]
    log.info(f"\n[1/3] train={len(tr):,}, test={len(te):,}")

    Xtr, names = build_features(tr); Xte, _ = build_features(te)
    Xtr, Xte = standardize(Xtr, Xte)
    ytr = tr["n_claims"].values.astype(float)
    yte = te["n_claims"].values.astype(float)

    log.info("\n[2/3] MLP training (Poisson, momentum GD)")
    d = Xtr.shape[1]
    W1 = rng.normal(0, 0.1, (d, HIDDEN)); b1 = np.zeros(HIDDEN)
    W2 = rng.normal(0, 0.1, (HIDDEN, 1)); b2 = np.array([np.log(max(ytr.mean(), 1e-3))])
    vW1 = np.zeros_like(W1); vb1 = np.zeros_like(b1)
    vW2 = np.zeros_like(W2); vb2 = np.zeros_like(b2)
    n = len(ytr)
    for ep in range(EPOCHS):
        Z1 = Xtr @ W1 + b1; A1 = np.maximum(Z1, 0)
        eta = (A1 @ W2 + b2).ravel()
        mu = np.exp(np.clip(eta, -20, 20))
        # Poisson deviance gradient wrt eta: (mu - y)
        deta = (mu - ytr) / n
        dW2 = A1.T @ deta.reshape(-1, 1); db2 = np.array([deta.sum()])
        dA1 = deta.reshape(-1, 1) @ W2.T
        dZ1 = dA1 * (Z1 > 0)
        dW1 = Xtr.T @ dZ1; db1 = dZ1.sum(0)
        for p, g, v in [(W1, dW1, vW1), (b1, db1, vb1),
                        (W2, dW2, vW2), (b2, db2, vb2)]:
            v *= MOMENTUM; v -= LR * g; p += v
    # test
    A1t = np.maximum(Xte @ W1 + b1, 0)
    mu_te = np.exp(np.clip((A1t @ W2 + b2).ravel(), -20, 20))
    mu_c = np.clip(mu_te, 1e-9, None)
    dev = float(2 * np.sum(np.where(yte > 0, yte*np.log(yte/mu_c), 0.0) - (yte - mu_c)))
    log.info(f"  Test deviance = {dev:,.0f}")

    log.info("\n[3/3] Pure-premium + Gini")
    sev_mean = (tr.loc[tr["n_claims"] > 0, "ultimate_rbns"]
                / tr.loc[tr["n_claims"] > 0, "n_claims"]).mean()
    pp_pred = mu_te * sev_mean; pp_true = te["ultimate_rbns"].values
    g = gini(pp_true, pp_pred); lo, hi = gini_bootstrap(pp_true, pp_pred)
    log.info(f"  Pure-premium Gini = {g:.3f} [{lo:.3f}, {hi:.3f}]")

    pd.DataFrame({"policy_ref": te["policy_ref"].values, "freq_pred": mu_te,
                  "pp_pred": pp_pred, "pp_true": pp_true}).to_parquet(
        DATA_PROC_DIR / "mlp_preds.parquet", index=False)
    outfiles.append(str((DATA_PROC_DIR / "mlp_preds.parquet").resolve()))
    pd.DataFrame([{"model": "MLP", "test_deviance": dev, "pp_gini": g,
                   "gini_lo": lo, "gini_hi": hi, "n_test": len(te)}]).to_csv(
        OUTPUTS_DIR / "16_mlp.csv", index=False)
    outfiles.append(str((OUTPUTS_DIR / "16_mlp.csv").resolve()))

    manifest = {
        "script": SCRIPT_NAME,
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "test_deviance": dev, "pp_gini": g, "gini_ci": [lo, hi],
        "hyperparams": {"hidden": HIDDEN, "epochs": EPOCHS, "lr": LR},
        "output_files": outfiles,
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__},
    }
    with open(OUTPUTS_DIR / "16_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log.info("\n" + "=" * 78)
    log.info("SCRIPT 16 COMPLETED")
    log.info("=" * 78)


if __name__ == "__main__":
    main()
