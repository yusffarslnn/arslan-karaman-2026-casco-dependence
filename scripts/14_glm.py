"""
================================================================================
SCRIPT 14 — GLM Baseline (Frequency + Pure Premium, Gini)
================================================================================
PURPOSE
    Actuarial pricing baseline: frequency (Poisson/NegBin GLM) and pure-premium
    (Tweedie-like) prediction from the covariates. The subsequent copula
    augmentation (Section 17) is evaluated against this baseline. Discriminatory
    power is measured by the Gini, calibration by the test deviance.

METHOD
    • Time split: train (2014-2016) / test (2017-2018) — prevents future-information
      leakage.
    • Frequency GLM: log-link Poisson; deviance is reported on the test set.
    • Pure premium: frequency × mean severity (two-stage); Gini on the test
      set, with bootstrap CI.
    • Pure-numpy IRLS (no external dependency, easy to reproduce).

INPUT : data/proc/analytical_dataset.parquet
OUTPUT: outputs/14_glm.csv ; data/proc/glm_preds.parquet
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
# and in the parent folder (both the flat scripts/ layout AND the src/ layout are supported) ---
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
    NUMERIC_COVARS, CATEG_COVARS, PICKLE_PROTOCOL,
    GLM_N_ITER, GLM_RIDGE, GINI_BOOTSTRAP_B,
)

SCRIPT_NAME = "14_glm"


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


def build_design(df: pd.DataFrame, ref_columns=None):
    """Numeric (standardised) + categorical (one-hot) design matrix.

    Returns a pandas DataFrame; if ref_columns (the train columns) is given, the
    test set is aligned by reindex to the SAME columns/order (missing dummy=0,
    new levels dropped). This removes the risk of categorical column shift
    between train and test.
    """
    parts = {}
    parts["intercept"] = np.ones(len(df))
    for c in NUMERIC_COVARS:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce").fillna(0).values.astype(float)
            mu, sd = x.mean(), x.std() + 1e-9
            parts[c] = (x - mu) / sd
    base = pd.DataFrame(parts, index=df.index)
    cat_frames = []
    for c in CATEG_COVARS:
        if c in df.columns:
            d = pd.get_dummies(df[c].astype(str), prefix=c, drop_first=True)
            cat_frames.append(d.astype(float))
    X = pd.concat([base] + cat_frames, axis=1) if cat_frames else base
    if ref_columns is not None:
        X = X.reindex(columns=ref_columns, fill_value=0.0)
    return X


def poisson_irls(X, y, exposure, n_iter=GLM_N_ITER, ridge=GLM_RIDGE):
    """Pure-numpy Poisson GLM (log-link) IRLS."""
    beta = np.zeros(X.shape[1])
    log_e = np.log(np.clip(exposure, 1e-6, None))
    for _ in range(n_iter):
        eta = X @ beta + log_e
        mu = np.exp(np.clip(eta, -20, 20))
        W = mu
        z = (y - mu) / np.clip(mu, 1e-6, None)
        XtW = X.T * W
        H = XtW @ X + ridge * np.eye(X.shape[1])
        g = X.T @ (W * z)
        try:
            beta += np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
    return beta


def poisson_deviance(y, mu):
    mu = np.clip(mu, 1e-9, None)
    term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(2 * np.sum(term - (y - mu)))


def gini(y_true, y_pred):
    """Normalised (Lorenz) Gini: cumulative actuals ordered by prediction."""
    order = np.argsort(y_pred)
    y_sorted = np.asarray(y_true)[order]
    cum = np.cumsum(y_sorted) / (np.sum(y_sorted) + 1e-12)
    n = len(y_true)
    lorenz = np.sum(cum) / n
    g = 1 - 2 * lorenz
    # normalise: divide by the Gini of the perfect model
    order_p = np.argsort(y_true)
    y_perf = np.asarray(y_true)[order_p]
    cum_p = np.cumsum(y_perf) / (np.sum(y_perf) + 1e-12)
    g_perf = 1 - 2 * np.sum(cum_p) / n
    return float(g / g_perf) if g_perf != 0 else 0.0


def gini_bootstrap(y_true, y_pred, B=GINI_BOOTSTRAP_B, seed=GLOBAL_SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true); vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        vals.append(gini(y_true[idx], y_pred[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> None:
    global log
    log = setup_logger()
    log.info("=" * 78)
    log.info(f"SCRIPT {SCRIPT_NAME} — GLM Baseline")
    log.info("=" * 78)

    outfiles: list = []
    ana = load_proc("analytical_dataset")
    ana["exposure"] = pd.to_numeric(ana["exposure"], errors="coerce").fillna(1.0)
    ana.loc[ana["exposure"] <= 0, "exposure"] = 1.0

    tr = ana[ana["policy_year"].isin(TRAIN_YEARS)].copy()
    te = ana[ana["policy_year"].isin(TEST_YEARS)].copy()
    log.info(f"\n[1/3] Split: train={len(tr):,} ({TRAIN_YEARS}), "
             f"test={len(te):,} ({TEST_YEARS})")

    Xtr_df = build_design(tr)
    names = list(Xtr_df.columns)
    Xte_df = build_design(te, ref_columns=names)   # align to the train columns
    Xtr = Xtr_df.values.astype(float)
    Xte = Xte_df.values.astype(float)
    log.info(f"  Design matrix: {Xtr.shape[1]} columns (train/test aligned)")

    log.info("\n[2/3] Frequency GLM (Poisson IRLS)")
    ytr = tr["n_claims"].values.astype(float)
    yte = te["n_claims"].values.astype(float)
    beta = poisson_irls(Xtr, ytr, tr["exposure"].values)
    mu_te = np.exp(np.clip(Xte @ beta + np.log(te["exposure"].values), -20, 20))
    dev = poisson_deviance(yte, mu_te)
    log.info(f"  Test deviance = {dev:,.0f}")

    log.info("\n[3/3] Pure-premium + Gini")
    # mean severity (from train policies with claims)
    sev_mean = (tr.loc[tr["n_claims"] > 0, "ultimate_rbns"]
                / tr.loc[tr["n_claims"] > 0, "n_claims"]).mean()
    pp_pred = mu_te * sev_mean
    pp_true = te["ultimate_rbns"].values
    g = gini(pp_true, pp_pred)
    lo, hi = gini_bootstrap(pp_true, pp_pred)
    log.info(f"  Pure-premium Gini = {g:.3f} [{lo:.3f}, {hi:.3f}]")

    preds = pd.DataFrame({"policy_ref": te["policy_ref"].values,
                          "freq_pred": mu_te, "pp_pred": pp_pred,
                          "pp_true": pp_true})
    preds.to_parquet(DATA_PROC_DIR / "glm_preds.parquet", index=False)
    outfiles.append(str((DATA_PROC_DIR / "glm_preds.parquet").resolve()))

    summary = pd.DataFrame([{"model": "GLM", "test_deviance": dev,
                             "pp_gini": g, "gini_lo": lo, "gini_hi": hi,
                             "n_test": len(te)}])
    summary.to_csv(OUTPUTS_DIR / "14_glm.csv", index=False)
    outfiles.append(str((OUTPUTS_DIR / "14_glm.csv").resolve()))

    manifest = {
        "script": SCRIPT_NAME,
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "test_deviance": dev, "pp_gini": g, "gini_ci": [lo, hi],
        "n_train": len(tr), "n_test": len(te),
        "design_choices": {"time_split": True, "pure_numpy_irls": True},
        "output_files": outfiles,
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__},
    }
    with open(OUTPUTS_DIR / "14_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log.info("\n" + "=" * 78)
    log.info("SCRIPT 14 COMPLETED")
    log.info("=" * 78)


if __name__ == "__main__":
    main()
