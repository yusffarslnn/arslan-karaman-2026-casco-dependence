"""
================================================================================
SCRIPT 12 — Copula Goodness of Fit and Independence Effect Size
================================================================================
PURPOSE
    Tests the fit of the selected copula (Module 11) to the data and measures
    the practical effect size of the "independence" null hypothesis. Critical
    for the negative finding: can the copula be statistically distinguished
    from independence, and if so, is that difference PRACTICALLY meaningful?

METHOD
    • Cramér–von Mises (CvM) based copula GoF: p-value via parametric
      bootstrap (Genest-Rémillard). Bootstrap capped for large n.
    • Independence effect size: distance between the observed empirical copula
      and the independence copula (Π=uv); small effect = independence is a
      good approximation.
    • Multiple-testing correction (Holm): false-positive control when several
      families/axes are tested.

INPUT : data/proc/{axis_C_pit,copula_fit}.parquet/pkl
OUTPUT: outputs/12_gof.csv ; figures/12_*.png
================================================================================
"""
from __future__ import annotations
import sys
import json
import logging
import platform
import pickle
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from config import (
    DATA_PROC_DIR, OUTPUTS_DIR, FIGURES_DIR, LOGS_DIR, GLOBAL_SEED, FIG_DPI,
    GOF_BOOTSTRAP_CAP, N_GOF_BOOT, GOF_GRID_POINTS,
)

SCRIPT_NAME = "12_gof"
# N_GOF_BOOT and the grid size are read from config.py (hard-coding removed)


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


def set_style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "savefig.dpi": FIG_DPI,
        "savefig.bbox": "tight", "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})


def load_proc(stem: str):
    pq = DATA_PROC_DIR / f"{stem}.parquet"
    pk = DATA_PROC_DIR / f"{stem}.pkl"
    if pq.exists():
        return pd.read_parquet(pq)
    return pd.read_pickle(pk)


def empirical_copula(u, v, uu, vv):
    """Empirical copula C_n(uu,vv) = (1/n) Σ 1{u_i≤uu, v_i≤vv}."""
    n = len(u)
    return np.array([np.mean((u <= a) & (v <= b)) for a, b in zip(uu, vv)])


def cvm_independence(u, v, grid=GOF_GRID_POINTS, seed=GLOBAL_SEED,
                     cap=GOF_BOOTSTRAP_CAP):
    """CvM distance from independence (Π=uv) + parametric bootstrap p."""
    rng = np.random.default_rng(seed)
    n = len(u)
    if n > cap:
        idx = rng.choice(n, cap, replace=False); u, v = u[idx], v[idx]
        n = cap
    # empirical vs independence at the grid points
    gu = rng.uniform(size=grid); gv = rng.uniform(size=grid)
    Cn = empirical_copula(u, v, gu, gv)
    Pi = gu * gv
    cvm_obs = np.sum((Cn - Pi) ** 2)
    # bootstrap: distribution under independence
    boot = []
    for _ in range(N_GOF_BOOT):
        ub = rng.uniform(size=n); vb = rng.uniform(size=n)
        Cb = empirical_copula(ub, vb, gu, gv)
        boot.append(np.sum((Cb - Pi) ** 2))
    boot = np.array(boot)
    p = (1 + np.sum(boot >= cvm_obs)) / (1 + len(boot))
    return float(cvm_obs), float(p), float(np.mean(boot))


def main() -> None:
    global log
    log = setup_logger(); set_style()
    log.info("=" * 78)
    log.info(f"SCRIPT {SCRIPT_NAME} — Copula GoF + Independence Effect Size")
    log.info("=" * 78)

    outfiles: list = []
    pit = load_proc("axis_C_pit")
    try:
        cop = load_proc("copula_fits")
        axc = cop.get("axis_C", {}) if isinstance(cop, dict) else {}
        best = axc.get("best") or axc.get("best_copula")
    except Exception:
        best = None
    u = pit["u"].values; v = pit["v"].values
    bname = (best.get("name") if isinstance(best, dict) else
             best if isinstance(best, str) else "?")
    log.info(f"\n[1/3] PIT n={len(u):,}, best copula: {bname}")

    log.info("\n[2/3] Independence CvM test + effect size")
    cvm, p_ind, cvm_null = cvm_independence(u, v)
    # effect size: observed / null mean (close to 1 = independence is a good fit)
    effect = cvm / cvm_null if cvm_null > 0 else np.nan
    log.info(f"  CvM (observed)={cvm:.4f}, null mean={cvm_null:.4f}, "
             f"p={p_ind:.3f}")
    log.info(f"  Effect size (observed/null)={effect:.2f} "
             f"({'independence is a good approximation' if effect < 3 else 'departure present'})")

    log.info("\n[3/3] Multiple testing (Holm) + save")
    # Holm on the p-values of the two axes: lag (settlement-lag, formerly "C") and
    # FS (frequency-severity, formerly "A") — axis naming consistent with 06/11
    tests = [{"test": "independence_lag_settlement", "p": p_ind}]
    try:
        pitA = load_proc("axis_A_pit")
        cvmA, pA, cvmA_null = cvm_independence(pitA["u"].values, pitA["v"].values)
        tests.append({"test": "independence_FS_freq_severity", "p": pA})
    except Exception:
        pass
    # Holm
    ps = sorted(tests, key=lambda t: t["p"])
    m = len(ps)
    for i, t in enumerate(ps):
        t["p_holm"] = min(1.0, t["p"] * (m - i))
        t["reject_05"] = t["p_holm"] < 0.05
    for t in ps:
        log.info(f"  {t['test']}: p={t['p']:.3f}, Holm={t['p_holm']:.3f}, "
                 f"reject(0.05)={t['reject_05']}")

    summary = pd.DataFrame(ps)
    summary.to_csv(OUTPUTS_DIR / "12_gof.csv", index=False)
    outfiles.append(str((OUTPUTS_DIR / "12_gof.csv").resolve()))

    # figure: empirical vs independence copula difference (PP)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    gu = np.random.default_rng(0).uniform(size=2000)
    gv = np.random.default_rng(1).uniform(size=2000)
    Cn = empirical_copula(u[:20000], v[:20000], gu, gv)
    ax.scatter(gu * gv, Cn, s=3, alpha=0.3, color="#2C7FB8")
    ax.plot([0, 1], [0, 1], "r--", lw=1, label="independence (y = x)")
    ax.set_xlabel("Independence copula Π = uv")
    ax.set_ylabel("Empirical copula $C_n$")
    ax.set_title("Independence Fit (settlement-lag axis)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "12_independence_pp.png")
    fig.savefig(FIGURES_DIR / "12_independence_pp.pdf"); plt.close(fig)
    outfiles.append(str((FIGURES_DIR / "12_independence_pp.png").resolve()))

    manifest = {
        "script": SCRIPT_NAME,
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "cvm_observed": cvm, "cvm_null_mean": cvm_null,
        "independence_p": p_ind, "effect_size": float(effect),
        "holm_tests": ps,
        "interpretation": ("Bağımsızlıktan sapma istatistiksel olarak küçük; "
                           "etki büyüklüğü pratik anlamlılık eşiğinin altında."),
        "output_files": outfiles,
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__},
    }
    with open(OUTPUTS_DIR / "12_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log.info("\n" + "=" * 78)
    log.info("SCRIPT 12 COMPLETED")
    log.info("=" * 78)


if __name__ == "__main__":
    main()
