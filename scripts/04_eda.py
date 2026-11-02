"""
================================================================================
SCRIPT 04 — Descriptive Analysis and Exploration (EDA)
================================================================================
PURPOSE
    Descriptive statistics introducing the data and tail diagnostics: severity
    distribution, Hill tail index (multi-k), yearly Loss Ratio (HPO, valuation-
    date based), and a PRE-TEST of frequency-severity dependence. This pre-test
    deliberately places THREE measurements side by side (mechanical / ratio /
    correct) so that the reader is prepared for the bias discussion in Section 5.

DESIGN DECISIONS
    • The Hill index is reported on a multi-k grid, NOT at a SINGLE k
      (a single-k choice may mask the tail regime).
    • HPO uses the valuation-date (2020-12) snapshot definition: cumulative paid +
      case reserves open at that date / real premium. NOT a Chain-Ladder projection.
    • The frequency-severity pre-test explicitly labels three measurements:
        – MECHANICAL (N vs ΣY): arithmetic tautology, NOT dependence
        – RATIO (N vs ΣY/N): common-denominator fallacy
        – CORRECT (per-claim severity, by N groups): the genuine pre-test
      The definitive verdict is deferred to Sections 6-7 (conditional analysis).

INPUT : data/proc/{analytical_dataset,claim_analytical,paid_long_real,
        reserve_panel_real}.parquet ; data/raw/cpi_monthly.*
OUTPUT: outputs/04_*.csv ; figures/04_*.png
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from config import (
    DATA_PROC_DIR, DATA_RAW_DIR, OUTPUTS_DIR, FIGURES_DIR, LOGS_DIR,
    CPI_SHEET_CANDIDATES, CPI_CANDIDATES, INFLATION_BASE_YEAR_MONTH,
    HPO_VALUATION_YEAR, HPO_VALUATION_MONTH, CURRENCY, FIG_DPI, HILL_K_GRID,
)

SCRIPT_NAME = "04_eda"
# HILL_K_GRID is now read from config.py (hard-coding removed)


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


def load_proc(stem: str) -> pd.DataFrame:
    pq = DATA_PROC_DIR / f"{stem}.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    return pd.read_pickle(DATA_PROC_DIR / f"{stem}.pkl")


def load_cpi() -> pd.DataFrame:
    for cand in CPI_CANDIDATES:
        p = DATA_RAW_DIR / cand
        if p.exists():
            if p.suffix.lower() in (".xlsx", ".xls"):
                xl = pd.ExcelFile(p)
                for sh in CPI_SHEET_CANDIDATES + xl.sheet_names:
                    if sh in xl.sheet_names:
                        d = pd.read_excel(p, sheet_name=sh)
                        if {"year_month", "cpi_index"}.issubset(d.columns):
                            break
            else:
                d = pd.read_csv(p)
            d = d.dropna(subset=["year_month"]).copy()
            if not pd.api.types.is_string_dtype(d["year_month"]):
                d["year_month"] = pd.to_datetime(
                    d["year_month"], errors="coerce").dt.strftime("%Y-%m")
            dt = pd.to_datetime(d["year_month"], format="%Y-%m")
            d["ym_key"] = (dt.dt.year * 100 + dt.dt.month).astype(int)
            d["cpi_index"] = pd.to_numeric(d["cpi_index"], errors="coerce")
            return d.dropna(subset=["cpi_index"])
    raise FileNotFoundError("CPI yok")


# =========================================================
# Hill tail index (multi-k)
# =========================================================
def hill_estimator(x: np.ndarray, k: int) -> float:
    xs = np.sort(x)[::-1]
    if k >= len(xs) or xs[k] <= 0:
        return np.nan
    logs = np.log(xs[:k]) - np.log(xs[k])
    alpha = 1.0 / np.mean(logs)
    return float(alpha)


# =========================================================
# Yearly HPO (valuation-date based)
# =========================================================
def compute_hpo(ana, paid, reserve, cpi) -> pd.DataFrame:
    prim_col = "Toplam_Prim_TL" if "Toplam_Prim_TL" in ana.columns else "Prim_TL"
    base_cpi = float(cpi.loc[cpi["ym_key"] == INFLATION_BASE_YEAR_MONTH,
                             "cpi_index"].iloc[0])
    yearly_cpi = cpi.assign(yr=cpi["ym_key"] // 100).groupby("yr")["cpi_index"].mean()
    ana = ana.copy()
    ana[prim_col] = pd.to_numeric(ana[prim_col], errors="coerce").fillna(0.0)
    ycpi = ana["policy_year"].astype(int).map(yearly_cpi).fillna(base_cpi)
    ana["prim_real"] = ana[prim_col].values * base_cpi / ycpi.values

    vy, vm = HPO_VALUATION_YEAR, HPO_VALUATION_MONTH
    paid_cut = paid[(paid["payment_year"] < vy)
                    | ((paid["payment_year"] == vy)
                       & (paid["payment_month"] <= vm))]
    paid_by = paid_cut.groupby("policy_ref")["paid_total_real"].sum()
    res_cut = reserve[(reserve["reserve_year"] < vy)
                      | ((reserve["reserve_year"] == vy)
                         & (reserve["reserve_month"] <= vm))].copy()
    res_cut = res_cut.sort_values(["reserve_year", "reserve_month"])
    last_res = (res_cut.groupby("claim_file").tail(1)
                .groupby("policy_ref")["reserve_amount_real"].sum())
    inc = pd.DataFrame({"paid": paid_by}).join(
        pd.DataFrame({"res": last_res}), how="outer").fillna(0.0)
    inc["incurred"] = inc["paid"] + inc["res"]; inc.index = inc.index.astype(str)
    ana["policy_ref"] = ana["policy_ref"].astype(str)
    ana = ana.merge(inc, left_on="policy_ref", right_index=True, how="left")
    for c in ["paid", "res", "incurred"]:
        ana[c] = ana[c].fillna(0.0)
    g = ana.groupby("policy_year").agg(
        prim=("prim_real", "sum"), paid=("paid", "sum"),
        incurred=("incurred", "sum"), reserve=("res", "sum")).reset_index()
    g = g[g["prim"] > 0]
    g["HPO_paid"] = g["paid"] / g["prim"]
    g["HPO_incurred"] = g["incurred"] / g["prim"]
    return g


def main() -> None:
    global log
    log = setup_logger(); set_style()
    log.info("=" * 78)
    log.info(f"SCRIPT {SCRIPT_NAME} — Descriptive Analysis (EDA)")
    log.info("=" * 78)

    outfiles: list = []
    ana = load_proc("analytical_dataset")
    claim = load_proc("claim_analytical")
    paid = load_proc("paid_long_real")
    reserve = load_proc("reserve_panel_real")
    cpi = load_cpi()

    log.info("\n[1/5] Severity descriptives + Hill multi-k")
    sev = claim["severity_real"].values
    sev = sev[sev > 0]
    log.info(f"  Per-claim severity: n={len(sev):,}, "
             f"median={np.median(sev):,.0f}, mean={sev.mean():,.0f}")
    hill_rows = []
    for k in HILL_K_GRID:
        a = hill_estimator(sev, k)
        hill_rows.append({"k": k, "alpha_hat": a,
                          "tail_regime": "ağır (α<2)" if a < 2 else "ılımlı"})
    hill_df = pd.DataFrame(hill_rows)
    log.info(f"  Hill α̂ range: "
             f"[{hill_df['alpha_hat'].min():.2f}, {hill_df['alpha_hat'].max():.2f}]")

    log.info("\n[2/5] Severity distribution fit (AIC)")
    fits = {}
    for name, dist in [("LogNormal", stats.lognorm), ("Gamma", stats.gamma),
                       ("Weibull", stats.weibull_min)]:
        try:
            par = dist.fit(sev, floc=0)
            ll = np.sum(dist.logpdf(sev, *par))
            aic = 2 * len(par) - 2 * ll
            fits[name] = {"params": par, "aic": aic}
        except Exception:
            continue
    best_sev = min(fits, key=lambda n: fits[n]["aic"])
    log.info(f"  Best severity distribution (AIC): {best_sev}")

    log.info("\n[3/5] Frequency descriptives")
    nclaims = ana["n_claims"].values
    disp = nclaims.var() / nclaims.mean() if nclaims.mean() > 0 else np.nan
    zero_frac = (nclaims == 0).mean()
    log.info(f"  Frequency: mean={nclaims.mean():.4f}, dispersion={disp:.3f} "
             f"({'overdispersed→NegBin' if disp>1.1 else 'Poisson-consistent'}), "
             f"zeros={zero_frac:.3f}")

    log.info("\n[4/5] Frequency-severity PRE-TEST (three measurements side by side)")
    # MECHANICAL: N vs total ultimate
    m = ana["n_claims"] > 0
    tau_mech = stats.kendalltau(ana.loc[m, "n_claims"],
                                ana.loc[m, "ultimate_rbns"])[0]
    # RATIO: N vs average (ult/N)
    avg = ana.loc[m, "ultimate_rbns"] / ana.loc[m, "n_claims"]
    tau_ratio = stats.kendalltau(ana.loc[m, "n_claims"], avg)[0]
    log.info(f"  MECHANICAL (N vs ΣY)  τ={tau_mech:+.4f}  [tautology — not dependence]")
    log.info(f"  RATIO   (N vs ΣY/N)   τ={tau_ratio:+.4f}  [common-denominator fallacy]")
    log.info(f"  → The CORRECT pre-test is the distribution of per-claim severity "
             f"by N groups; the definitive verdict is deferred to the Sections 6-7 conditional analysis.")

    log.info("\n[5/5] Yearly HPO + figures")
    hpo = compute_hpo(ana, paid, reserve, cpi)
    for _, r in hpo.iterrows():
        log.info(f"  {int(r['policy_year'])}: HPO_paid={r['HPO_paid']:.3f}, "
                 f"HPO_incurred={r['HPO_incurred']:.3f}")
    hpo.to_csv(OUTPUTS_DIR / "04_yearly_hpo.csv", index=False)
    hill_df.to_csv(OUTPUTS_DIR / "04_hill.csv", index=False)
    outfiles += [str((OUTPUTS_DIR / f"04_{n}.csv").resolve())
                 for n in ["yearly_hpo", "hill"]]

    # HPO figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    years = hpo["policy_year"].values; x = np.arange(len(years)); w = 0.38
    ax1.axhspan(0.50, 0.95, color="#C7E9C0", alpha=0.4, label="Expected range")
    ax1.axhline(1.0, color="#D7301F", ls="--", lw=1, label="100% limit")
    ax1.bar(x - w/2, hpo["HPO_incurred"], w, color="#3182BD",
            label=f"HPO (incurred @ {HPO_VALUATION_YEAR}-{HPO_VALUATION_MONTH:02d})")
    ax1.bar(x + w/2, hpo["HPO_paid"], w, color="#41AB5D", label="HPO (paid only)")
    ax1.set_xticks(x); ax1.set_xticklabels([int(y) for y in years])
    ax1.set_xlabel("Policy year"); ax1.set_ylabel("HPO")
    ax1.set_title("Annual HPO — motor casco (TR)"); ax1.legend(fontsize=7)
    sc = 1e9
    ax2.plot(years, hpo["prim"]/sc, "o-", color="#3182BD", label="Premium (bn TRY)")
    ax2.plot(years, hpo["paid"]/sc, "s-", color="#D7301F", label="Ult. paid")
    ax2.plot(years, hpo["incurred"]/sc, "^--", color="#FD8D3C", label="Ult. incurred")
    ax2.set_xlabel("Policy year"); ax2.set_ylabel("Billion TRY (real)")
    ax2.set_title("Annual premium and ultimate"); ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_yearly_hpo.png")
    fig.savefig(FIGURES_DIR / "04_yearly_hpo.pdf"); plt.close(fig)
    outfiles.append(str((FIGURES_DIR / "04_yearly_hpo.png").resolve()))

    # Hill figure
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(hill_df["k"], hill_df["alpha_hat"], "o-", color="#2C7FB8")
    ax.axhline(2.0, color="#D7301F", ls="--", lw=1, label="α=2 (variance boundary)")
    ax.set_xscale("log"); ax.set_xlabel("k (upper order statistics)")
    ax.set_ylabel("Hill α̂"); ax.set_title("Hill tail index (multiple k)")
    ax.legend(fontsize=8); fig.savefig(FIGURES_DIR / "04_hill.png")
    plt.close(fig)
    outfiles.append(str((FIGURES_DIR / "04_hill.png").resolve()))

    manifest = {
        "script": SCRIPT_NAME,
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "severity": {"n": int(len(sev)), "median": float(np.median(sev)),
                     "best_dist_aic": best_sev,
                     "hill_alpha_range": [float(hill_df["alpha_hat"].min()),
                                          float(hill_df["alpha_hat"].max())]},
        "frequency": {"mean": float(nclaims.mean()), "dispersion": float(disp),
                      "zero_frac": float(zero_frac)},
        "pretest_tau": {"mechanical": float(tau_mech), "ratio": float(tau_ratio),
                        "note": "clean verdict deferred to conditional analysis"},
        "hpo": hpo.to_dict(orient="records"),
        "output_files": outfiles,
        "environment": {"python": platform.python_version(),
                        "pandas": pd.__version__, "numpy": np.__version__},
    }
    with open(OUTPUTS_DIR / "04_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log.info("\n" + "=" * 78)
    log.info("SCRIPT 04 COMPLETED")
    log.info("=" * 78)


if __name__ == "__main__":
    main()
