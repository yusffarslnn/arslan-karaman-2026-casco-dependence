"""
================================================================================
SCRIPT 06 — Dependence Diagnostics (Five Measurement Axes)
================================================================================
PURPOSE
    Measures dependence free of the biases characterised in Section 5.
    Two different kinds of dependence are separated conceptually:

    Axis A (MECHANICAL) : N vs ΣY — tautology, reported as a reference
    Axis B (RATIO)      : N vs ΣY/N — fallacy, reported as a reference
    Axis FS (CLEAN)     : policy N × per-claim severity — the TRUE frequency-
                          severity signal (for pricing and compound capital); BCa CI.
    Axis FS-adj         : log(sev) ~ N + accident-year covariate-adjusted direct test
                          (Garrido-Genest-Schulz 2016 type), policy-cluster boot.
    Axis lag (CLEAN)    : payment delay × per-claim severity — SETTLEMENT-LAG
                          association (for reserve risk); SEPARATE from frequency-severity.

DESIGN DECISIONS
    • Naive (A, B) and clean (FS, lag) axes are reported together so that the
      reader sees the discrepancy — the empirical verification of Section 5.
    • BCa (bias-corrected accelerated) bootstrap CI for FS and lag.
    • Frequency-severity (FS) feeds 11_copulas/13_capital; settlement-lag feeds
      the reserving discussion. Delay = payment year − accident year; severity = real paid.

INPUT : data/proc/paid_long_real.parquet
OUTPUT: outputs/06_dependence_summary.csv ; figures/06_*.png
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
# and in the parent folder (flat scripts/ layout AND src/ layout are supported) ---
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
    TAU_N_BOOTSTRAP, TAU_BCA_CAP, AXIS_D_N_BOOTSTRAP, TAU_CLUSTER_N_BOOTSTRAP,
    NUMERIC_COVARS, CATEG_COVARS,
    apply_pub_style, FIG_W_FULL, env_info,
)

SCRIPT_NAME = "06_dependence"

# --- R1 revision constants ---
# Kendall τ: SciPy stats.kendalltau, tie-corrected τ-b variant (SciPy
# default; fixed EXPLICITLY here). All reported τ estimates and bootstrap
# replicates are computed on the RAW (tied) data — jitter is NOT APPLIED
# anywhere in this module. The explicit 'variant="b"' does not change the
# results (identical to the default).
KENDALL_VARIANT = "b"
# Figure 2 (06_three_axes): two-panel, full width (FIG_W_FULL).
# Colours: slots 1 (blue) and 2 (orange) of the dataviz reference categorical
# palette — naive constructions orange, clean (claim-level) axes blue.
FIG2_H = 2.6
FIG2_COLOR_NAIVE = "#eb6834"
FIG2_COLOR_CLEAN = "#2a78d6"
MINUS = "\u2212"   # typographic minus sign (U+2212)


def ktau(x, y) -> float:
    """Kendall τ-b (SciPy), raw data, no jitter. Returns the statistic only."""
    return float(stats.kendalltau(x, y, variant=KENDALL_VARIANT)[0])


def fmt_signed(v: float, nd: int = 4) -> str:
    """Signed number label with a proper minus sign (−0.0065, +0.0185)."""
    return f"{v:+.{nd}f}".replace("-", MINUS)
# Bootstrap parameters are read from config.py (hardcoding removed)
N_BOOTSTRAP = TAU_N_BOOTSTRAP
BCA_CAP = TAU_BCA_CAP   # subsample for the BCa jackknife (n² jackknife is expensive)


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


def bca_bootstrap_tau(x, y, B=N_BOOTSTRAP, alpha=0.05, seed=GLOBAL_SEED):
    """BCa bootstrap CI for Kendall tau."""
    rng = np.random.default_rng(seed)
    n = len(x)
    # point estimate
    tau_hat = ktau(x, y)
    # bootstrap distribution
    boots = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        boots[b] = ktau(x[idx], y[idx])
    # bias-correction z0
    z0 = stats.norm.ppf((np.sum(boots < tau_hat) + 0.5) / (B + 1))
    # acceleration (jackknife; subsample for large n)
    if n > BCA_CAP:
        sub = rng.choice(n, BCA_CAP, replace=False)
        xj, yj = x[sub], y[sub]
    else:
        xj, yj = x, y
    nj = len(xj)
    # sub-block approximation instead of the jackknife (speed): 200 slices
    n_blocks = min(200, nj)
    blocks = np.array_split(rng.permutation(nj), n_blocks)
    jack = np.array([ktau(np.delete(xj, blk), np.delete(yj, blk))
                     for blk in blocks])
    jbar = jack.mean()
    num = np.sum((jbar - jack) ** 3)
    den = 6 * (np.sum((jbar - jack) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0
    # adjusted percentiles
    z_a = stats.norm.ppf(alpha / 2); z_1a = stats.norm.ppf(1 - alpha / 2)
    def adj(z):
        return stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    lo = np.percentile(boots, 100 * adj(z_a))
    hi = np.percentile(boots, 100 * adj(z_1a))
    return float(tau_hat), float(lo), float(hi)


def cluster_bootstrap_tau(x, y, cluster_ids, B=TAU_CLUSTER_N_BOOTSTRAP,
                          alpha=0.05, seed=GLOBAL_SEED):
    """POLICY-CLUSTER bootstrap percentile CI for Kendall τ.

    Claim-level observations are NOT independent within a policy (shared
    driver/vehicle risk); an iid bootstrap CI ignores the within-cluster
    correlation and may be too narrow. Here POLICIES are resampled with
    replacement; when a policy is drawn, all of its claims come with it
    (consistent with the cluster design of axis FS-adj)."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(cluster_ids)
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    starts = np.searchsorted(sorted_codes, np.arange(len(uniq)))
    ends = np.append(starts[1:], len(sorted_codes))
    boots = np.empty(B)
    for b in range(B):
        pol = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([order[starts[p]:ends[p]] for p in pol])
        boots[b] = ktau(x[idx], y[idx])
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def axis_d_direct_test(claim: pd.DataFrame, B=AXIS_D_N_BOOTSTRAP,
                       seed=GLOBAL_SEED):
    """Axis FS-adj — FULLY covariate-adjusted DIRECT frequency→severity test.

    Garrido-Genest-Schulz (2016) approach: at claim level, log(severity) ~
    constant + N_policy + accident-year FE + POLICY COVARIATES (vehicle age,
    insured age, horsepower, engine volume, vehicle value — standardised; sex
    and body-type dummies). The N coefficient measures the direct effect of
    frequency on severity (net of risk profile and accident year) — it is
    algebraically independent of the count-sum and ratio identities (full
    covariate set). Uncertainty via POLICY-CLUSTER bootstrap.

    Returns: dict(beta_N, ci, pct_effect_per_claim, tau_D, n, covariates)
    """
    df = claim.dropna(subset=["n_claims"]).copy()
    y = np.log(df["severity"].values.astype(float))
    nclaims = df["n_claims"].values.astype(float)
    # accident-year dummies (first year as reference)
    years = sorted(df["accident_year"].unique())
    dums = [(df["accident_year"] == yr).values.astype(float)
            for yr in years[1:]]
    # policy covariates: numerics standardised, categoricals as dummies.
    # log(exposure): the claim count depends on the exposure period; it enters
    # offset-like so that the coefficient of N can be read as exposure-adjusted.
    cov_cols, cov_names = [], []
    if "exposure" in df.columns:
        ex = pd.to_numeric(df["exposure"], errors="coerce").fillna(1.0)
        cov_cols.append(np.log(np.clip(ex.values.astype(float), 1e-6, None)))
        cov_names.append("log_exposure")
    for c in NUMERIC_COVARS:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce").fillna(0).values.astype(float)
            cov_cols.append((x - x.mean()) / (x.std() + 1e-9))
            cov_names.append(c)
    for c in CATEG_COVARS:
        if c in df.columns:
            d = pd.get_dummies(df[c].astype(str), prefix=c, drop_first=True)
            for col in d.columns:
                cov_cols.append(d[col].values.astype(float))
            cov_names.append(f"{c}({d.shape[1]} kukla)")
    X = np.column_stack([np.ones(len(df)), nclaims] + dums + cov_cols)

    def ols_beta(Xm, ym):
        try:
            return np.linalg.lstsq(Xm, ym, rcond=None)[0][1]
        except Exception:
            return np.nan

    beta = ols_beta(X, y)
    # policy-cluster bootstrap
    codes, uniq = pd.factorize(df["policy_ref"].values)
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    starts = np.searchsorted(sorted_codes, np.arange(len(uniq)))
    ends = np.append(starts[1:], len(sorted_codes))
    rng = np.random.default_rng(seed)
    betas = []
    for _ in range(B):
        pol = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([order[starts[p]:ends[p]] for p in pol])
        betas.append(ols_beta(X[idx], y[idx]))
    lo, hi = np.nanpercentile(betas, [2.5, 97.5])
    # NOTE: tau_D is the UNADJUSTED (no covariates) claim-level τ — identical
    # to the axis FS τ; the covariate-adjusted effect is beta_N only (on the
    # log-severity scale). The key is kept for backward compatibility.
    tau_D = ktau(nclaims, df["severity"].values)
    return {"beta_N": float(beta), "ci": [float(lo), float(hi)],
            "pct_effect_per_claim": float(np.exp(beta) - 1.0),
            "tau_D": float(tau_D),
            "tau_D_note": ("UNADJUSTED claim-level Kendall tau-b of N vs "
                           "per-claim severity (no covariates); identical to "
                           "axis_tau.FS_freq_severity. The covariate-adjusted "
                           "quantity is beta_N (log-severity OLS coefficient)."),
            "n": int(len(df)), "B": int(B),
            "ci_method": "policy-cluster bootstrap (percentile), policies resampled with all their claims",
            "significant": bool(not (lo <= 0.0 <= hi)),
            "covariates": ["accident_year FE"] + cov_names}


def draw_figure2(tau_A, tau_B, tau_FS, ci_FS, tau_lag, ci_lag,
                 stem: Path) -> dict:
    """Figure 2 (R1): naive vs clean dependence measures on the Kendall τ-b scale.

    (a) four measures on a common scale; (b) the two clean axes on an enlarged
    scale. Error bars ONLY on the axes with a policy-cluster bootstrap CI (FS,
    lag). The covariate-adjusted effect (β_N) is a log-severity regression
    coefficient — it is NOT PLOTTED since it is not on the τ scale (the
    pre-R1 'FS-adj' bar actually repeated the unadjusted FS τ)."""
    from matplotlib.patches import Patch

    labels = ["A: count ×\naggregate loss\n(naive, B1)",
              "B: count ×\nmean severity\n(naive, B2)",
              "FS: count ×\nper-claim\nseverity (clean)",
              "lag: settlement\nlag × severity\n(clean)"]
    vals = [float(tau_A), float(tau_B), float(tau_FS), float(tau_lag)]
    cis = [None, None, tuple(map(float, ci_FS)), tuple(map(float, ci_lag))]
    cols = [FIG2_COLOR_NAIVE, FIG2_COLOR_NAIVE,
            FIG2_COLOR_CLEAN, FIG2_COLOR_CLEAN]

    def panel(ax, idx, ylim, pad):
        eb_handle = None
        for j, i in enumerate(idx):
            v = vals[i]
            ax.bar(j, v, width=0.62, color=cols[i], edgecolor="none",
                   zorder=2)
            lo_i, hi_i = (cis[i] if cis[i] is not None else (v, v))
            if cis[i] is not None:
                eb_handle = ax.errorbar(
                    j, v, yerr=[[v - lo_i], [hi_i - v]], fmt="none",
                    ecolor="black", elinewidth=0.8, capsize=3,
                    capthick=0.8, zorder=3)
            if v >= 0:
                ax.text(j, max(v, hi_i) + pad, fmt_signed(v, 3 if abs(v) >= 0.05 else 4), ha="center",
                        va="bottom", fontsize=7)
            else:
                ax.text(j, min(v, lo_i) - pad, fmt_signed(v, 3 if abs(v) >= 0.05 else 4), ha="center",
                        va="top", fontsize=7)
        ax.axhline(0, color="0.3", lw=0.8, zorder=2.5)
        ax.set_xticks(np.arange(len(idx)))
        ax.set_xticklabels([labels[i] for i in idx])
        ax.tick_params(axis="x", length=0, pad=3)
        ax.set_xlim(-0.55, len(idx) - 0.45)
        ax.set_ylim(*ylim)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel(r"Kendall's $\tau_b$")
        return eb_handle

    with plt.rc_context({"mathtext.fontset": "dejavuserif"}):
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(FIG_W_FULL, FIG2_H), layout="constrained",
            gridspec_kw={"width_ratios": [2.35, 1.0]})
        fig.get_layout_engine().set(w_pad=0.04, wspace=0.06)
        eb = panel(ax1, [0, 1, 2, 3], (-0.05, 0.30), 0.006)
        ax1.set_title("(a) Naive constructions and clean axes", loc="left")
        handles = [Patch(color=FIG2_COLOR_NAIVE),
                   Patch(color=FIG2_COLOR_CLEAN)]
        leg_labels = ["Naive construction (policy level)",
                      "Clean axis (claim level)"]
        if eb is not None:
            handles.append(eb)
            leg_labels.append("Policy-cluster bootstrap 95% CI")
        ax1.legend(handles, leg_labels, loc="upper right", handlelength=1.2)
        panel(ax2, [2, 3], (-0.015, 0.028), 0.0008)
        ax2.set_title("(b) Clean axes, enlarged", loc="left")
        fig.savefig(stem.with_suffix(".pdf"))
        fig.savefig(stem.with_suffix(".png"), dpi=600)
        plt.close(fig)

    return {
        "file_stem": str(stem.name),
        "size_in": [float(FIG_W_FULL), float(FIG2_H)],
        "panels": {"a": "A, B, FS, lag on a common Kendall tau-b scale",
                   "b": "FS and lag on an enlarged scale"},
        "bars": {"A": "claim count x aggregate loss (policy level, B1)",
                 "B": "claim count x mean severity (policy level, B2)",
                 "FS": "claim count x per-claim severity (claim level)",
                 "lag": "settlement lag x severity (claim level)"},
        "bar_values": {"A": vals[0], "B": vals[1], "FS": vals[2],
                       "lag": vals[3]},
        "error_bars": ("FS and lag only: policy-cluster bootstrap percentile "
                       f"95% CI, B={TAU_CLUSTER_N_BOOTSTRAP}; no CI computed "
                       "for A and B"),
        "fs_adj_bar": ("removed in R1: the former 'FS-adj' bar plotted "
                       "axis_D_direct.tau_D, the unadjusted FS tau; beta_N "
                       "is a log-severity coefficient and is not on the "
                       "tau scale"),
        "value_labels": "4 decimals, U+2212 minus sign",
    }


def main() -> None:
    global log
    log = setup_logger(); set_style(); apply_pub_style()
    log.info("=" * 78)
    log.info(f"SCRIPT {SCRIPT_NAME} — Dependence Diagnostics (Five Measurement Axes)")
    log.info("=" * 78)

    outfiles: list = []
    paid = load_proc("paid_long_real")

    log.info("\n[1/4] Per-claim axis data (delay × severity)")
    # each claim file: total real paid (severity), mean delay
    claim = paid.groupby("claim_file").agg(
        policy_ref=("policy_ref", "first"),
        accident_year=("accident_year", "first"),
        severity=("paid_total_real", "sum"),
        delay=("dev_year", "mean")).reset_index()
    claim = claim[claim["severity"] > 0]
    # policy-level frequency
    freq = (paid.groupby("policy_ref")["claim_file"].nunique()
            .rename("n_claims").reset_index())
    claim = claim.merge(freq, on="policy_ref", how="left")
    # policy covariates (for the FS-adj full-covariate regression)
    ana = load_proc("analytical_dataset")
    keep = ["policy_ref", "exposure"] + [c for c in NUMERIC_COVARS
                                         + CATEG_COVARS if c in ana.columns]
    claim = claim.merge(ana[keep].drop_duplicates("policy_ref"),
                        on="policy_ref", how="left")
    log.info(f"  Claim level n={len(claim):,} (+{len(keep)-1} policy covariates)")

    log.info("\n[2/5] Axis A (mechanical) and B (ratio) — naive reference")
    pol = claim.groupby("policy_ref").agg(
        n_claims=("n_claims", "first"),
        total=("severity", "sum")).reset_index()
    m = pol["n_claims"] > 0
    tau_A = ktau(pol.loc[m, "n_claims"], pol.loc[m, "total"])
    avg = pol.loc[m, "total"] / pol.loc[m, "n_claims"]
    tau_B = ktau(pol.loc[m, "n_claims"], avg)
    log.info(f"  Axis A (N×ΣY) MECHANICAL : τ={tau_A:+.4f}")
    log.info(f"  Axis B (N×ΣY/N) RATIO : τ={tau_B:+.4f}")

    # The TRUE dependence = FREQUENCY-SEVERITY: per-claim (policy N × claim severity).
    # Pricing and compound capital require this. (Separate concept: "delay×
    # severity" was actually settlement-lag; reported separately below.)
    log.info("\n[3/5] Axis FS (frequency-severity: N × per-claim severity) + CI")
    xfs = claim["n_claims"].values.astype(float)
    yfs = claim["severity"].values.astype(float)
    tau_FS, lo_FS_iid, hi_FS_iid = bca_bootstrap_tau(xfs, yfs)
    # PRIMARY CI: policy-cluster bootstrap (robust to within-cluster correlation)
    lo_FS, hi_FS = cluster_bootstrap_tau(xfs, yfs,
                                         claim["policy_ref"].values)
    log.info(f"  Axis FS (N×severity) FREQUENCY-SEVERITY: τ={tau_FS:+.4f} "
             f"cluster-CI[{lo_FS:+.4f}, {hi_FS:+.4f}] "
             f"(iid-BCa[{lo_FS_iid:+.4f}, {hi_FS_iid:+.4f}])")

    log.info("\n[3b/5] Axis FS-adj (covariate-adjusted direct frequency→severity)")
    axD = axis_d_direct_test(claim)
    log.info(f"  log(sev) ~ N + accident-year + full covariates "
             f"({', '.join(axD['covariates'][1:])}): β_N={axD['beta_N']:+.5f} "
             f"cluster-bootstrap 95% CI [{axD['ci'][0]:+.5f}, {axD['ci'][1]:+.5f}]")
    log.info(f"  Severity effect per additional claim ≈ {axD['pct_effect_per_claim']:+.2%} "
             f"({'significant' if axD['significant'] else 'not significant'}); "
             f"unadjusted τ(N×per-claim sev)={axD['tau_D']:+.4f} (= axis FS)")

    # SECONDARY dependence = SETTLEMENT-LAG (payment delay × severity): relevant
    # for reserve risk; conceptually SEPARATE from frequency-severity.
    log.info("\n[4/5] Axis lag (settlement-lag: delay × severity) + CI")
    xlag = claim["delay"].values.astype(float)
    ylag = claim["severity"].values.astype(float)
    tau_lag, lo_lag_iid, hi_lag_iid = bca_bootstrap_tau(xlag, ylag,
                                                        seed=GLOBAL_SEED + 1)
    lo_lag, hi_lag = cluster_bootstrap_tau(xlag, ylag,
                                           claim["policy_ref"].values,
                                           seed=GLOBAL_SEED + 1)
    log.info(f"  Axis lag (delay×severity) SETTLEMENT-LAG: τ={tau_lag:+.4f} "
             f"cluster-CI[{lo_lag:+.4f}, {hi_lag:+.4f}] "
             f"(iid-BCa[{lo_lag_iid:+.4f}, {hi_lag_iid:+.4f}])")

    log.info("\n[5/5] Figure + save")
    ci_note = (f"CI = policy-cluster bootstrap percentile 95% "
               f"(B={TAU_CLUSTER_N_BOOTSTRAP}; policies resampled with all "
               f"their claims)")
    summary = pd.DataFrame([
        {"eksen": "A_mechanical", "tanim": "N × ΣY", "kendall_tau": tau_A,
         "yorum": "tautoloji — bağımlılık değil",
         "note": ("Policy-level Kendall tau-b (policies with N>=1), raw tied "
                  "data; naive count-sum construction (B1); no CI computed.")},
        {"eksen": "B_ratio", "tanim": "N × ΣY/N", "kendall_tau": tau_B,
         "yorum": "ratio-fallacy",
         "note": ("Policy-level Kendall tau-b (policies with N>=1), raw tied "
                  "data; naive count-average construction (B2); no CI "
                  "computed.")},
        {"eksen": "FS_freq_severity", "tanim": "N × per-claim severity",
         "kendall_tau": tau_FS, "ci_low": lo_FS, "ci_high": hi_FS,
         "yorum": "ASIL frekans-şiddet sinyali (fiyatlama/sermaye)",
         "note": ("Claim-level Kendall tau-b (policy claim count vs claim "
                  "severity), raw tied data, unadjusted; " + ci_note + ".")},
        # R1: kendall_tau INTENTIONALLY blank — previously the unadjusted FS τ
        # was repeated here (not covariate-adjusted). Adjusted effect = beta_N.
        {"eksen": "FS_adj_direct",
         "tanim": "log(sev)~N+kaza-yılı+tam kovaryatlar (küme-boot)",
         "kendall_tau": np.nan, "beta_N": axD["beta_N"],
         "ci_low": axD["ci"][0], "ci_high": axD["ci"][1],
         "yorum": "kovaryat-ayarlı doğrudan frekans→severity",
         "note": ("kendall_tau intentionally blank: the covariate-adjusted "
                  "estimate is beta_N, an OLS coefficient on log severity "
                  "(not a Kendall tau); ci_low/ci_high are the "
                  f"policy-cluster bootstrap 95% CI of beta_N "
                  f"(B={axD['B']}). The unadjusted claim-level tau "
                  "(previously repeated in this row) is in row "
                  "FS_freq_severity.")},
        {"eksen": "lag_settlement", "tanim": "ödeme gecikmesi × severity",
         "kendall_tau": tau_lag, "ci_low": lo_lag, "ci_high": hi_lag,
         "yorum": "settlement-lag bağı (rezerv riski)",
         "note": ("Claim-level Kendall tau-b (mean payment delay vs claim "
                  "severity), raw tied data; " + ci_note + ".")},
    ])
    summary = summary[["eksen", "tanim", "kendall_tau", "yorum", "ci_low",
                       "ci_high", "beta_N", "note"]]
    summary.to_csv(OUTPUTS_DIR / "06_dependence_summary.csv", index=False)
    outfiles.append(str((OUTPUTS_DIR / "06_dependence_summary.csv").resolve()))

    fig2_meta = draw_figure2(tau_A, tau_B, tau_FS, (lo_FS, hi_FS),
                             tau_lag, (lo_lag, hi_lag),
                             FIGURES_DIR / "06_three_axes")
    outfiles.append(str((FIGURES_DIR / "06_three_axes.pdf").resolve()))
    outfiles.append(str((FIGURES_DIR / "06_three_axes.png").resolve()))

    manifest = {
        "script": SCRIPT_NAME,
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "axis_tau": {"A_mechanical": float(tau_A), "B_ratio": float(tau_B),
                     "FS_freq_severity": float(tau_FS),
                     "lag_settlement": float(tau_lag),
                     # backward-compatibility alias: "C_clean" = settlement-lag axis
                     "C_clean": float(tau_lag)},
        # PRIMARY CIs = policy-cluster bootstrap (robust to within-cluster correlation);
        # iid-BCa is stored as secondary (backward compatibility: the *_bca_ci keys
        # carry the primary CI — downstream 18 reads these keys)
        "axis_FS_bca_ci": [float(lo_FS), float(hi_FS)],
        "axis_lag_bca_ci": [float(lo_lag), float(hi_lag)],
        "axis_FS_ci_iid_bca": [float(lo_FS_iid), float(hi_FS_iid)],
        "axis_lag_ci_iid_bca": [float(lo_lag_iid), float(hi_lag_iid)],
        "ci_method_primary": "policy-cluster bootstrap (percentile)",
        "axis_D_direct": axD,
        "n_claims_axis": int(len(claim)),
        # --- R1 additions (new keys only) ---
        "kendall_estimator": {
            "implementation": "scipy.stats.kendalltau",
            "scipy_version": env_info().get("scipy"),
            "variant": KENDALL_VARIANT,
            "variant_description": ("tau-b (tie-corrected); SciPy default, "
                                    "passed explicitly"),
            "data": "raw tied values (no ranking transform, no jitter)",
            "jitter_applied_to_reported_estimates": False,
            "jitter_applied_anywhere_in_module": False,
            "applies_to": ["axis_tau.*", "axis_*_ci (all bootstrap "
                           "replicates)", "axis_D_direct.tau_D"],
        },
        "bootstrap_design": {
            "cluster_bootstrap_B": int(TAU_CLUSTER_N_BOOTSTRAP),
            "cluster_bootstrap_ci": "percentile 95%, policies resampled "
                                    "with replacement with all their claims",
            "iid_bca_B": int(N_BOOTSTRAP),
            "iid_bca_jackknife_subsample_cap": int(BCA_CAP),
            "iid_bca_jackknife_blocks": 200,
            "axis_D_cluster_B": int(AXIS_D_N_BOOTSTRAP),
            "seeds": {"FS": int(GLOBAL_SEED), "lag": int(GLOBAL_SEED + 1),
                      "axis_D": int(GLOBAL_SEED)},
            "ci_available_for": ["FS_freq_severity", "lag_settlement",
                                 "axis_D_direct.beta_N"],
            "ci_not_computed_for": ["A_mechanical", "B_ratio"],
        },
        "summary_csv_note": ("06_dependence_summary.csv row FS_adj_direct: "
                             "kendall_tau intentionally blank since R1 (it "
                             "previously repeated the unadjusted FS tau); "
                             "ci_low/ci_high in that row are the beta_N CI. "
                             "Column 'note' documents every row."),
        "figure2": fig2_meta,
        "interpretation": (
            "Naif eksenler (A,B) yapay biçimde pozitif (B1/B2 artefaktları). "
            f"Temiz hasar-düzeyi FS τ-b = {tau_FS:+.4f} (küçük, negatif); "
            f"kovaryat-ayarlı β_N = {axD['beta_N']:+.4f} (log-severity "
            "ölçeği; sınırlı bir kompozisyon etkisi, τ ölçeğinde "
            f"gösterilmez); settlement-lag τ-b = {tau_lag:+.4f} (zayıf, "
            "pozitif). Bölüm 5 yanlılık karakterizasyonunu doğrular."),
        "output_files": outfiles,
        "environment": env_info(),
    }
    with open(OUTPUTS_DIR / "06_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log.info("\n" + "=" * 78)
    log.info("SCRIPT 06 COMPLETED")
    log.info("=" * 78)


if __name__ == "__main__":
    main()
