"""
================================================================================
SCRIPT 05 — Characterisation of Dependence Biases (Methodological Foundation)
================================================================================
PURPOSE
    The theoretical foundation of this study's METHODOLOGICAL CONTRIBUTION. It
    shows by controlled simulation that THREE common biases in measuring
    frequency-severity dependence produce systematic SPURIOUS association even
    on KNOWN-INDEPENDENT (true τ=0) data. This section justifies why the
    observations in the subsequent sections (06-07) are interpreted only on
    the "clean axis".

    [B1] MECHANICAL TAUTOLOGY — N vs ΣY (total ultimate)
         ΣY = Σ_{i=1}^N Y_i increases with N by definition; even with independent Y,
         τ(N,ΣY) is strongly positive. An arithmetic identity, not dependence.
    [B2] RATIO-FALLACY — N vs ΣY/N (average severity)
         Common N in numerator and denominator → spurious correlation (Pearson 1897).
    [B3] COMPOUND-LEAKAGE — leakage of the frequency margin into severity in MC
         If U_N both determines N and conditions the severity, the monotonicity
         of the NegBin ppf carries over to severity → spurious SCR inflation.

METHOD
    Generates data with Casco-like marginals over a grid of true τ (0=control → 0.20)
    and compares the τ measured by the three biases (and the SCR inflation in B3)
    with the CORRECT measurement. The association is built through the policy's
    common latent risk (Gaussian copula ρ(τ)) — because the high zero proportion
    in the NegBin truncates U_N, direct conditioning on U_N dilutes the association;
    the latent-risk mechanism is both actuarially correct and technically robust.

R1 REVISION (mathematics-4559487, September 2026)
    * The experimental design and RNG stream are UNCHANGED → all numbers identical.
    * B1/B2 replication counts vary with n: reps = max(20, N_REP // (n//1000+1))
      = 100 / 33 / 20 / 20 (n = 1,000 / 5,000 / 25,000 / 100,000). Written to the
      manifest (the "200 replications" statement in the paper was wrong).
    * B3 contains NO injected dependence (τ is not used): 30 replications × 20,000 policies;
      the original loop ran 4 blocks along the τ grid → these are 4 independent
      replications of the same τ=0 experiment. The headline result (276%) = block 1.
      Replication-level VaRs are written to outputs/05_bias_compound_reps.csv.
    * Per-claim recovery ratio (τ_correct/τ_true, τ>0) table:
      outputs/05_recovery_ratio.csv.
    * Figure 1 redrawn (print size; panel (c) without a τ axis, the distribution
      of the 30 replications + ratio-of-means estimate and 95% bootstrap interval).

OUTPUT : outputs/05_bias_*.csv ; outputs/05_recovery_ratio.csv ;
        figures/05_bias_characterization.{pdf,png} ; outputs/05_manifest.json
================================================================================
"""
from __future__ import annotations
import sys
import json
import logging
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
from matplotlib.lines import Line2D
from scipy import stats

from config import (
    DATA_PROC_DIR, OUTPUTS_DIR, FIGURES_DIR, LOGS_DIR, GLOBAL_SEED, FIG_DPI,
    SOLVENCY_ALPHA, BIAS_N_GRID, BIAS_TAU_TRUE_GRID, BIAS_N_REP,
    BIAS_SEV_MEANLOG, BIAS_SEV_SDLOG, BIAS_FREQ_LAMBDA, BIAS_MC_CAP, BIAS_NMAX,
    FIG_W_FULL, apply_pub_style, env_info, POSCTRL_DEFAULT_LN,
)

SCRIPT_NAME = "05_bias_theory"
# All simulation parameters are read from config.py (hard-coding removed)
N_GRID = BIAS_N_GRID
TAU_TRUE_GRID = BIAS_TAU_TRUE_GRID
N_REP = BIAS_N_REP
SEV_MEANLOG, SEV_SDLOG = BIAS_SEV_MEANLOG, BIAS_SEV_SDLOG
FREQ_LAMBDA = BIAS_FREQ_LAMBDA
MC_CAP = BIAS_MC_CAP

# --- R1 revision constants ---------------------------------------------------
# B1/B2 replication rule (UNCHANGED from the original run; only documented).
B12_REP_RULE = "max(20, BIAS_N_REP // (n // 1000 + 1))"
# B3: replications per block (was the hard-coded `range(30)`; unchanged).
B3_N_REP = 30
# The original B3 loop iterated over TAU_TRUE_GRID, but compound_leak() never
# used tau, so it ran len(TAU_TRUE_GRID) = 4 independent blocks of the SAME
# zero-dependence experiment. The loop is kept (identical RNG stream); the
# headline figure (276%) is block 1 (formerly labelled tau_true = 0.0).
B3_N_BLOCKS = len(BIAS_TAU_TRUE_GRID)
B3_HEADLINE_BLOCK = 1
B3_TAU_INJECTED = 0.0
# Monte Carlo interval for the B3 headline (ratio of mean VaRs over the 30
# replications): percentile bootstrap over replications, separate RNG stream
# (does not touch the simulation stream, so all simulated numbers are unchanged).
B3_BOOT_B = 10_000
B3_BOOT_SEED = GLOBAL_SEED + 505
# Figure 1
FIG_STEM = "05_bias_characterization"
FIG_H = 2.75
# Ordinal blue ramp (light → dark = small → large portfolio) + distinct markers
N_COLORS = ["#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"]
N_MARKERS = ["o", "s", "^", "D"]
B3_POINT_COLOR = "#eb6834"
B3_JITTER_SEED = GLOBAL_SEED + 506   # jitter of strip-plot x positions only
# -----------------------------------------------------------------------------


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


# =========================================================
# Common-latent-risk generator (Gaussian copula ρ(τ))
# =========================================================
def generate(n_pol, tau_true, rng):
    """The policy's common latent risk Z affects both frequency and severity.
    Z~N(0,1); U_N=Φ(Z) → N (Poisson); each claim's U_Y is linked to Z with ρ."""
    rho = np.sin(np.pi * tau_true / 2)
    Z = rng.normal(size=n_pol)
    U_N = stats.norm.cdf(Z)
    N = stats.poisson.ppf(U_N, FREQ_LAMBDA).astype(int)
    N = np.minimum(N, BIAS_NMAX)
    tot = int(N.sum())
    if tot == 0:
        return None
    pol_idx = np.repeat(np.arange(n_pol), N)
    pcN = np.repeat(N, N).astype(float)
    eps = rng.normal(size=tot)
    Z_Y = rho * Z[pol_idx] + np.sqrt(max(1 - rho**2, 1e-12)) * eps
    U_Y = stats.norm.cdf(Z_Y)
    Y = stats.lognorm.ppf(np.clip(U_Y, 1e-9, 1 - 1e-9),
                          SEV_SDLOG, scale=np.exp(SEV_MEANLOG))
    total = np.bincount(pol_idx, weights=Y, minlength=n_pol)
    avg = np.divide(total, N, out=np.zeros_like(total), where=N > 0)
    return {"N": N, "Z": Z, "pol_idx": pol_idx, "pcN": pcN, "U_Y": U_Y,
            "Y": Y, "total": total, "avg": avg}


def measure(d):
    m = d["N"] > 0
    tau_mech = stats.kendalltau(d["N"][m], d["total"][m])[0]
    tau_ratio = stats.kendalltau(d["N"][m], d["avg"][m])[0]
    tau_correct = (stats.kendalltau(d["pcN"], d["Y"])[0]
                   if len(d["pcN"]) > 10 else np.nan)
    return tau_mech, tau_ratio, tau_correct


def compound_leak(n_mc, rng):
    """[B3] Shared-uniform leakage vs clean simulation, 99.5% VaR of the
    aggregate loss. NO dependence is injected: counts come from U_N alone and
    (clean arm) severities from independent uniforms. The leaky arm reuses the
    rank of U_N as the severity uniform. (R1: the unused `tau_true` argument was
    removed; the RNG draws are exactly those of the original function.)"""
    Z = rng.normal(size=n_mc)
    U_N = stats.norm.cdf(Z)
    N = np.minimum(stats.poisson.ppf(U_N, FREQ_LAMBDA).astype(int), BIAS_NMAX)
    tot = int(N.sum())
    if tot == 0:
        return np.nan, np.nan
    idx = np.repeat(np.arange(n_mc), N)
    # leaky: U_N (rank) propagates into severity
    U_leak = np.repeat(stats.rankdata(U_N) / (n_mc + 1), N)
    Y_leak = stats.lognorm.ppf(np.clip(U_leak, 1e-9, 1-1e-9),
                               SEV_SDLOG, scale=np.exp(SEV_MEANLOG))
    # clean: independent uniform
    Y_clean = stats.lognorm.ppf(rng.uniform(size=tot),
                                SEV_SDLOG, scale=np.exp(SEV_MEANLOG))
    agg_leak = np.bincount(idx, weights=Y_leak, minlength=n_mc)
    agg_clean = np.bincount(idx, weights=Y_clean, minlength=n_mc)
    return (float(np.quantile(agg_leak, SOLVENCY_ALPHA)),
            float(np.quantile(agg_clean, SOLVENCY_ALPHA)))


def _sd(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.std(x, ddof=1)) if len(x) > 1 else np.nan


def _n_ok(x):
    return int(np.isfinite(np.asarray(x, float)).sum())


def b3_block_summary(vls, vcs, boot_rng):
    """Headline = ratio of mean VaRs (original definition); plus per-replication
    distribution and a percentile-bootstrap 95% MC interval over replications."""
    vl = np.asarray(vls, float); vc = np.asarray(vcs, float)
    ok = np.isfinite(vl) & np.isfinite(vc)
    vl, vc = vl[ok], vc[ok]
    infl_rom = (np.mean(vl) / np.mean(vc) - 1) * 100
    per_rep = (vl / vc - 1) * 100
    k = len(vl)
    idx = boot_rng.integers(0, k, size=(B3_BOOT_B, k))
    boot = (vl[idx].mean(axis=1) / vc[idx].mean(axis=1) - 1) * 100
    se_mean = per_rep.std(ddof=1) / np.sqrt(k)
    tcrit = stats.t.ppf(0.975, k - 1)
    return {
        "n_rep_valid": int(k),
        "inflation_pct_ratio_of_means": float(infl_rom),
        "inflation_pct_ratio_of_means_boot95": [float(np.percentile(boot, 2.5)),
                                                float(np.percentile(boot, 97.5))],
        "per_rep_inflation_pct_mean": float(per_rep.mean()),
        "per_rep_inflation_pct_mean_t95": [float(per_rep.mean() - tcrit * se_mean),
                                           float(per_rep.mean() + tcrit * se_mean)],
        "per_rep_inflation_pct_median": float(np.median(per_rep)),
        "per_rep_inflation_pct_sd": float(per_rep.std(ddof=1)),
        "per_rep_inflation_pct_p2_5_p97_5": [float(np.percentile(per_rep, 2.5)),
                                             float(np.percentile(per_rep, 97.5))],
        "per_rep_inflation_pct_min_max": [float(per_rep.min()), float(per_rep.max())],
        "VaR_leaky_mean": float(vl.mean()), "VaR_clean_mean": float(vc.mean()),
    }


def draw_figure(mech_df, ratio_df, reps_df, b3_head):
    """Figure 1 at print size (FIG_W_FULL), three panels."""
    set_style(); apply_pub_style()
    fig = plt.figure(figsize=(FIG_W_FULL, FIG_H))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.80], left=0.075,
                          right=0.972, bottom=0.31, top=0.90, wspace=0.40)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1], sharey=ax_a)
    ax_c = fig.add_subplot(gs[2])

    xt = list(TAU_TRUE_GRID)
    xtl = ["0" if t == 0 else f"{t:.2f}" for t in xt]
    for ax, df, col, title in [
            (ax_a, mech_df, "tau_mechanical", "(a) Count-sum association (B1)"),
            (ax_b, ratio_df, "tau_ratio", "(b) Count-average association (B2)")]:
        ax.plot([0, 0.2], [0, 0.2], color="0.15", ls="--", lw=0.9, zorder=1)
        for i, n_pol in enumerate(N_GRID):
            sub = df[df["n"] == n_pol].sort_values("tau_true")
            ax.plot(sub["tau_true"], sub[col], marker=N_MARKERS[i], ms=3.6,
                    lw=1.0, color=N_COLORS[i], mec="white", mew=0.4,
                    zorder=3 + i)
        ax.set_xticks(xt); ax.set_xticklabels(xtl)
        ax.set_xlim(-0.012, 0.212)
        ax.set_xlabel("True Kendall's τ")
        ax.set_title(title, loc="center")
        # direct label of the reference line (below it, in empty space)
        ax.text(0.163, 0.082, "y = x (exact recovery)", fontsize=6.5,
                color="0.15", ha="center", va="center")
    ax_a.set_ylabel("Measured Kendall's τ")
    ax_a.set_ylim(-0.015, 0.42)
    ax_a.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax_a.set_yticklabels(["0", "0.1", "0.2", "0.3", "0.4"])

    # shared legend for panels (a)-(b), below the figure
    handles = [Line2D([], [], color=N_COLORS[i], marker=N_MARKERS[i], ms=3.6,
                      lw=1.0, mec="white", mew=0.4, label=f"{n:,}")
               for i, n in enumerate(N_GRID)]
    bb_a = ax_a.get_position(); bb_b = ax_b.get_position()
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=((bb_a.x0 + bb_b.x1) / 2, 0.0),
               title="Portfolio size n (policies), panels (a) and (b)",
               title_fontsize=7, handlelength=2.2, columnspacing=1.6,
               frameon=False)

    # (c) B3: distribution over the 30 replications of the headline block
    head = reps_df[reps_df["block"] == B3_HEADLINE_BLOCK]
    y = head["inflation_pct"].to_numpy()
    jit = np.random.default_rng(B3_JITTER_SEED).uniform(-0.15, 0.15, size=len(y))
    ax_c.scatter(jit, y, s=11, color=B3_POINT_COLOR, alpha=0.8,
                 edgecolor="white", linewidth=0.4, zorder=3,
                 label="One replication")
    est = b3_head["inflation_pct_ratio_of_means"]
    lo, hi = b3_head["inflation_pct_ratio_of_means_boot95"]
    ax_c.errorbar([0.45], [est], yerr=[[est - lo], [hi - est]], fmt="D",
                  color="0.1", ms=4, capsize=2.5, lw=1.0, zorder=4,
                  label="Ratio of mean VaRs\n(95% bootstrap interval)")
    ax_c.annotate(f"{est:.0f}%", xy=(0.45, est), xytext=(0.57, est),
                  va="center", ha="left", fontsize=7)
    ax_c.set_xlim(-0.32, 0.85)
    ax_c.set_xticks([])
    ax_c.grid(axis="x", visible=False)
    ax_c.set_xlabel(f"{B3_N_REP} replications × {MC_CAP:,} policies,\n"
                    "true τ = 0")
    ax_c.set_ylabel("99.5% VaR inflation (%)")
    ymin = np.floor(min(y.min(), lo) / 50) * 50 - 10
    ymax = np.ceil(max(y.max(), hi) / 50) * 50 + 10
    ax_c.set_ylim(ymin, ymax)
    ax_c.set_title("(c) Shared-uniform leakage (B3)", loc="center")
    bb_c = ax_c.get_position()
    h_c, l_c = ax_c.get_legend_handles_labels()
    fig.legend(handles=h_c, labels=l_c, loc="lower center",
               bbox_to_anchor=((bb_c.x0 + bb_c.x1) / 2, -0.02), ncol=1,
               frameon=False, handletextpad=0.4, markerscale=0.9)
    return fig


def main() -> None:
    global log
    log = setup_logger(); set_style(); apply_pub_style()
    log.info("=" * 78)
    log.info(f"SCRIPT {SCRIPT_NAME} — Bias Characterisation")
    log.info("=" * 78)

    outfiles: list = []
    rng = np.random.default_rng(GLOBAL_SEED)

    log.info("\n[1/3] [B1][B2] Mechanical & Ratio — scan over n and true τ")
    mech_rows, ratio_rows = [], []
    reps_per_n = {}
    for tau_true in TAU_TRUE_GRID:
        for n_pol in N_GRID:
            reps = max(20, N_REP // (n_pol // 1000 + 1))   # UNCHANGED rule
            reps_per_n[int(n_pol)] = int(reps)
            tm, tr, tc = [], [], []
            for _ in range(reps):
                d = generate(n_pol, tau_true, rng)
                if d is None:
                    continue
                a, b, c = measure(d)
                tm.append(a); tr.append(b); tc.append(c)
            k_tc = _n_ok(tc)
            mech_rows.append({"tau_true": tau_true, "n": n_pol,
                              "tau_mechanical": np.nanmean(tm),
                              "tau_correct": np.nanmean(tc),
                              # R1 additions (appended columns)
                              "n_rep": int(reps), "n_rep_valid": _n_ok(tm),
                              "tau_mechanical_sd": _sd(tm),
                              "tau_correct_sd": _sd(tc),
                              "tau_correct_mc_se": (_sd(tc) / np.sqrt(k_tc)
                                                    if k_tc > 1 else np.nan)})
            ratio_rows.append({"tau_true": tau_true, "n": n_pol,
                               "tau_ratio": np.nanmean(tr),
                               "n_rep": int(reps), "n_rep_valid": _n_ok(tr),
                               "tau_ratio_sd": _sd(tr)})
        log.info(f"  τ_true={tau_true:.2f}: mechanical τ̄={mech_rows[-1]['tau_mechanical']:+.3f}, "
                 f"ratio τ̄={ratio_rows[-1]['tau_ratio']:+.3f}, "
                 f"correct τ̄={mech_rows[-1]['tau_correct']:+.3f}")
    mech_df = pd.DataFrame(mech_rows); ratio_df = pd.DataFrame(ratio_rows)
    ctrl = mech_df[mech_df["tau_true"] == 0.0]["tau_mechanical"].mean()
    ctrl_ratio = ratio_df[ratio_df["tau_true"] == 0.0]["tau_ratio"].mean()
    log.info(f"  → EVIDENCE: with true τ=0, mechanical τ̄={ctrl:+.3f} (spuriously positive; "
             f"tautology), ratio also departs from zero; correct measurement ≈0.")
    log.info(f"  Replications per n (rule {B12_REP_RULE}): "
             + ", ".join(f"n={n:,}: {r}" for n, r in reps_per_n.items()))

    # --- R1: per-claim recovered-to-injected ratio (tau > 0) ---------------
    rec = mech_df[mech_df["tau_true"] > 0][
        ["tau_true", "n", "n_rep", "tau_correct", "tau_correct_mc_se"]].copy()
    rec["recovery_ratio"] = rec["tau_correct"] / rec["tau_true"]
    rec["recovery_ratio_mc_se"] = rec["tau_correct_mc_se"] / rec["tau_true"]
    rec = rec.reset_index(drop=True)
    i_min = int(rec["recovery_ratio"].idxmin()); i_max = int(rec["recovery_ratio"].idxmax())
    log.info(f"  Per-claim recovery ratio (τ>0): min {rec.loc[i_min, 'recovery_ratio']:.3f} "
             f"(n={int(rec.loc[i_min, 'n']):,}, τ={rec.loc[i_min, 'tau_true']:.2f}), "
             f"max {rec.loc[i_max, 'recovery_ratio']:.3f} "
             f"(n={int(rec.loc[i_max, 'n']):,}, τ={rec.loc[i_max, 'tau_true']:.2f})")

    log.info("\n[2/3] [B3] Compound-leakage SCR inflation (NO injected dependence)")
    comp_rows, rep_rows = [], []
    boot_rng = np.random.default_rng(B3_BOOT_SEED)
    b3_blocks = {}
    # Loop kept over the legacy grid so the RNG stream is identical to the
    # original run; tau is NOT used by the generator (4 blocks at tau = 0).
    for block, legacy_label in enumerate(TAU_TRUE_GRID, start=1):
        vls, vcs = [], []
        for r in range(B3_N_REP):
            vl, vc = compound_leak(MC_CAP, rng)
            vls.append(vl); vcs.append(vc)
            rep_rows.append({"block": block, "rep": r + 1,
                             "tau_injected": B3_TAU_INJECTED,
                             "n_policies": MC_CAP,
                             "VaR_leaky": vl, "VaR_clean": vc,
                             "VaR_ratio": vl / vc if vc else np.nan,
                             "inflation_pct": (vl / vc - 1) * 100 if vc else np.nan})
        vl_m, vc_m = np.nanmean(vls), np.nanmean(vcs)
        infl = (vl_m / vc_m - 1) * 100 if vc_m else np.nan
        comp_rows.append({"block": block, "tau_injected": B3_TAU_INJECTED,
                          "legacy_tau_label": legacy_label,
                          "n_rep": B3_N_REP, "n_policies": MC_CAP,
                          "VaR_leaky": vl_m, "VaR_clean": vc_m,
                          "spurious_inflation_pct": infl})
        b3_blocks[block] = b3_block_summary(vls, vcs, boot_rng)
        log.info(f"  block {block} (τ injected = 0): leaky VaR={vl_m:,.0f} vs "
                 f"clean={vc_m:,.0f} → spurious inflation {infl:+.1f}%")
    comp_df = pd.DataFrame(comp_rows)
    reps_df = pd.DataFrame(rep_rows)
    c0 = comp_df[comp_df["block"] == B3_HEADLINE_BLOCK]["spurious_inflation_pct"].iloc[0]
    b3_head = b3_blocks[B3_HEADLINE_BLOCK]
    # pooled over all 4 blocks (120 replications) — robustness only
    vl_all, vc_all = reps_df["VaR_leaky"].to_numpy(), reps_df["VaR_clean"].to_numpy()
    b3_pooled = {"n_rep": int(len(reps_df)),
                 "inflation_pct_ratio_of_means":
                     float((np.nanmean(vl_all) / np.nanmean(vc_all) - 1) * 100),
                 "block_inflation_pct_range":
                     [float(comp_df["spurious_inflation_pct"].min()),
                      float(comp_df["spurious_inflation_pct"].max())]}
    log.info(f"  → EVIDENCE: even with true τ=0, leakage inflates the VaR by {c0:.0f}% "
             f"(95% bootstrap [{b3_head['inflation_pct_ratio_of_means_boot95'][0]:.0f}, "
             f"{b3_head['inflation_pct_ratio_of_means_boot95'][1]:.0f}]; "
             f"per-replication range {b3_head['per_rep_inflation_pct_min_max'][0]:.0f}"
             f"–{b3_head['per_rep_inflation_pct_min_max'][1]:.0f}%).")
    log.info(f"  All {b3_pooled['n_rep']} replications pooled: "
             f"{b3_pooled['inflation_pct_ratio_of_means']:.1f}%; "
             f"block range {b3_pooled['block_inflation_pct_range'][0]:.1f}"
             f"–{b3_pooled['block_inflation_pct_range'][1]:.1f}%.")

    log.info("\n[3/3] Figure + saving")
    fig = draw_figure(mech_df, ratio_df, reps_df, b3_head)
    fig.savefig(FIGURES_DIR / f"{FIG_STEM}.pdf")
    fig.savefig(FIGURES_DIR / f"{FIG_STEM}.png")
    plt.close(fig)
    outfiles.append(str((FIGURES_DIR / f"{FIG_STEM}.png").resolve()))
    outfiles.append(str((FIGURES_DIR / f"{FIG_STEM}.pdf").resolve()))
    for df, nm in [(mech_df, "05_bias_mechanical"), (ratio_df, "05_bias_ratio"),
                   (comp_df, "05_bias_compound"),
                   (reps_df, "05_bias_compound_reps"),
                   (rec, "05_recovery_ratio")]:
        p = OUTPUTS_DIR / f"{nm}.csv"
        df.to_csv(p, index=False); outfiles.append(str(p.resolve()))

    # --- manuscript-number checks ------------------------------------------
    def _by_n(df, col, tau):
        s = df[df["tau_true"] == tau].set_index("n")[col]
        return {str(int(k)): float(v) for k, v in s.items()}
    pc = mech_df.set_index(["tau_true", "n"])["tau_correct"]
    checks = {
        "B1_tau_at_zero_mean_over_n": float(ctrl),
        "B1_tau_at_zero_by_n": _by_n(mech_df, "tau_mechanical", 0.0),
        "B2_tau_at_zero_mean_over_n": float(ctrl_ratio),
        "B2_tau_at_zero_by_n": _by_n(ratio_df, "tau_ratio", 0.0),
        "per_claim_tau_at_zero_by_n": _by_n(mech_df, "tau_correct", 0.0),
        "B3_inflation_pct_headline": float(c0),
        "per_claim_tau_true_0.05_by_n": _by_n(mech_df, "tau_correct", 0.05),
        "per_claim_tau_true_0.10_by_n": _by_n(mech_df, "tau_correct", 0.10),
        "per_claim_tau_true_0.05_n100000": float(pc.loc[(0.05, 100000)]),
        "per_claim_tau_true_0.10_n100000": float(pc.loc[(0.10, 100000)]),
        "note": ("Manuscript quotes +0.29 (B1) and +0.07 (B2) = means over the "
                 "four n at tau=0; 276% = B3 block 1; per-claim 0.032 "
                 "(tau=0.05) and 0.067 (tau=0.10) are the n = 100,000 values."),
    }
    rec_records = [{k: (float(v) if isinstance(v, (float, np.floating)) else int(v))
                    for k, v in row.items()} for row in rec.to_dict("records")]
    n_max = max(N_GRID)
    rec_nmax = rec[rec["n"] == n_max]["recovery_ratio"]

    # --- R1 final: Remark 1 sign check (B2 mechanism), seeded Monte Carlo -----
    # E[sgn(A_{n2} - A_{n1})] for independent means A_n of n i.i.d. lognormal
    # severities (shape = fitted sdlog; the sign is scale-invariant).
    r1_s = float(POSCTRL_DEFAULT_LN[0]); r1_M = 400_000
    r1_rng = np.random.default_rng(GLOBAL_SEED + 505)
    r1_rows = []
    for n1 in range(1, 7):
        for n2 in range(n1 + 1, 7):
            a1 = np.exp(r1_s * r1_rng.standard_normal((r1_M, n1))).mean(axis=1)
            a2 = np.exp(r1_s * r1_rng.standard_normal((r1_M, n2))).mean(axis=1)
            sg = np.sign(a2 - a1)
            r1_rows.append({"n1": n1, "n2": n2, "E_sgn": float(sg.mean()),
                            "mc_se": float(sg.std(ddof=1) / np.sqrt(r1_M))})
    r1_df = pd.DataFrame(r1_rows)
    r1_df.to_csv(OUTPUTS_DIR / "05_remark1_sign_check.csv", index=False)
    outfiles.append(str((OUTPUTS_DIR / "05_remark1_sign_check.csv").resolve()))
    log.info(f"  Remark 1 sign check: min E[sgn] = {r1_df['E_sgn'].min():+.4f}; "
             f"(1,2) = {r1_df.loc[0, 'E_sgn']:+.4f}")

    manifest = {
        "script": SCRIPT_NAME,
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "method": ["B1 mechanical tautology", "B2 ratio-fallacy",
                   "B3 compound leakage"],
        "control_finding": {"mechanical_tau_at_zero": float(ctrl),
                            "compound_inflation_at_zero": float(c0),
                            # R1 additions
                            "ratio_tau_at_zero": float(ctrl_ratio),
                            "compound_inflation_at_zero_boot95":
                                b3_head["inflation_pct_ratio_of_means_boot95"]},
        "params": {"N_GRID": N_GRID, "TAU_TRUE_GRID": TAU_TRUE_GRID,
                   "FREQ_LAMBDA": FREQ_LAMBDA, "GLOBAL_SEED": GLOBAL_SEED,
                   # R1 additions
                   "BIAS_N_REP": N_REP, "SEV_MEANLOG": SEV_MEANLOG,
                   "SEV_SDLOG": SEV_SDLOG, "BIAS_NMAX": BIAS_NMAX,
                   "BIAS_MC_CAP": MC_CAP, "SOLVENCY_ALPHA": SOLVENCY_ALPHA},
        "design": {
            "rng": ("single numpy default_rng(GLOBAL_SEED) stream, consumed "
                    "sequentially: B1/B2 loop (tau outer, n inner) then B3 blocks"),
            "B1_B2": {
                "generator": ("common latent risk Z~N(0,1); N = Poisson(lambda) "
                              "inverse CDF of Phi(Z), capped at BIAS_NMAX; each "
                              "severity lognormal via Gaussian copula with "
                              "rho = sin(pi*tau/2) to Z"),
                "tau_true_grid": TAU_TRUE_GRID, "n_grid": N_GRID,
                "replication_rule": B12_REP_RULE,
                "replications_per_n": {str(k): v for k, v in reps_per_n.items()},
                "replications_note": ("NOT 200 per cell: BIAS_N_REP = 200 is "
                                      "divided by (n//1000 + 1) with a floor of 20"),
                "total_replications": int(len(TAU_TRUE_GRID)
                                          * sum(reps_per_n.values())),
                "B1_B2_computed_on": "policies with N >= 1",
                "per_claim_measure": "Kendall tau-b of (policy count, claim severity) over all claims",
            },
            "B3": {
                "n_policies_per_replication": MC_CAP,
                "n_rep_per_block": B3_N_REP,
                "tau_injected": B3_TAU_INJECTED,
                "n_blocks_run": B3_N_BLOCKS,
                "headline_block": B3_HEADLINE_BLOCK,
                "risk_measure": f"empirical {SOLVENCY_ALPHA:.3f} quantile of aggregate loss per replication",
                "headline_definition": "(mean_r VaR_leaky / mean_r VaR_clean - 1) x 100 over the 30 replications of block 1",
                "note": ("compound_leak() never used tau: the original loop over "
                         "TAU_TRUE_GRID ran 4 independent blocks of the same "
                         "zero-dependence experiment (legacy labels 0/0.05/0.10/0.20 "
                         "in 05_bias_compound.csv column legacy_tau_label)."),
                "bootstrap": {"method": "percentile bootstrap over replications of the ratio of mean VaRs",
                              "B": B3_BOOT_B, "seed": B3_BOOT_SEED,
                              "separate_stream": True},
                "headline_block_summary": b3_head,
                "all_blocks_summary": {str(k): v for k, v in b3_blocks.items()},
                "pooled_all_blocks": b3_pooled,
            },
        },
        "recovery_ratio": {
            "definition": "per-claim tau_correct / tau_true, tau_true > 0 (stylized generator)",
            "table": rec_records,
            "min": {"value": float(rec.loc[i_min, "recovery_ratio"]),
                    "n": int(rec.loc[i_min, "n"]),
                    "tau_true": float(rec.loc[i_min, "tau_true"])},
            "max": {"value": float(rec.loc[i_max, "recovery_ratio"]),
                    "n": int(rec.loc[i_max, "n"]),
                    "tau_true": float(rec.loc[i_max, "tau_true"])},
            f"range_n{n_max}": [float(rec_nmax.min()), float(rec_nmax.max())],
            "file": "outputs/05_recovery_ratio.csv",
        },
        "manuscript_checks": checks,
        "remark1_sign_check": {"lognormal_sdlog": r1_s, "M": r1_M, "all_positive": bool((r1_df["E_sgn"] > 0).all()), "E_sgn_1_2": float(r1_df.loc[0, "E_sgn"]), "file": "outputs/05_remark1_sign_check.csv"},
        "figure": {"stem": FIG_STEM, "width_in": FIG_W_FULL, "height_in": FIG_H,
                   "panels": ["(a) Count-sum association (B1): measured vs true tau, one line per n, y = x reference",
                              "(b) Count-average association (B2): same layout",
                              "(c) Shared-uniform leakage (B3): 30 per-replication 99.5% VaR inflations "
                              "(block 1, tau = 0) + ratio of mean VaRs with 95% bootstrap interval"],
                   "strip_jitter_seed": B3_JITTER_SEED},
        "output_files": outfiles,
        "environment": env_info(),
    }
    with open(OUTPUTS_DIR / "05_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log.info("\n" + "=" * 78)
    log.info("SCRIPT 05 COMPLETED")
    log.info("=" * 78)


if __name__ == "__main__":
    main()
