"""
================================================================================
SCRIPT 09 — Positive Control (Method Validity / Second Regime)
================================================================================
PURPOSE
    Proves that the negative finding (weak dependence) is a genuine property
    of the data, not a METHOD DEFECT. Positive-control logic: generate
    synthetic data with deliberately injected strong dependence and check
    whether the pipeline detects it. If the pipeline recovers the strong
    association, its failure to find association in the real data is credible.

METHOD
    Two regimes, differing ONLY in dependence:
      Regime A : real Casco (τ_C ≈ 0.02)
      Regime B : Casco's REAL marginals (NegBin frequency, LogNormal
                severity — from Section 10) + KNOWN τ (0 → 0.40)
    The association runs through the policy's common latent risk (Gaussian copula ρ(τ));
    because the high zero proportion of the NegBin dilutes direct conditioning on U_N,
    the latent-risk mechanism is used (both actuarially correct and technically robust).
    Recovery: injected τ vs measured τ (monotone?), and SCR deviation (material
    under strong association, negligible at zero?).

R1 REVISION (September 2026)
    (a) Capital arm: because the latent Z drives both N and severity, the
        claim-level severity marginal in the dependent arm shifts UPWARD (claims
        occur only at high Z). The comparison against an independent arm drawn
        from the unconditional lognormal therefore gives the "total effect"
        (marginal shift + dependence). In addition a MARGIN-PRESERVING independent
        arm was added: the REALISED claim severities of the dependent arm are
        randomly permuted across all claims of the portfolio (Y_perm =
        rng.permutation(Y_dep)); the claim-level distribution stays EXACTLY the
        same, the policy/count/latent link is broken. The "material capital effect
        under strong association" criterion is evaluated on this margin-fixed
        effect (threshold unchanged, 10%).
    (b) Attenuation diagnosis: population latent τ (one draw per policy, as if
        every policy had a claim), claimant policies only (one draw per policy),
        claim-level latent τ, per-claim τ-b(N,Y) and their ratios to the injected
        τ + the tie ceiling of the synthetic count composition sqrt(1 − Σ q_k²).
        The dominant mechanism is selection / range restriction (severity is
        observed only for N ≥ 1), not count ties.
    (c) Figure 3 redrawn at print size (no inset; ±1.96 SD error bars in both
        panels; total and margin-fixed effect in panel (b)).

LIMITATION (honesty)
    Regime B is synthetic; it proves INTERNAL VALIDITY ("the method sees the signal"),
    not EXTERNAL VALIDITY ("other lines of business") — a limitations item.

INPUT  : (marginals from Section 10; a reasonable default is used if absent)
OUTPUT : outputs/09_recovery_table.csv ; outputs/09_manifest.json ;
        figures/09_positive_control.{pdf,png}
================================================================================
"""
from __future__ import annotations
import sys
import json
import logging
import pickle
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
    DATA_PROC_DIR, OUTPUTS_DIR, FIGURES_DIR, LOGS_DIR, GLOBAL_SEED,
    SOLVENCY_ALPHA, TAU_INJECT_GRID, N_SYNTH_POSCTRL, POSCTRL_NMAX,
    POSCTRL_DEFAULT_NB, POSCTRL_DEFAULT_LN, POSCTRL_N_REP,
    FIG_W_FULL, apply_pub_style, env_info,
)

# --- R1 revision constants ---------------------------------------------------
# Prespecified recovery thresholds (UNCHANGED from the original run; they were
# hard-coded as 10 and 2 in main()). Only the arm on which the "material
# capital effect" criterion is evaluated changes (margin-preserving arm).
SCR_MATERIAL_PCT = 10.0     # |capital effect| at max injected τ must exceed this
SCR_NULL_PCT = 2.0          # |capital effect| at τ = 0 must stay below this
ERRBAR_Z = 1.96             # figure error bars: ±1.96 SD across replications
INDEP_ARM_TOTAL = ("unconditional: Y_ind = LN^{-1}(Phi(eps)), eps ~ N(0,1) "
                   "i.i.d. per claim (marginal shift + dependence)")
INDEP_ARM_MARGIN_FIXED = ("margin-preserving: Y_perm = rng.permutation(Y_dep) "
                          "across ALL claims of the portfolio (identical "
                          "claim-level severity distribution; link to the "
                          "policy, its count and its latent broken)")
# -----------------------------------------------------------------------------

SCRIPT_NAME = "09_positive_control"
# All parameters are read from config.py (hard-coding removed)
N_SYNTH = N_SYNTH_POSCTRL
DEFAULT_NB = POSCTRL_DEFAULT_NB            # NegBin (r, p) fallback
DEFAULT_LN = POSCTRL_DEFAULT_LN            # LogNormal (s, loc, scale) fallback


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


def get_marginals():
    """Load the Section 10 marginals; otherwise a Casco-consistent default."""
    p = DATA_PROC_DIR / "marginal_fits.pkl"
    if p.exists():
        mb = pickle.load(open(p, "rb"))
        nb = tuple(mb["frequency"]["params"])
        ln = tuple(mb["severity"]["params"])
        return nb, ln, "fitted"
    return DEFAULT_NB, DEFAULT_LN, "default"


def generate_regime(n, tau, nb_params, ln_params, rng):
    """Common latent risk: Z affects both N and severity (Gaussian ρ(τ))."""
    r, p = nb_params
    rho = np.sin(np.pi * tau / 2)
    Z = rng.normal(size=n)
    N = np.minimum(stats.nbinom.ppf(stats.norm.cdf(Z), r, p).astype(int),
                   POSCTRL_NMAX)
    tot = int(N.sum())
    if tot == 0:
        return None
    pol_idx = np.repeat(np.arange(n), N)
    pcN = np.repeat(N, N).astype(float)
    eps = rng.normal(size=tot)
    Z_Y = rho * Z[pol_idx] + np.sqrt(max(1 - rho**2, 1e-12)) * eps
    U_Y = stats.norm.cdf(Z_Y)
    Y = stats.lognorm.ppf(np.clip(U_Y, 1e-9, 1 - 1e-9), *ln_params)
    return {"N": N, "Z": Z, "pol_idx": pol_idx, "pcN": pcN, "U_Y": U_Y, "Y": Y}


def _clopper_pearson(k, n, alpha=0.05):
    """Exact (Clopper-Pearson) 95% CI for a binomial proportion."""
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha/2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - alpha/2, k + 1, n - k))
    return [lo, hi]


def recover_tau_latent(d):
    return float(stats.kendalltau(stats.norm.cdf(d["Z"])[d["pol_idx"]],
                                  d["U_Y"])[0])


def recover_scr(d, tau, ln_params, rng) -> dict:
    """Policy-level 99.5% loss quantile: dependent arm vs two independent arms.

    Dependent arm : Y_dep = LN⁻¹(Φ(ρ Z_pol + √(1−ρ²) ε))  (latent Z also drives N)
    Total effect  : Y_ind = LN⁻¹(Φ(ε)) — UNCONDITIONAL marginal (original comparison;
                  because the dependent arm's claim-level marginal shifts upward,
                  marginal shift + dependence are measured together)
    Dependence    : Y_perm = permutation(Y_dep) — the REALISED claim-level
    effect          distribution of the dependent arm is preserved exactly, the
                  policy/count/latent link is broken (margin-fixed effect)
    All effects (independent / dependent − 1)·100: negative = the independence
    assumption UNDERSTATES the quantile.

    RNG order: ε (original) → permutation (R1 addition). Because the original draws
    come first, the total-effect numbers are identical to those before R1.
    """
    N = d["N"]; Z = d["Z"]; n = len(N)
    tot = int(N.sum())
    pol_idx = np.repeat(np.arange(n), N)
    rho = np.sin(np.pi * tau / 2)
    eps = rng.normal(size=tot)
    Z_dep = rho * Z[pol_idx] + np.sqrt(max(1 - rho**2, 1e-12)) * eps
    Y_dep = stats.lognorm.ppf(np.clip(stats.norm.cdf(Z_dep), 1e-9, 1-1e-9), *ln_params)
    Y_ind = stats.lognorm.ppf(np.clip(stats.norm.cdf(eps), 1e-9, 1-1e-9), *ln_params)
    agg_dep = np.bincount(pol_idx, weights=Y_dep, minlength=n)
    agg_ind = np.bincount(pol_idx, weights=Y_ind, minlength=n)
    sd = float(np.quantile(agg_dep, SOLVENCY_ALPHA))
    si = float(np.quantile(agg_ind, SOLVENCY_ALPHA))
    # --- R1: margin-preserving independent arm ------------------------------
    Y_perm = rng.permutation(Y_dep)
    agg_perm = np.bincount(pol_idx, weights=Y_perm, minlength=n)
    sp = float(np.quantile(agg_perm, SOLVENCY_ALPHA))
    ok = sd > 1
    return {
        "total_pct": (si / sd - 1) * 100 if ok else np.nan,
        "margin_fixed_pct": (sp / sd - 1) * 100 if ok else np.nan,
        # claim-level mean severity ratio: dependent arm / unconditional arm
        "sev_mean_ratio": float(Y_dep.mean() / Y_ind.mean()),
        "sev_median_ratio": float(np.median(Y_dep) / np.median(Y_ind)),
    }


def attenuation_diagnostics(d, tau, rng) -> dict:
    """Range-restriction (selection) diagnosis — on the latent-variable scale.

    tau_pop       : τ(Z, Z_Y) over ALL policies, a single draw per policy
                    (as if every policy had one claim) → ≈ injected τ
    tau_claimant  : the same draws, ONLY policies with N ≥ 1 (one value per
                    policy; no claim-count weighting) → pure range restriction
    (claim-level latent τ and per-claim τ-b(N,Y) are computed inside main())
    tie_ceiling   : τ-b ceiling of the synthetic claim-count composition
                    sqrt(1 − Σ_k q_k²) and τ-a ceiling 1 − Σ_k q_k²,
                    q_k = share of claims on policies with count k.
    RNG: drawn only AFTER recover_scr → the preceding numbers do not change.
    """
    Z, N = d["Z"], d["N"]
    rho = np.sin(np.pi * tau / 2)
    Z_Y_all = rho * Z + np.sqrt(max(1 - rho**2, 1e-12)) * rng.normal(size=len(Z))
    tau_pop = float(stats.kendalltau(Z, Z_Y_all)[0])
    m = N > 0
    tau_claimant = float(stats.kendalltau(Z[m], Z_Y_all[m])[0])
    _, cnt = np.unique(d["pcN"], return_counts=True)
    q = cnt / cnt.sum()
    s2 = float(np.sum(q ** 2))
    return {"tau_pop": tau_pop, "tau_claimant": tau_claimant,
            "tie_ceiling_taub": float(np.sqrt(1 - s2)),
            "tie_ceiling_taua": float(1 - s2),
            "share_no_claim": float(np.mean(~m))}


def observed_tie_ceiling():
    """Count-composition ceiling of the observed portfolio (aggregate only; optional)."""
    try:
        p = pd.read_parquet(DATA_PROC_DIR / "analytical_dataset.parquet",
                            columns=["n_claims"])
        k = p["n_claims"].to_numpy()
        k = k[k > 0].astype(int)
        vals, npol = np.unique(k, return_counts=True)
        q = vals * npol / float(np.sum(vals * npol))
        s2 = float(np.sum(q ** 2))
        return {"source": "analytical_dataset.parquet (n_claims)",
                "n_claims_total": int(np.sum(vals * npol)),
                "tie_ceiling_taub": float(np.sqrt(1 - s2)),
                "tie_ceiling_taua": float(1 - s2)}
    except Exception as e:  # pragma: no cover — diagnostic purpose, not mandatory
        return {"source": None, "error": str(e)}


def _ratio(x, tau):
    return float(x / tau) if tau > 0 else np.nan


def _json_clean(o):
    """NaN/inf → None (strict JSON); numpy scalars → Python types."""
    if isinstance(o, dict):
        return {k: _json_clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_clean(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    return o


def draw_figure(df: pd.DataFrame, path_stem: Path) -> list:
    """Figure 3 — print size (FIG_W_FULL), two panels."""
    apply_pub_style()
    C_LAT, C_PC = "#2a78d6", "#eb6834"          # (a) latent / per-claim
    C_TOT, C_FIX = "#a3a29c", "#4a3aa7"         # (b) total / margin-fixed
    MINUS = "−"
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_FULL, 2.6),
                             gridspec_kw={"width_ratios": [1, 1.25]})
    # ---------------- (a) recovery ----------------
    ax = axes[0]
    tmax = max(TAU_INJECT_GRID)
    ax.plot([0, tmax], [0, tmax], ls="--", color="0.35", lw=0.8,
            label="Perfect recovery", zorder=1)
    ax.errorbar(df["tau_inject"], df["tau_recovered_perclaim"],
                yerr=ERRBAR_Z * df["tau_perclaim_sd"], fmt="D-", ms=4,
                color=C_PC, capsize=2.5, lw=1.2, label="Per-claim, τ-b(N, Y)",
                zorder=3)
    ax.errorbar(df["tau_inject"], df["tau_recovered_latent"],
                yerr=ERRBAR_Z * df["tau_latent_sd"], fmt="o-", ms=4,
                color=C_LAT, capsize=2.5, lw=1.2, label="Latent, claim level",
                zorder=4)
    ax.set_xlim(-0.02, tmax + 0.02); ax.set_ylim(-0.02, tmax + 0.02)
    ax.set_xticks(TAU_INJECT_GRID)
    ax.set_xticklabels([f"{t:.2f}" for t in TAU_INJECT_GRID])
    ax.set_xlabel("Injected Kendall's τ")
    ax.set_ylabel("Recovered Kendall's τ")
    ax.set_title("(a) Recovery of injected dependence", loc="left")
    ax.legend(loc="upper left", handlelength=2.2)
    # ---------------- (b) capital effect ----------------
    ax = axes[1]
    x = np.arange(len(df)); w = 0.36
    series = [
        ("scr_bias_pct", "scr_bias_sd", C_TOT, -w / 2,
         "Total effect\n(unconditional severity margin)"),
        ("scr_marginfixed_pct", "scr_marginfixed_sd", C_FIX, w / 2,
         "Dependence effect\n(claim-severity margin fixed)"),
    ]
    for col, sdcol, c, dx, lab in series:
        y = df[col].to_numpy(); e = ERRBAR_Z * df[sdcol].to_numpy()
        ax.bar(x + dx, y, width=w, color=c, label=lab, zorder=2,
               edgecolor="white", linewidth=0.6)
        ax.errorbar(x + dx, y, yerr=e, fmt="none", ecolor="0.15",
                    elinewidth=0.8, capsize=2, zorder=3)
        for xi, yi, ei in zip(x + dx, y, e):
            if abs(yi) < 0.05:
                txt = "0.0"
            else:
                txt = f"{MINUS if yi < 0 else '+'}{abs(yi):.1f}"
            ybase = yi - ei if yi <= 0 else yi + ei
            ax.annotate(txt, (xi, ybase), xytext=(0, -2 if yi <= 0 else 2),
                        textcoords="offset points", ha="center",
                        va="top" if yi <= 0 else "bottom", fontsize=6,
                        zorder=4, bbox=dict(boxstyle="square,pad=0.08",
                                            fc="white", ec="none"))
    ax.axhline(0, color="0.2", lw=0.7, zorder=2)
    ax.axhline(-SCR_MATERIAL_PCT, color="0.35", lw=0.7, ls=":", zorder=1,
               label=f"Materiality threshold ({MINUS}{SCR_MATERIAL_PCT:.0f}%)")
    ax.set_xticks(x); ax.set_xticklabels([f"{t:.2f}" for t in df["tau_inject"]])
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    ymin = float(np.nanmin(df["scr_bias_pct"] - ERRBAR_Z * df["scr_bias_sd"]))
    # bottom margin: keep the legend (lower left) from overlapping bars and labels
    ybot = min(-110.0, 10 * np.floor((ymin - 40) / 10))
    ax.set_ylim(ybot, 5)
    ax.set_yticks(np.arange(0, ybot - 1, -20))
    ax.set_xlabel("Injected Kendall's τ")
    ax.set_ylabel("Change in policy-level\n99.5% loss quantile (%)")
    ax.set_title("(b) Capital effect of assuming independence", loc="left")
    ax.legend(loc="lower left", handlelength=1.4, borderaxespad=0.3,
              labelspacing=0.5)
    ax.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=1.5)
    out = []
    for ext in ("pdf", "png"):
        fp = path_stem.with_suffix(f".{ext}")
        fig.savefig(fp); out.append(str(fp.resolve()))
    plt.close(fig)
    return out


def main() -> None:
    global log
    log = setup_logger(); apply_pub_style()
    log.info("=" * 78)
    log.info(f"SCRIPT {SCRIPT_NAME} — Positive Control")
    log.info("=" * 78)

    outfiles: list = []
    nb, ln, src = get_marginals()
    log.info(f"\n[1/3] Marginals ({src}): NegBin{np.round(nb,3)}, "
             f"LogNormal{np.round(ln,2)}")
    log.info(f"  → Regime B uses these marginals; only τ is injected.")
    p0 = float(stats.nbinom.pmf(0, nb[0], nb[1]))
    z0 = float(stats.norm.ppf(p0))
    log.info(f"  P(N=0)={p0:.4f} → claims occur only for Z > Φ⁻¹(P(N=0)) = {z0:.3f} "
             f"(the top {100*(1-p0):.1f}% of the latent risk)")

    log.info(f"\n[2/3] Two-regime scan (R={POSCTRL_N_REP} independent "
             f"replications per τ level)")
    # R independent synthetic portfolios at each injection level: mean recovery,
    # bias, RMSE, Type-I error at τ=0 / power at positive τ via the iid z test.
    rows, rep_detail = [], []
    for tau in TAU_INJECT_GRID:
        lat, pc, scrs, rejs = [], [], [], []
        fix, mratio, medratio = [], [], []
        pop, clm, ceil_b, ceil_a, share0 = [], [], [], [], []
        for r_i in range(POSCTRL_N_REP):
            rng = np.random.default_rng(GLOBAL_SEED + 137 * r_i + int(tau * 1e4))
            d = generate_regime(N_SYNTH, tau, nb, ln, rng)
            if d is None:
                continue
            t_lat = recover_tau_latent(d)
            t_pc = float(stats.kendalltau(d["pcN"], d["Y"])[0])
            n_cl = len(d["Y"])
            se0 = np.sqrt(2 * (2 * n_cl + 5) / (9 * n_cl * (n_cl - 1)))
            rejs.append(abs(t_pc) / se0 > stats.norm.ppf(0.975))
            lat.append(t_lat); pc.append(t_pc)
            sc = recover_scr(d, tau, ln, rng)
            scrs.append(sc["total_pct"]); fix.append(sc["margin_fixed_pct"])
            mratio.append(sc["sev_mean_ratio"]); medratio.append(sc["sev_median_ratio"])
            ad = attenuation_diagnostics(d, tau, rng)
            pop.append(ad["tau_pop"]); clm.append(ad["tau_claimant"])
            ceil_b.append(ad["tie_ceiling_taub"]); ceil_a.append(ad["tie_ceiling_taua"])
            share0.append(ad["share_no_claim"])
            rep_detail.append({"tau_inject": tau, "rep": r_i,
                               "tau_latent": t_lat, "tau_perclaim": t_pc})
        (lat, pc, scrs, fix, mratio, medratio, pop, clm, ceil_b, ceil_a,
         share0) = map(np.asarray, (lat, pc, scrs, fix, mratio, medratio, pop,
                                    clm, ceil_b, ceil_a, share0))
        R = len(lat)
        rej_rate = float(np.mean(rejs))
        rows.append({
            "tau_inject": tau, "n_rep": int(R),
            "tau_recovered_latent": float(lat.mean()),
            "tau_latent_sd": float(lat.std(ddof=1)),
            "tau_recovered_perclaim": float(pc.mean()),
            "tau_perclaim_sd": float(pc.std(ddof=1)),
            "bias_latent": float(lat.mean() - tau),
            "rmse_latent": float(np.sqrt(np.mean((lat - tau) ** 2))),
            # TOTAL effect (original; against the unconditional severity marginal)
            "scr_bias_pct": float(scrs.mean()),
            "scr_bias_sd": float(scrs.std(ddof=1)),
            # rejection rate at τ=0 = Type-I error; at τ>0 = power; Clopper-Pearson CI
            "reject_rate": rej_rate,
            "reject_rate_ci95": _clopper_pearson(int(round(rej_rate*R)), R),
            # --- R1 (a): margin-fixed dependence effect -------------------
            "scr_marginfixed_pct": float(fix.mean()),
            "scr_marginfixed_sd": float(fix.std(ddof=1)),
            "scr_marginfixed_se": float(fix.std(ddof=1) / np.sqrt(R)),
            "scr_marginfixed_frac_material": float(np.mean(np.abs(fix) > SCR_MATERIAL_PCT)),
            "sev_mean_ratio": float(mratio.mean()),
            "sev_mean_ratio_sd": float(mratio.std(ddof=1)),
            "sev_median_ratio": float(medratio.mean()),
            # --- R1 (b): attenuation diagnosis ----------------------------
            "tau_pop_latent": float(pop.mean()),
            "tau_pop_latent_sd": float(pop.std(ddof=1)),
            "tau_claimant_policy": float(clm.mean()),
            "tau_claimant_policy_sd": float(clm.std(ddof=1)),
            "ratio_pop_latent": _ratio(pop.mean(), tau),
            "ratio_claimant_policy": _ratio(clm.mean(), tau),
            "ratio_latent": _ratio(lat.mean(), tau),
            "ratio_perclaim": _ratio(pc.mean(), tau),
            "tie_ceiling_taub": float(ceil_b.mean()),
            "tie_ceiling_taua": float(ceil_a.mean()),
            "share_no_claim": float(share0.mean()),
        })
        log.info(f"  τ_inject={tau:.2f} (R={R}): "
                 f"τ̄_latent={lat.mean():+.4f}±{lat.std(ddof=1):.4f} "
                 f"(bias={lat.mean()-tau:+.4f}, RMSE={np.sqrt(np.mean((lat-tau)**2)):.4f}), "
                 f"τ̄_perclaim={pc.mean():+.4f}, SCR={scrs.mean():+.1f}%±{scrs.std(ddof=1):.1f}, "
                 f"{'Type-I' if tau == 0 else 'power'}={rej_rate:.3f}")
        log.info(f"      [R1a] SCR margin-fixed={fix.mean():+.2f}%±{fix.std(ddof=1):.2f} "
                 f"(SE {fix.std(ddof=1)/np.sqrt(R):.2f}; |effect|>{SCR_MATERIAL_PCT:.0f}% "
                 f"share of replications={np.mean(np.abs(fix) > SCR_MATERIAL_PCT):.2f}); "
                 f"claim mean severity ratio dependent/unconditional={mratio.mean():.3f}"
                 f"±{mratio.std(ddof=1):.3f} (median ratio {medratio.mean():.3f})")
        log.info(f"      [R1b] τ_pop={pop.mean():+.4f}, τ_claimant-policy={clm.mean():+.4f}, "
                 f"τ_claim-latent={lat.mean():+.4f}, τ-b_perclaim={pc.mean():+.4f}; "
                 f"ratios={_ratio(pop.mean(), tau):.3f}/{_ratio(clm.mean(), tau):.3f}/"
                 f"{_ratio(lat.mean(), tau):.3f}/{_ratio(pc.mean(), tau):.3f}; "
                 f"tie ceiling τ-b={ceil_b.mean():.4f} (τ-a {ceil_a.mean():.4f}); "
                 f"P(N=0)={share0.mean():.4f}")
    df = pd.DataFrame(rows)

    log.info("\n[3/3] Recovery verdict + figure")
    r0 = df[df["tau_inject"] == 0.0].iloc[0]
    rmax = df[df["tau_inject"] == max(TAU_INJECT_GRID)].iloc[0]
    taus = df.sort_values("tau_inject")["tau_recovered_latent"].values
    monotonic = bool(np.all(np.diff(taus) > 0))
    # Original (total effect) criteria — kept for traceability
    scr_material_total = bool(abs(rmax["scr_bias_pct"]) > SCR_MATERIAL_PCT)
    scr_null_total = bool(abs(r0["scr_bias_pct"]) < SCR_NULL_PCT)
    # R1: capital criteria on the MARGIN-FIXED effect (thresholds unchanged)
    scr_material = bool(abs(rmax["scr_marginfixed_pct"]) > SCR_MATERIAL_PCT)
    scr_null = bool(abs(r0["scr_marginfixed_pct"]) < SCR_NULL_PCT)
    recovers = monotonic and scr_material and scr_null
    recovers_total = monotonic and scr_material_total and scr_null_total
    mf_ci = [float(rmax["scr_marginfixed_pct"] - 1.96 * rmax["scr_marginfixed_se"]),
             float(rmax["scr_marginfixed_pct"] + 1.96 * rmax["scr_marginfixed_se"])]
    log.info(f"  Criteria: monotone={monotonic}; material (margin-fixed, "
             f"|{rmax['scr_marginfixed_pct']:+.2f}%| > {SCR_MATERIAL_PCT:.0f}%)="
             f"{scr_material} [95% MC interval for the mean {mf_ci[0]:+.2f}, "
             f"{mf_ci[1]:+.2f}]; null (margin-fixed, "
             f"|{r0['scr_marginfixed_pct']:+.2f}%| < {SCR_NULL_PCT:.0f}%)={scr_null}")
    log.info(f"  (Original total-effect criteria: material={scr_material_total} "
             f"[{rmax['scr_bias_pct']:+.1f}%], null={scr_null_total} "
             f"[{r0['scr_bias_pct']:+.1f}%])")
    if recovers:
        log.info(f"  >>> RESULT: The pipeline DETECTS strong association — latent τ monotone "
                 f"(0→{rmax['tau_recovered_latent']:.2f}), margin-fixed SCR effect under "
                 f"strong association {abs(rmax['scr_marginfixed_pct']):.1f}% "
                 f"(total {abs(rmax['scr_bias_pct']):.0f}%), in the negative control "
                 f"{abs(r0['scr_marginfixed_pct']):.2f}%.")
    else:
        log.info(f"  >>> WARNING: monotone={monotonic}, material(margin-fixed)="
                 f"{scr_material}, neg-null={scr_null}")

    outfiles += draw_figure(df, FIGURES_DIR / "09_positive_control")
    df_out = df.copy()
    df_out.to_csv(OUTPUTS_DIR / "09_recovery_table.csv", index=False)
    outfiles.append(str((OUTPUTS_DIR / "09_recovery_table.csv").resolve()))

    obs_ceiling = observed_tie_ceiling()
    pos = df[df["tau_inject"] > 0]
    atten = {
        "mechanism": ("selection / range restriction: severities exist only "
                      "for N >= 1, i.e. latent Z > Phi^-1(P(N=0)); the "
                      "population latent tau equals the injected tau, the "
                      "claim-sample tau does not. Count ties are not the "
                      "dominant source (latent measure has no count; tau-b "
                      "tie ceiling not binding)."),
        "p_no_claim_theoretical": p0,
        "latent_threshold_z0": z0,
        "per_level": [{k: r[k] for k in (
            "tau_inject", "tau_pop_latent", "tau_pop_latent_sd",
            "tau_claimant_policy", "tau_claimant_policy_sd",
            "tau_recovered_latent", "tau_recovered_perclaim",
            "ratio_pop_latent", "ratio_claimant_policy", "ratio_latent",
            "ratio_perclaim", "tie_ceiling_taub", "tie_ceiling_taua",
            "share_no_claim")} for r in rows],
        "ratio_ranges_positive_levels": {
            c: [float(pos[c].min()), float(pos[c].max())]
            for c in ("ratio_pop_latent", "ratio_claimant_policy",
                      "ratio_latent", "ratio_perclaim")},
        "tie_ceiling_synthetic": {
            "taub": float(df["tie_ceiling_taub"].mean()),
            "taua": float(df["tie_ceiling_taua"].mean()),
            "definition": "sqrt(1 - sum_k q_k^2), q_k = share of claims on "
                          "policies with count k (tau-a: 1 - sum_k q_k^2); "
                          "averaged over all replications and levels"},
        "tie_ceiling_observed": obs_ceiling,
    }
    log.info(f"  Tie ceiling: synthetic τ-b={atten['tie_ceiling_synthetic']['taub']:.4f} "
             f"(τ-a {atten['tie_ceiling_synthetic']['taua']:.4f}); observed "
             f"τ-b={obs_ceiling.get('tie_ceiling_taub', float('nan')):.4f} "
             f"(τ-a {obs_ceiling.get('tie_ceiling_taua', float('nan')):.4f})")

    manifest = {
        "script": SCRIPT_NAME,
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "marginals_source": src,
        "marginals": {"negbin_r_p": [float(v) for v in nb],
                      "lognormal_s_loc_scale": [float(v) for v in ln]},
        "recovery_table": rows,
        # R1: PRIMARY verdict — capital criteria on the margin-preserving arm
        "recovers_signal": bool(recovers),
        "recovers_signal_total_effect": bool(recovers_total),
        "recovery_criteria": {
            "monotonic_latent": monotonic,
            "material_capital_effect_margin_fixed": scr_material,
            "material_capital_effect_margin_fixed_value_pct": float(rmax["scr_marginfixed_pct"]),
            "material_capital_effect_margin_fixed_mc95_of_mean": mf_ci,
            "material_capital_effect_margin_fixed_frac_reps": float(rmax["scr_marginfixed_frac_material"]),
            "null_capital_effect_margin_fixed": scr_null,
            "null_capital_effect_margin_fixed_value_pct": float(r0["scr_marginfixed_pct"]),
            "material_capital_effect_total": scr_material_total,
            "null_capital_effect_total": scr_null_total,
            "threshold_material_pct": SCR_MATERIAL_PCT,
            "threshold_null_pct": SCR_NULL_PCT,
            "thresholds_note": "prespecified values unchanged from the original "
                               "submission; only the evaluated arm changed (R1)",
        },
        "capital_arms": {
            "dependent": "Y_dep = LN^{-1}(Phi(rho*Z_pol + sqrt(1-rho^2)*eps)), "
                         "Z_pol also drives N",
            "independent_total_effect": INDEP_ARM_TOTAL,
            "independent_margin_fixed": INDEP_ARM_MARGIN_FIXED,
            "metric": "(Q_0.995(indep policy aggregate) / Q_0.995(dep policy "
                      "aggregate) - 1) * 100; negative = understatement",
            "note": "the permutation breaks both the count-severity link and "
                    "within-policy co-movement of severities; the difference "
                    "between the total and margin-fixed effects is the shift "
                    "of the claim-level severity margin",
        },
        "attenuation_diagnostics": atten,
        "limitation": ("Regime B synthetic; internal validity only "
                       "(method recovers signal), not external."),
        "params": {"TAU_INJECT_GRID": TAU_INJECT_GRID, "N_SYNTH": N_SYNTH,
                   "GLOBAL_SEED": GLOBAL_SEED, "POSCTRL_N_REP": POSCTRL_N_REP,
                   "POSCTRL_NMAX": POSCTRL_NMAX,
                   "SOLVENCY_ALPHA": SOLVENCY_ALPHA,
                   "seed_scheme": "GLOBAL_SEED + 137*rep + int(tau*1e4); "
                                  "draw order: generate_regime -> capital eps "
                                  "-> permutation -> population-latent eps",
                   "SCR_MATERIAL_PCT": SCR_MATERIAL_PCT,
                   "SCR_NULL_PCT": SCR_NULL_PCT, "ERRBAR_Z": ERRBAR_Z,
                   "type1_test": "iid no-ties z for Kendall tau (unchanged)"},
        "output_files": outfiles,
        "environment": env_info(),
    }
    with open(OUTPUTS_DIR / "09_manifest.json", "w", encoding="utf-8") as f:
        json.dump(_json_clean(manifest), f, ensure_ascii=False, indent=2,
                  allow_nan=False)
    log.info("\n" + "=" * 78)
    log.info("SCRIPT 09 COMPLETED")
    log.info("=" * 78)


if __name__ == "__main__":
    main()
