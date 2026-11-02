"""
================================================================================
SCRIPT 18 — RORAC-Targeted Technical Premium (Premium Principles + Euler Allocation)
================================================================================
PURPOSE
    Converts the validated pure-premium risk score (14b best model) into an
    ECONOMICALLY OPTIMAL premium. The decision layer that actually delivers the
    thesis/title claim of "Optimal Premium Pricing": how should the risk loading
    be set, and what is the optimal loading under a target return on capital (RORAC)?

METHOD
    • Risk score m(x): the best model's per-policy pure-premium prediction (14b).
    • Process variance: COMPOUND-POISSON, with the common severity empirical moment:
      V_i = m_i·E[Y²]/E[Y] (the heavy tail — GPD ξ≈0.31 — is carried via the moment).
    • Portfolio SCR: VaR_99.5−E[S] distance of the independent-sum aggregate
      distribution, Cornish-Fisher (mean/variance/skewness, compound-Poisson cumulants).
    • RORAC-OPTIMAL TECHNICAL PREMIUM: π*_i=(m_i+ρ·K_i)/(1−expense), ρ=target RORAC;
      K_i = SCR·V_i/ΣV_j Euler (covariance) allocation — since the frequency-severity
      association is negligible (Section 6), the independent-sum allocation is consistent.
      Frontier: the (combined ratio, RORAC) curve on a grid of target returns ρ.
      NOTE: the total premium depends only on the aggregate margin; the CHOICE of
      principle changes the DISTRIBUTION, not the total → the frontier is a single
      curve (an honest finding).
    • Principle DIFFERENTIATION: the same total risk margin M=ρ·SCR is distributed to
      policies with five principle 'shapes' (EV∝m, SD∝√V, variance∝V, entropic, CoC∝K);
      loading progressivity across risk deciles is compared.
      Under compound-Poisson V∝m ⇒ EV/variance/CoC coincide as risk-proportional; SD ∝√m
      remains regressive — this is where the practical effect of the principle choice shows.
    • Comparison: the premium ACTUALLY collected (Kasko_Prim_TL) → RELATIVE risk-decile
      adequacy (independent of the nominal-premium/real-loss basis mismatch):
      does the market price high risk adequately?

INPUT : data/proc/pricing_best_preds.parquet (14b) ; outputs/13_manifest.json
OUTPUT: outputs/18_optimal_pricing.csv ; outputs/18_adequacy.csv ;
        figures/18_*.png ; outputs/18_manifest.json
================================================================================
"""
from __future__ import annotations
import sys
import json
import logging
import platform
from pathlib import Path
from datetime import datetime

# --- Spyder-compatible path setup ---
try:
    _HERE = Path(__file__).resolve().parent
except NameError:
    _HERE = Path.cwd()
for _cand in (_HERE, _HERE.parent):
    if (_cand / "config.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy import stats

from config import (
    DATA_PROC_DIR, DATA_RAW_DIR, OUTPUTS_DIR, FIGURES_DIR, LOGS_DIR, FIG_DPI,
    SOLVENCY_ALPHA, COC_RATE, EXPENSE_RATIO, TARGET_RORAC, LOADING_GRID_N,
    PRICING_SCR_MC_REPS, PREMIUM_DEFLATE, INFLATION_BASE_YEAR_MONTH,
    GLOBAL_SEED, NB_DISP_BOOT, PRICING_LEVEL_ADJUSTMENT,
    apply_pub_style, FIG_W_TEXT, env_info,
)

SCRIPT_NAME = "18_optimal_pricing"

# --- R1 revision constants ---------------------------------------------------
# (1) Figure 9 (18_adequacy) is drawn at print size (single panel, text width).
# (2) Monte Carlo standard error of the MC capital cross-check: bootstrap of the
#     PRICING_SCR_MC_REPS aggregate draws with its OWN generator (seed offset
#     +8), so the MC draws (seed +7) and the dispersion bootstrap (seed +21)
#     are untouched -> no estimate changes. A normal-approximation SE is also
#     recorded: SE(q_hat) = sqrt(a(1-a)/R) / phi(z_a) * sigma.
# (3) Traceability: every Section 3.7 number is written to the manifest
#     block "section_3_7_trace"; the target RORAC is exported as the hurdle
#     rate h (same value as config TARGET_RORAC).
R1_FIG9_HEIGHT_IN = 3.1           # Figure 9 height (in); width = FIG_W_TEXT
R1_SCR_MC_SE_BOOT_B = 2000        # bootstrap replicates for the MC-SE
R1_SCR_MC_SE_SEED_OFFSET = 8      # rng = default_rng(GLOBAL_SEED + 8)
R1_COLOR_MARKET = "#3182BD"       # actual (market) premium
R1_COLOR_TARGET = "#D7301F"       # RORAC-target premium


def gini_14b(y_true, y_pred):
    """Normalised pure-premium Gini, identical to 14b_pricing_models.gini()."""
    order = np.argsort(y_pred); ys = np.asarray(y_true)[order]
    cum = np.cumsum(ys) / (np.sum(ys) + 1e-12); n = len(y_true)
    g = 1 - 2 * np.sum(cum) / n
    op = np.argsort(y_true); yp = np.asarray(y_true)[op]
    cp = np.cumsum(yp) / (np.sum(yp) + 1e-12)
    gp = 1 - 2 * np.sum(cp) / n
    return float(g / gp) if gp != 0 else 0.0


def setup_logger():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(SCRIPT_NAME); lg.setLevel(logging.INFO); lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOGS_DIR / f"{SCRIPT_NAME}.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt); sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    lg.addHandler(fh); lg.addHandler(sh); return lg


log = logging.getLogger(SCRIPT_NAME)


def set_style():
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "savefig.dpi": FIG_DPI,
        "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25})


def cornish_fisher_var(mean, var, m3, alpha):
    """Cornish-Fisher VaR: skewness-corrected normal quantile.
    q_α = μ + σ[z + (z²−1)/6·γ], γ=skewness=m3/σ³."""
    sd = np.sqrt(max(var, 1e-12))
    gamma = m3 / (sd ** 3 + 1e-12)
    z = stats.norm.ppf(alpha)
    zcf = z + (z ** 2 - 1) / 6.0 * gamma
    return mean + sd * zcf, float(gamma)


def rorac_of(P, L_exp, scr, expense):
    """RORAC = expected underwriting profit / SCR."""
    uw = P * (1.0 - expense) - L_exp
    return float(uw / scr) if scr > 0 else np.nan


def main():
    global log
    log = setup_logger(); set_style(); apply_pub_style()
    log.info("=" * 78)
    log.info(f"SCRIPT {SCRIPT_NAME} — Optimal Premium Pricing (RORAC)")
    log.info("=" * 78)
    outfiles = []

    pth = DATA_PROC_DIR / "pricing_best_preds.parquet"
    if not pth.exists():
        log.error("pricing_best_preds.parquet is missing — run 14b first.")
        raise SystemExit(1)
    df = pd.read_parquet(pth)
    m_raw = np.clip(df["pp_pred"].values.astype(float), 1e-6, None)
    y = np.clip(df["pp_true"].values.astype(float), 0, None)
    # R2b (Reviewer 1, round 2, point 4b): the RORAC layer uses the selected
    # model's predictions ON-LEVELLED by the scalar factor chosen in config
    # (PRICING_LEVEL_ADJUSTMENT; estimated by 14b without test-year data). The
    # factor rescales the level only: ranking, Gini and the relative adequacy
    # diagnostics are unchanged; the raw-level headline is kept as a sensitivity.
    _col = f"pp_pred_onlevel_{PRICING_LEVEL_ADJUSTMENT}"
    if PRICING_LEVEL_ADJUSTMENT not in (None, "none") and _col in df.columns:
        m = np.clip(df[_col].values.astype(float), 1e-6, None)
        level_factor = float(np.median(m / m_raw))
    else:
        m = m_raw.copy(); level_factor = 1.0
    level_adjustment = {
        "config_PRICING_LEVEL_ADJUSTMENT": PRICING_LEVEL_ADJUSTMENT,
        "column_used": _col if level_factor != 1.0 else "pp_pred",
        "factor": level_factor,
        "PO_raw": float(m_raw.sum() / max(y.sum(), 1e-9)),
        "PO_used": float(m.sum() / max(y.sum(), 1e-9)),
        "expected_loss_raw": float(m_raw.sum()), "expected_loss_used": float(m.sum()),
        "actual_loss": float(y.sum()),
        "note": ("Level factor from 14b (no test-year information). Ranking-based "
                 "quantities (Gini, decile adequacy ratios, principle progressivity) "
                 "are invariant to the factor; the technical premium, E[S] and the SCR "
                 "scale with it."),
    }
    log.info(f"  R2b level adjustment: {PRICING_LEVEL_ADJUSTMENT} factor={level_factor:.4f}; "
             f"P/O raw={level_adjustment['PO_raw']:.3f} → used={level_adjustment['PO_used']:.3f}")
    mkt = pd.to_numeric(df.get("market_premium"), errors="coerce").values.astype(float)
    # R1: risk-score provenance. 14b writes pricing_best_preds.parquet from the
    # model with the highest VALIDATION-year pure-premium Gini among the pure-premium,
    # two-stage and optional-package models; we re-derive the Gini here and
    # check it against the 14b manifest (aggregate only).
    risk_score_source = {
        "file": "data/proc/pricing_best_preds.parquet",
        "written_by": "14b_pricing_models",
        "selection_rule_14b": ("argmax of VALIDATION-year (2016) pure-premium "
                               "Gini over models with tip in {pure-prem, "
                               "iki-asamali (two-stage), opsiyonel}; frequency-"
                               "only GLMs excluded; test years never used (R1b)"),
        "gini_recomputed_from_parquet": gini_14b(y, m),
    }
    try:
        m14b = json.load(open(OUTPUTS_DIR / "14b_manifest.json", encoding="utf-8"))
        risk_score_source["best_model_14b"] = m14b.get("best_model")
        risk_score_source["calibration_model_14b"] = (
            m14b.get("calibration", {}) or {}).get("best_model")
        risk_score_source["best_gini_14b"] = m14b.get("best_gini")
        _g14 = m14b.get("best_gini")
        risk_score_source["gini_matches_14b"] = (
            bool(abs(risk_score_source["gini_recomputed_from_parquet"]
                     - float(_g14)) < 1e-9) if _g14 is not None else None)
    except Exception as _e:
        risk_score_source["best_model_14b"] = None
        risk_score_source["error"] = str(_e)
    log.info(f"  Risk score source: {risk_score_source.get('best_model_14b')} "
             f"(Gini parquet={risk_score_source['gini_recomputed_from_parquet']:.5f}, "
             f"14b={risk_score_source.get('best_gini_14b')}, "
             f"match={risk_score_source.get('gini_matches_14b')})")
    defl_mean = None
    # Premiums are collected in NOMINAL terms; losses are on a 2018-12 real basis.
    # The premium is deflated to 2018-12 with the MID-YEAR (June) CPI of the
    # policy year. The absolute market LR is still NOT INTERPRETED (the coverage
    # semantics of the premium column are unclear); only RELATIVE (portfolio-normalised) adequacy is read.
    premium_basis = "nominal"
    if PREMIUM_DEFLATE:
        try:
            # same robust loader as 02/04: xlsx candidate sheets → csv fallback
            from config import CPI_CANDIDATES, CPI_SHEET_CANDIDATES
            cpi = None
            for fn in CPI_CANDIDATES:
                fp = DATA_RAW_DIR / fn
                if not fp.exists():
                    continue
                if fp.suffix in (".xlsx", ".xls"):
                    xl = pd.ExcelFile(fp)
                    for sh in CPI_SHEET_CANDIDATES + xl.sheet_names:
                        try:
                            cand = pd.read_excel(fp, sheet_name=sh)
                        except Exception:
                            continue
                        if {"year_month", "cpi_index"} <= set(cand.columns):
                            cpi = cand
                            break
                else:
                    try:
                        cand = pd.read_csv(fp)
                        if {"year_month", "cpi_index"} <= set(cand.columns):
                            cpi = cand
                    except Exception:
                        pass
                if cpi is not None:
                    break
            if cpi is None:
                raise FileNotFoundError("No CPI source with year_month+cpi_index was found")
            dtx = pd.to_datetime(cpi["year_month"], errors="coerce")
            cpi["ym"] = (dtx.dt.year * 100 + dtx.dt.month).astype("Int64")
            cmap_cpi = dict(zip(cpi["ym"].astype(int),
                                pd.to_numeric(cpi["cpi_index"],
                                              errors="coerce")))
            base = float(cmap_cpi[int(INFLATION_BASE_YEAR_MONTH)])
            py = pd.to_numeric(df["policy_year"], errors="coerce").values
            defl = np.array([base / cmap_cpi.get(int(y) * 100 + 6, np.nan)
                             if np.isfinite(y) else np.nan for y in py])
            mkt = mkt * defl
            defl_mean = float(np.nanmean(defl))
            premium_basis = "real (Dec-2018, mid-year CPI)"
            log.info(f"  Market premium deflated to the 2018-12 real basis "
                     f"(policy-year mid-year CPI; mean multiplier="
                     f"{np.nanmean(defl):.3f})")
        except Exception as e:
            log.warning(f"  Premium deflation skipped: {e}")
    log.info(f"\n[1/5] Risk score: n={len(m):,} policies (test)")

    # Process variance: COMPOUND-POISSON (common severity) with the empirical moment.
    # Var(loss_i)=λ_i·E[Y²], λ_i=m_i/E[Y] ⇒ V_i = m_i·E[Y²]/E[Y]. The heavy tail
    # (GPD ξ=0.31) is carried into capital via the E[Y²]/E[Y] ratio. Moments come from
    # the ACTUAL claim severities (claim_analytical) — the arbitrariness of Tweedie-φ is avoided.
    cpath = DATA_PROC_DIR / "claim_analytical.parquet"
    claim = (pd.read_parquet(cpath) if cpath.exists()
             else pd.read_pickle(DATA_PROC_DIR / "claim_analytical.pkl"))
    sev = claim["severity_real"].values.astype(float); sev = sev[sev > 0]
    ES1 = float(np.mean(sev)); ES2 = float(np.mean(sev ** 2)); ES3 = float(np.mean(sev ** 3))
    ratio2 = ES2 / ES1                       # Var(loss)/E[loss]  (currency units)
    ratio3 = ES3 / ES1                       # 3rd cumulant / E[loss]
    V = m * ratio2                           # per-policy variance (∝ expected loss)
    sd = np.sqrt(V)
    mbar = float(np.mean(m))
    with np.errstate(divide="ignore", invalid="ignore"):
        k_gamma = np.where(V > 0, m ** 2 / V, 1e6)      # Gamma shape
        th_gamma = np.where(m > 0, V / m, ratio2)       # Gamma scale (=ratio2)
    log.info(f"  Compound-Poisson: E[Y]={ES1:,.0f}, E[Y²]/E[Y]={ratio2:,.0f}; "
             f"mean pure-prem={mbar:,.0f}")

    log.info("\n[2/5] Portfolio SCR (compound-Poisson, independent-sum, Cornish-Fisher)")
    agg_mean = float(np.sum(m))
    agg_var = float(ratio2 * np.sum(m))              # = Σ V_i
    agg_m3 = float(ratio3 * np.sum(m))               # compound-Poisson 3rd cumulant
    scr_var, gamma_cf = cornish_fisher_var(agg_mean, agg_var, agg_m3, SOLVENCY_ALPHA)
    scr = float(scr_var - agg_mean)
    log.info(f"  E[S]={agg_mean:,.0f}, σ={np.sqrt(agg_var):,.0f}, skewness γ={gamma_cf:.3f}")
    log.info(f"  SCR(CF, α={SOLVENCY_ALPHA})={scr:,.0f} ({100*scr/agg_mean:.2f}% E[S])")
    # With a heavy tail (ξ=0.31) Cornish-Fisher is not left on its own;
    # cross-validate with compound-Poisson MC: K~Poisson(Λ), severities are
    # resampled from the ACTUAL claim file (consistent with the empirical moments).
    rng_mc = np.random.default_rng(GLOBAL_SEED + 7)
    Lam = agg_mean / ES1
    aggs = np.empty(PRICING_SCR_MC_REPS)
    for r_i in range(PRICING_SCR_MC_REPS):
        k = rng_mc.poisson(Lam)
        aggs[r_i] = sev[rng_mc.integers(0, len(sev), k)].sum()
    scr_mc = float(np.quantile(aggs, SOLVENCY_ALPHA) - aggs.mean())
    log.info(f"  MC cross-check (R={PRICING_SCR_MC_REPS}): "
             f"SCR_MC={scr_mc:,.0f} (CF/MC ratio={scr/scr_mc:.3f})")
    # R1: Monte Carlo standard error of the MC capital distance (reporting
    # only; separate generator, the MC draws above are not re-used by any
    # estimate). (a) normal approximation for the quantile estimator,
    # (b) bootstrap of the R aggregate draws.
    z_a = float(stats.norm.ppf(SOLVENCY_ALPHA))
    agg_sd = float(np.sqrt(agg_var))
    se_q_normal = float(np.sqrt(SOLVENCY_ALPHA * (1 - SOLVENCY_ALPHA)
                                / PRICING_SCR_MC_REPS)
                        / stats.norm.pdf(z_a) * agg_sd)
    rng_se = np.random.default_rng(GLOBAL_SEED + R1_SCR_MC_SE_SEED_OFFSET)
    scr_mc_boot = np.empty(R1_SCR_MC_SE_BOOT_B)
    for b_i in range(R1_SCR_MC_SE_BOOT_B):
        ab = aggs[rng_se.integers(0, PRICING_SCR_MC_REPS, PRICING_SCR_MC_REPS)]
        scr_mc_boot[b_i] = np.quantile(ab, SOLVENCY_ALPHA) - ab.mean()
    se_scr_mc_boot = float(np.std(scr_mc_boot, ddof=1))
    n_above_q = int(np.sum(aggs > np.quantile(aggs, SOLVENCY_ALPHA)))
    cf_mc_gap_in_se = float((scr - scr_mc) / se_scr_mc_boot)
    # Leading Cornish-Fisher skewness term as a share of the capital distance
    skew_term = float((z_a ** 2 - 1) / 6.0 * gamma_cf)
    skew_share = float(skew_term / (z_a + skew_term))
    scr_mc_se = {
        "mc_reps": int(PRICING_SCR_MC_REPS),
        "n_draws_above_q995": n_above_q,
        "se_normal_approx": se_q_normal,
        "se_normal_approx_pct_of_scr_mc": float(se_q_normal / scr_mc),
        "se_bootstrap": se_scr_mc_boot,
        "se_bootstrap_pct_of_scr_mc": float(se_scr_mc_boot / scr_mc),
        "bootstrap_B": int(R1_SCR_MC_SE_BOOT_B),
        "bootstrap_seed": int(GLOBAL_SEED + R1_SCR_MC_SE_SEED_OFFSET),
        "cf_minus_mc": float(scr - scr_mc),
        "cf_minus_mc_in_bootstrap_se": cf_mc_gap_in_se,
        "method": ("(a) sqrt(a(1-a)/R)/phi(z_a)*sigma_S (normal approx. for "
                   "the empirical quantile); (b) SD of q_0.995-mean over B "
                   "bootstrap resamples of the R MC aggregate draws"),
    }
    log.info(f"  MC standard error: normal-approximation={se_q_normal:,.0f} "
             f"({100*se_q_normal/scr_mc:.1f}% SCR_MC), bootstrap={se_scr_mc_boot:,.0f} "
             f"({100*se_scr_mc_boot/scr_mc:.1f}%); CF−MC={scr-scr_mc:,.0f} "
             f"(={cf_mc_gap_in_se:.2f} SE); draws above q={n_above_q}")
    log.info(f"  CF skewness term (γ/6)(z²−1)={skew_term:.4f}σ → "
             f"{100*skew_share:.2f}% of the capital distance")
    # Frequency-model sensitivity: the marginal choice is NB (overdispersion 1.23),
    # while the main variance identity is compound-Poisson. With the general compound
    # variance Var(S)=E[N]Var(Y)+Var(N)E[Y]², under NB V_i^NB = m_i·[E[Y²]+
    # (disp−1)·E[Y]²]/E[Y]. With skewness ≈0.03 the quantile ~ mean+z·σ →
    # the SCR scales approximately with σ. Parameter uncertainty: conservative
    # scenario with the policy-bootstrap 97.5% upper bound of the dispersion.
    try:
        disp = json.load(open(OUTPUTS_DIR / '10_manifest.json',
                              encoding='utf-8'))['frequency']['dispersion']
    except Exception:
        disp = 1.233
    ana_nc = None
    try:
        ana_nc = pd.read_parquet(DATA_PROC_DIR / 'analytical_dataset.parquet',
                                 columns=['n_claims'])['n_claims'].values
    except Exception:
        pass
    if ana_nc is not None:
        rng_d = np.random.default_rng(GLOBAL_SEED + 21)
        n_pol_d = len(ana_nc)
        disps = np.empty(NB_DISP_BOOT)
        for b_i in range(NB_DISP_BOOT):
            samp = ana_nc[rng_d.integers(0, n_pol_d, n_pol_d)]
            mu_s = samp.mean()
            disps[b_i] = samp.var() / mu_s if mu_s > 0 else np.nan
        disp_hi = float(np.nanpercentile(disps, 97.5))
    else:
        disp_hi = disp * 1.05
    var_scale_nb = 1.0 + (disp - 1.0) * ES1 / ratio2       # Var_NB/Var_Pois
    var_scale_hi = 1.0 + (disp_hi - 1.0) * ES1 / ratio2
    scr_nb = scr * np.sqrt(var_scale_nb)
    scr_nb_hi = scr * np.sqrt(var_scale_hi)
    freq_var_sens = {
        'dispersion': float(disp), 'dispersion_boot_hi97_5': disp_hi,
        'scr_poisson': float(scr), 'scr_negbin': float(scr_nb),
        'scr_negbin_param_unc': float(scr_nb_hi),
        'nb_over_poisson': float(scr_nb / scr),
        'note': 'kuantil≈ort+z·σ (çarpıklık≈0.03) → SCR σ ile ölçeklenir'}
    log.info(f"  Frequency-variance sensitivity: SCR Poisson={scr:,.0f} | "
             f"NB(disp={disp:.3f})={scr_nb:,.0f} (×{scr_nb/scr:.3f}) | "
             f"NB+param(disp={disp_hi:.3f})={scr_nb_hi:,.0f} "
             f"(×{scr_nb_hi/scr:.3f})")
    # Euler (covariance/independence) allocation: K_i = SCR·V_i/ΣV_j, ΣK_i=SCR
    K = scr * V / (np.sum(V) + 1e-12)

    log.info("\n[3/5] RORAC-targeted technical premium + frontier")
    L_exp = float(np.sum(m))         # expected total loss
    L_real = float(np.sum(y))        # realized total loss
    e = EXPENSE_RATIO
    # The technical (office) premium is grossed up for expenses: office=risk_premium/(1−e).
    # RORAC-optimal technical premium: π*_i=(m_i+ρ·K_i)/(1−e), ρ=target RORAC ⇒ portfolio
    # RORAC=ρ (by construction). Total risk+profit margin above expected loss is ρ·SCR.
    rho = TARGET_RORAC
    pi_opt = (m + rho * K) / (1.0 - e)
    P_opt = float(np.sum(pi_opt))
    rorac_opt = (P_opt * (1 - e) - L_exp) / scr
    log.info(f"  RORAC-targeted technical premium (ρ={rho}): total={P_opt:,.0f}, "
             f"LR(expected)={L_exp/P_opt:.3f}, LR(realized)={L_real/P_opt:.3f}, "
             f"RORAC(by construction)={rorac_opt:.3f}")
    log.info(f"  Capital risk margin = ρ·SCR = {rho*scr:,.0f} "
             f"({100*rho*scr/P_opt:.3f}% of premium) — small in attritional motor; "
             f"the premium is dominated by expected loss+expenses")

    # Frontier: a SINGLE (LR, RORAC) curve on the grid of target returns ρ. The total
    # premium (hence LR/RORAC) depends only on the aggregate margin; the CHOICE of
    # principle changes only the DISTRIBUTION (not the aggregate) → the frontier is a single line.
    rhos = np.linspace(0.0, 0.5, LOADING_GRID_N)
    frontier = []
    for r in rhos:
        P = (L_exp + r * scr) / (1 - e)
        frontier.append({"target_rorac": float(r), "premium": float(P),
                         "loss_ratio_exp": float(L_exp / P),
                         "loss_ratio_real": float(L_real / P),
                         "combined_ratio": float(L_exp / P + e), "rorac": float(r)})

    log.info("\n[3b/5] Principle DIFFERENTIATION (same total premium, different distribution)")
    # The same total risk margin M=ρ·SCR is distributed with different 'shapes':
    #   EV∝m, SD∝sd, variance∝V, exponential∝entropic loading, CoC∝K(=Euler).
    # Under compound-Poisson V∝m, so EV/variance/CoC are risk-proportional; SD is
    # ∝√m (loads low risk relatively more = regressive). This is the differentiation gap.
    M = rho * scr
    ent = -(k_gamma) * np.log(np.clip(
        1 - th_gamma * (0.5 / (th_gamma.max() + 1e-12)), 1e-9, None))
    shapes = {"expected_value": m, "standard_deviation": sd, "variance": V,
              "exponential": np.clip(ent, 0, None), "cost_of_capital": K}
    order = np.argsort(m); deciles = np.array_split(order, 10)
    diff_rows = []; prin_alloc = {}
    for pr, shp in shapes.items():
        shp = np.clip(np.asarray(shp, float), 1e-12, None)
        margin_i = M * shp / shp.sum()
        office_i = (m + margin_i) / (1 - e)
        prin_alloc[pr] = office_i
        for d, ix in enumerate(deciles, 1):
            diff_rows.append({"principle": pr, "decile": d,
                              "mean_expected_loss": float(m[ix].mean()),
                              "mean_office_premium": float(office_i[ix].mean()),
                              "loading_pct": float(office_i[ix].sum()
                                                   / (m[ix].sum() + 1e-9) - 1)})
    ddf = pd.DataFrame(diff_rows)
    ddf.to_csv(OUTPUTS_DIR / "18_principle_differentiation.csv", index=False)
    outfiles.append(str((OUTPUTS_DIR / "18_principle_differentiation.csv").resolve()))
    prin_summary = {}
    for pr in shapes:
        sub = ddf[ddf["principle"] == pr].sort_values("decile")
        prog = float(sub["loading_pct"].iloc[-1] / (sub["loading_pct"].iloc[0] + 1e-9))
        prin_summary[pr] = {"progressivity_high_over_low": prog}
        log.info(f"  {pr:20s} loading progressivity (high/low risk decile)={prog:.2f}")

    log.info("\n[4/5] Actual (market) premium comparison + risk-decile adequacy")
    valid = np.isfinite(mkt) & (mkt > 0)
    market = None
    if valid.sum() > 0:
        P_mkt = float(np.sum(mkt[valid]))
        L_real_v = float(np.sum(y[valid]))
        L_exp_v = float(np.sum(m[valid]))
        market = {"n_valid": int(valid.sum()), "premium": P_mkt,
                  "premium_basis": premium_basis,
                  "loss_ratio_real": float(L_real_v / P_mkt),
                  "loss_ratio_exp": float(L_exp_v / P_mkt),
                  "absolute_lr_interpreted": False,
                  "note": ("mutlak LR yorumlanmaz (prim kolonu kapsam "
                           "semantiği belirsiz); yalnız göreli adequacy")}
        log.info(f"  Market premium (n={valid.sum():,}): total={P_mkt:,.0f}; "
                 f"the ABSOLUTE LR carries the nominal-premium/real-loss basis mismatch "
                 f"(={market['loss_ratio_real']:.2f}); interpretation via the RELATIVE metric.")
    else:
        log.info("  Actual premium column missing/empty — market comparison skipped")

    # Adequacy: realized LR by risk decile (market vs risk-based optimal).
    # Risk-based optimal premium = RORAC-optimal technical premium π*_i (Section [3/5]).
    pi_coc = pi_opt
    ad789 = []
    for d, ix in enumerate(deciles, 1):
        row = {"decile": d, "n": len(ix), "mean_pure_prem": float(m[ix].mean()),
               "mean_pred_loss": float(m[ix].mean()),
               "realized_loss": float(y[ix].sum()),
               "coc_premium": float(pi_coc[ix].sum()),
               "coc_loss_ratio": float(y[ix].sum() / (pi_coc[ix].sum() + 1e-9))}
        vv = valid[ix]
        if vv.sum() > 0:
            row["market_premium"] = float(mkt[ix][vv].sum())
            row["market_loss_ratio"] = float(y[ix][vv].sum()
                                             / (mkt[ix][vv].sum() + 1e-9))
        ad789.append(row)
    adq = pd.DataFrame(ad789)
    # Relative adequacy: LR normalised to the portfolio mean (a risk-differentiation
    # measure INDEPENDENT of the nominal-premium vs real-loss basis mismatch). In a
    # decile, relative LR>1 → that risk class is UNDER-priced (relative to the portfolio).
    coc_lr_port = float(y.sum() / (pi_coc.sum() + 1e-9))
    adq["coc_lr_rel"] = adq["coc_loss_ratio"] / coc_lr_port
    risk_diff_mkt = risk_diff_coc = None
    if "market_loss_ratio" in adq:
        mkt_lr_port = float(y[valid].sum() / (mkt[valid].sum() + 1e-9))
        adq["market_lr_rel"] = adq["market_loss_ratio"] / mkt_lr_port
        risk_diff_mkt = float(adq["market_lr_rel"].iloc[-1] / (adq["market_lr_rel"].iloc[0] + 1e-9))
        risk_diff_coc = float(adq["coc_lr_rel"].iloc[-1] / (adq["coc_lr_rel"].iloc[0] + 1e-9))
    adq.to_csv(OUTPUTS_DIR / "18_adequacy.csv", index=False)
    outfiles.append(str((OUTPUTS_DIR / "18_adequacy.csv").resolve()))
    if "market_loss_ratio" in adq:
        log.info(f"  Relative adequacy (portfolio-normalised) — market high/low "
                 f"risk LR ratio={risk_diff_mkt:.2f} (>1 ⇒ high risk relatively "
                 f"UNDER-priced); risk-based optimal ratio={risk_diff_coc:.2f} "
                 f"(≈1 ⇒ risk-fair)")

    log.info("\n[5/5] Saving + figures")
    fdf = pd.DataFrame(frontier)
    fdf.to_csv(OUTPUTS_DIR / "18_rorac_frontier.csv", index=False)
    outfiles.append(str((OUTPUTS_DIR / "18_rorac_frontier.csv").resolve()))

    # figure 1: RORAC frontier (single curve) — combined ratio vs RORAC + market
    # (supplementary figure, not in the paper; print-size for consistency)
    fig, ax = plt.subplots(figsize=(FIG_W_TEXT, R1_FIG9_HEIGHT_IN))
    ax.plot(fdf["combined_ratio"], fdf["rorac"], "-", color="#08519C", lw=1.5,
            label="Technical-premium frontier")
    ax.scatter([L_exp / P_opt + e], [rho], marker="o", s=30, color="#D7301F",
               zorder=5, label=f"RORAC target (h = {rho:.0%})")
    ax.axhline(TARGET_RORAC, color="grey", ls="--", lw=0.8)
    ax.axvline(1.0, color="grey", ls=":", lw=0.8, label="Combined ratio = 1 (break-even)")
    ax.set_xlabel("Combined ratio ((expected loss + expenses) / premium)")
    ax.set_ylabel("RORAC (underwriting profit / capital proxy)")
    ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "18_rorac_frontier.png")
    fig.savefig(FIGURES_DIR / "18_rorac_frontier.pdf"); plt.close(fig)
    outfiles.append(str((FIGURES_DIR / "18_rorac_frontier.png").resolve()))

    # figure 2: principle differentiation — loading % by risk decile
    # (supplementary figure, not in the paper; print-size for consistency)
    fig, ax = plt.subplots(figsize=(FIG_W_TEXT, R1_FIG9_HEIGHT_IN))
    cmap = {"expected_value": "#3182BD", "standard_deviation": "#41AB5D",
            "variance": "#807DBA", "exponential": "#FD8D3C",
            "cost_of_capital": "#D7301F"}
    plabel = {"expected_value": "Expected value", "standard_deviation":
              "Standard deviation", "variance": "Variance",
              "exponential": "Exponential", "cost_of_capital": "Cost of capital"}
    for pr in shapes:
        sub = ddf[ddf["principle"] == pr].sort_values("decile")
        ax.plot(sub["decile"], 100 * sub["loading_pct"], "o-",
                color=cmap.get(pr, "#888"), label=plabel.get(pr, pr),
                alpha=0.85, ms=3)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Decile of predicted pure premium (1 = lowest risk)")
    ax.set_ylabel("Premium loading (%)\n(office premium / expected loss − 1)")
    ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "18_differentiation.png")
    fig.savefig(FIGURES_DIR / "18_differentiation.pdf"); plt.close(fig)
    outfiles.append(str((FIGURES_DIR / "18_differentiation.png").resolve()))

    # figure 3: risk-decile adequacy (market vs risk-based optimal, RELATIVE)
    # FIGURE 9 (paper). R1: print size (single panel, text width), no title
    # (the caption carries it), deciles 1-10 on the x-axis, reference line at
    # 1 = risk-fair level, legend inside the empty upper-left region.
    fig, ax = plt.subplots(figsize=(FIG_W_TEXT, R1_FIG9_HEIGHT_IN))
    x = adq["decile"].values
    ax.axhline(1.0, color="0.35", ls=":", lw=1.0, zorder=1,
               label="Risk-fair level (relative loss ratio = 1)")
    if "market_lr_rel" in adq:
        ax.plot(x, adq["market_lr_rel"], "s--", color=R1_COLOR_MARKET, ms=4,
                lw=1.2, zorder=3, label="Actual (market) premium")
    ax.plot(x, adq["coc_lr_rel"], "o-", color=R1_COLOR_TARGET, ms=4, lw=1.2,
            zorder=3, label="RORAC-target premium")
    ax.set_xticks(range(1, 11))
    ax.set_xlim(0.5, 10.5)
    _yv = np.concatenate([adq["coc_lr_rel"].values,
                          adq["market_lr_rel"].values if "market_lr_rel" in adq
                          else np.array([])])
    _ylo = min(0.90, np.floor((_yv.min() - 0.02) * 20) / 20)
    _yhi = max(1.10, np.ceil((_yv.max() + 0.02) * 20) / 20) + 0.05
    ax.set_ylim(_ylo, _yhi)
    ax.yaxis.set_major_formatter(
        FormatStrFormatter("%.2f"))
    ax.set_xlabel("Decile of predicted pure premium (1 = lowest risk, "
                  "10 = highest risk)")
    ax.set_ylabel("Relative loss ratio\n(decile loss ratio / portfolio loss ratio)")
    # legend order: market, RORAC target, reference line
    _h, _l = ax.get_legend_handles_labels()
    _ord = [_l.index(s) for s in ("Actual (market) premium",
                                  "RORAC-target premium",
                                  "Risk-fair level (relative loss ratio = 1)")
            if s in _l]
    ax.legend([_h[i] for i in _ord], [_l[i] for i in _ord], loc="upper left")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "18_adequacy.png")
    fig.savefig(FIGURES_DIR / "18_adequacy.pdf"); plt.close(fig)
    outfiles.append(str((FIGURES_DIR / "18_adequacy.png").resolve()))
    outfiles.append(str((FIGURES_DIR / "18_adequacy.pdf").resolve()))

    # R1: one block that traces every number quoted in Section 3.7 (and the
    # Section 2.9 / Remark 2 constants) to this run. Values are identical to
    # the legacy keys above/below; nothing here feeds back into an estimate.
    P_nb_self = (L_exp + rho * scr_nb) / (1.0 - e)
    P_nb_hi_self = (L_exp + rho * scr_nb_hi) / (1.0 - e)
    section_3_7_trace = {
        "hurdle_rate_h": float(TARGET_RORAC),
        "hurdle_rate_h_note": ("config.TARGET_RORAC; the paper's hurdle rate h "
                               "(formerly written rho) in pi*(x) = (m(x) + "
                               "h*K(x))/(1-e)"),
        "expense_ratio_e": float(EXPENSE_RATIO),
        "solvency_alpha": float(SOLVENCY_ALPHA),
        "capital_proxy_definition": (
            "SCR = q_0.995(S) - E[S] of the independent-sum compound-Poisson "
            "aggregate S of the 2017-2018 test portfolio, q from a Cornish-"
            "Fisher expansion (mean, variance, skewness from compound "
            "cumulants kappa_j = sum_i lambda_i E[Y^j], empirical severity "
            "moments of the claim file); Euler allocation K_i = SCR*V_i/sum V_j"),
        "n_policies_test": int(len(m)),
        "expected_loss_E_S": float(agg_mean),
        "sd_S": float(agg_sd),
        "skewness_gamma": float(gamma_cf),
        "z_alpha": z_a,
        "z_cornish_fisher": float(z_a + skew_term),
        "cf_skewness_term_sigma_units": skew_term,
        "cf_skewness_term_share_of_capital_distance": skew_share,
        "scr_cf": float(scr),
        "scr_pct_of_expected_loss": float(scr / agg_mean),
        "scr_mc_crosscheck": float(scr_mc),
        "cf_over_mc": float(scr / scr_mc),
        "cf_above_mc_pct": float(scr / scr_mc - 1.0),
        "scr_mc_standard_error": scr_mc_se,
        "nb_dispersion": float(disp),
        "nb_dispersion_boot_upper_97_5": float(disp_hi),
        "nb_dispersion_boot_B": int(NB_DISP_BOOT),
        "scr_negbin": float(scr_nb),
        "scr_negbin_over_poisson": float(scr_nb / scr),
        "scr_negbin_param_unc": float(scr_nb_hi),
        "scr_negbin_param_unc_over_poisson": float(scr_nb_hi / scr),
        "capital_margin_h_times_scr": float(rho * scr),
        "capital_margin_pct_premium_poisson": float(rho * scr / P_opt),
        "capital_margin_pct_premium_negbin": float(rho * scr_nb / P_opt),
        "capital_margin_pct_premium_negbin_param_unc": float(rho * scr_nb_hi / P_opt),
        "capital_margin_pct_denominator": ("RORAC-target premium under the "
                                           "compound-Poisson SCR (legacy)"),
        "capital_margin_pct_premium_negbin_selfconsistent": float(rho * scr_nb / P_nb_self),
        "capital_margin_pct_premium_negbin_param_unc_selfconsistent": float(
            rho * scr_nb_hi / P_nb_hi_self),
        "rorac_target_premium_total": float(P_opt),
        "rorac_attained": float(rorac_opt),
        "expected_loss_ratio_target_premium": float(L_exp / P_opt),
        "realized_loss_total": float(L_real),
        "realized_loss_ratio_target_premium": float(L_real / P_opt),
        "market_premium_total_real": (market or {}).get("premium"),
        "market_premium_n_valid": (market or {}).get("n_valid"),
        "market_premium_basis": premium_basis,
        "premium_deflation_mean_multiplier": defl_mean,
        "sd_principle_progressivity_top_over_bottom": prin_summary.get(
            "standard_deviation", {}).get("progressivity_high_over_low"),
        "adequacy_deciles": ("10 equal-count groups of test policies sorted "
                             "by predicted pure premium m(x) of the best 14b "
                             "model (np.argsort + array_split)"),
        "adequacy_normalisation": ("relative LR = decile realized LR / "
                                   "portfolio realized LR, separately for each "
                                   "premium; market uses the n_valid policies "
                                   "with a positive recorded premium"),
        "rel_lr_market_by_decile": ([float(v) for v in adq["market_lr_rel"]]
                                    if "market_lr_rel" in adq else None),
        "rel_lr_rorac_target_by_decile": [float(v) for v in adq["coc_lr_rel"]],
        "top_over_bottom_rel_lr_market": risk_diff_mkt,
        "top_over_bottom_rel_lr_rorac_target": risk_diff_coc,
    }
    design_choices_r1 = {
        "figure9": {"file_stem": "figures/18_adequacy",
                    "width_in": float(FIG_W_TEXT),
                    "height_in": float(R1_FIG9_HEIGHT_IN),
                    "style": "config.apply_pub_style()",
                    "series": ["market_lr_rel", "coc_lr_rel"],
                    "reference_line": 1.0},
        "scr_mc_se": {"bootstrap_B": int(R1_SCR_MC_SE_BOOT_B),
                      "seed": int(GLOBAL_SEED + R1_SCR_MC_SE_SEED_OFFSET)},
    }

    manifest = {
        "script": SCRIPT_NAME,
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_policies": int(len(m)),
        "severity_moments": {"E_Y": ES1, "E_Y2_over_E_Y": ratio2,
                             "E_Y3_over_E_Y": ratio3},
        "scr_cf": scr, "scr_mc_crosscheck": scr_mc,
        "frequency_variance_sensitivity": freq_var_sens,
        "capital_margin_pct_scenarios": {
            "poisson": float(rho * scr / P_opt),
            "negbin": float(rho * scr_nb / P_opt),
            "negbin_param_unc": float(rho * scr_nb_hi / P_opt)},
        "scr_cf_over_mc": float(scr / scr_mc),
        "premium_basis": premium_basis,
        "scr_pct_expected": float(scr / agg_mean),
        "cf_skewness": gamma_cf,
        "expected_loss": L_exp, "realized_loss": L_real,
        "target_rorac": TARGET_RORAC, "coc_rate": COC_RATE,
        "expense_ratio": EXPENSE_RATIO,
        "optimal_technical_premium": {"rho": rho, "premium": P_opt,
                                      "loss_ratio_exp": float(L_exp / P_opt),
                                      "loss_ratio_real": float(L_real / P_opt),
                                      "rorac": float(rorac_opt),
                                      "capital_margin": float(rho * scr),
                                      "capital_margin_pct_premium": float(rho * scr / P_opt)},
        "principle_differentiation": prin_summary,
        "market": market,
        "risk_differentiation": {"market": risk_diff_mkt,
                                 "risk_based_optimal": risk_diff_coc},
        "adequacy": adq.to_dict(orient="records"),
        "note": ("SCR bileşik-Poisson ampirik severity momentiyle (ağır kuyruk "
                 "dahil); Euler tahsisi bağımsız-toplam varyansa dayanır (Bölüm "
                 "6: frekans-şiddet bağı ihmal edilebilir → tutarlı). Piyasa "
                 "mutlak LR'si nominal-prim/reel-hasar baz-uyumsuzluğu taşır; "
                 "risk-ayrımı için GÖRELİ (portföye normalize) LR kullanılır."),
        "hurdle_rate_h": float(TARGET_RORAC),
        "scr_mc_standard_error": scr_mc_se,
        "premium_deflation_mean_multiplier": defl_mean,
        "risk_score_source": risk_score_source,
        "level_adjustment_R2b": level_adjustment,
        "section_3_7_trace": section_3_7_trace,
        "design_choices_r1": design_choices_r1,
        "output_files": outfiles,
        "environment": env_info(),
    }
    with open(OUTPUTS_DIR / "18_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
    log.info("\n" + "=" * 78); log.info("SCRIPT 18 COMPLETED"); log.info("=" * 78)


if __name__ == "__main__":
    main()
