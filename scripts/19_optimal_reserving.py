"""
================================================================================
SCRIPT 19 — Risk-Based Reserving Strategies (Mack-Lognormal + CoC Margin + Newsvendor)
================================================================================
PURPOSE
    Turns the Chain-Ladder point estimate (Section 3) into a DECISION problem:
    at what level SHOULD the company set its reserve? Actually delivers the
    thesis/title claim of "Optimal Claims Reserving Strategies".

METHOD
    • Predictive distribution: England-Verrall (1999) over-dispersed Poisson (ODP)
      bootstrap Chain-Ladder. Pearson residuals are resampled to build a pseudo-
      triangle, the CL is refitted, the lower triangle is projected; process error
      (Gamma, variance=φ·mean) is added to obtain the FULL predictive distribution.
      Cross-validated against the Mack (1993) analytic s.e.
    • Reserving strategies:
        best_estimate : distribution mean (CL point)
        VaR_q         : confidence level q (IFRS 17 / Solvency II; q∈config)
        TVaR_q        : tail-conditional (penalty for extreme inadequacy)
        coc_margin    : technical provision = best estimate + Solvency II CoC risk
                        margin, RM = CoC·Σ_t SCR_t/(1+r)^t (run-off SCR pattern)
    • OPTIMAL reserve (asymmetric cost / newsvendor): balance between the cost of
      under-reserving (c_u: inadequacy/solvency penalty) and over-reserving (c_o:
      capital holding) → critical fractile R* = F⁻¹(c_u/(c_u+c_o)). The optimal
      confidence level and reserve are reported on a grid of penalty ratios c_u/c_o.

INPUT : outputs/03_development_triangle.csv ; outputs/03_mack_se.csv
OUTPUT: outputs/19_reserve_strategies.csv ; outputs/19_optimal_reserve.csv ;
        outputs/19_distribution_sensitivity.csv (R1) ;
        figures/19_*.{pdf,png} ; outputs/19_manifest.json

R1 REVISION (mathematics-4559487, September 2026)
    • Distributional sensitivity in the module: lognormal (primary), gamma, Weibull and
      inverse-Gaussian predictive distributions are matched to the SAME moments
      (mean=CL reserve, SD=Mack total s.e.); 75%/95%/99.5% VaR + TVaR, newsvendor R*
      on the penalty grid, ratios to lognormal and the c_u/c_o = F(TP)/(1−F(TP))
      implied by the CoC technical provision under each distribution → Tables 8-9 now
      come directly from the pipeline output (19_distribution_sensitivity.csv).
    • The risk-margin definition is documented in the manifest: r_f (discount), SCR_0
      definition, SCR_t run-off pattern, t range and every term; the ×BE ratios of
      Table 7 are recorded to 2 decimals.
    • Figures 10-11 redrawn at MDPI print size (apply_pub_style).
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
from matplotlib.ticker import FuncFormatter, FixedLocator, NullLocator
from scipy import stats, special, optimize, integrate

from config import (
    RESERVE_TRIANGLE_VARIANTS,
    OUTPUTS_DIR, FIGURES_DIR, LOGS_DIR, GLOBAL_SEED, FIG_DPI, CURRENCY,
    SOLVENCY_ALPHA, RESERVE_BOOTSTRAP_N, RESERVE_PROCESS_ERROR,
    RESERVE_CONF_LEVELS, RESERVE_DISCOUNT_RATE, RESERVE_PENALTY_GRID,
    RESERVE_COC_RATE,
    apply_pub_style, thousands_formatter, env_info, FIG_W_TEXT,
)

SCRIPT_NAME = "19_optimal_reserving"

# --- R1 revision constants ---------------------------------------------------
# Module-local design choices of the R1 revision (config.py is not edited).
# Moment-matched predictive distributions for the reserve (mean = CL reserve,
# SD = Mack total s.e.); "lognormal" is the primary model, the rest are the
# distributional-sensitivity alternatives of Table 9 / Section 3.8.
PRED_DISTS = ("lognormal", "gamma", "weibull", "invgauss")
PRED_DIST_LABELS = {"lognormal": "Lognormal (primary)", "gamma": "Gamma",
                    "weibull": "Weibull", "invgauss": "Inverse Gaussian"}
N_PRED_DRAWS = 200_000        # lognormal draws: Fig. 10 histogram + cost integral
WEIBULL_SHAPE_BRACKET = (0.05, 50.0)   # Brent bracket for the Weibull shape c
WEIBULL_SHAPE_XTOL = 1e-14             # Brent tolerance for c
TVAR_CHECK_RTOL = 1e-7        # closed-form vs numerical TVaR cross-check tolerance
FIG10_XMAX_Q = 0.999          # Fig. 10 x-range = [0, lognormal 99.9% quantile]
FIG10_N_BINS = 60             # histogram bins over the displayed range
# Timing of the run-off SCR pattern in the CoC risk margin (Eq. 5):
#   "as_submitted"       : O_t = sum_{k>=t} P_k with P_0 = 0 (O_0 = O_1), i.e.
#                          the SCR lags the expected payments by one year
#                          (reproduces the submitted RM = 372,265);
#   "solvency2_standard" : O_t = sum_{k>=t+1} P_k = expected outstanding at the
#                          END of year t (Delegated Reg. 2015/35 Art. 37 timing).
# Primary stays "as_submitted"; the other timing is ALWAYS computed and
# written to the manifest as a sensitivity (risk_margin.alternative_timing).
RISK_MARGIN_TIMING = "solvency2_standard"   # R1 final: standard Solvency II timing (O_t = outstanding at end of year t); "as_submitted" held SCR_0 for two years


def setup_logger(mode="w"):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(SCRIPT_NAME); lg.setLevel(logging.INFO); lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOGS_DIR / f"{SCRIPT_NAME}.log", mode=mode, encoding="utf-8")
    fh.setFormatter(fmt); sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    lg.addHandler(fh); lg.addHandler(sh); return lg


log = logging.getLogger(SCRIPT_NAME)


def set_style():
    """R1: MDPI print style (config.apply_pub_style) + serif mathtext."""
    apply_pub_style()
    plt.rcParams.update({"mathtext.fontset": "dejavuserif"})


def lognormal_from_moments(mean, sd):
    """Lognormal parameters from a positive mean/sd (moment matching).
    England-Verrall (1999): the CL predictive distribution can be approximated by a
    lognormal/Gamma with the Mack moments; on a mature triangle with recoveries the
    lognormal is unaffected by the ODP's wrinkles (positive-increment assumption)."""
    mean = max(mean, 1e-6)
    cv2 = (sd / mean) ** 2
    sig2 = np.log(1.0 + cv2)
    mu = np.log(mean) - 0.5 * sig2
    return mu, float(np.sqrt(sig2))


def ln_quantile(mean, sd, q):
    mu, sig = lognormal_from_moments(mean, sd)
    return float(np.exp(mu + sig * stats.norm.ppf(q)))


def ln_tvar(mean, sd, q):
    """Lognormal TVaR: E[X|X>VaR_q]=exp(μ+σ²/2)·Φ(σ−z_q)/(1−q)."""
    mu, sig = lognormal_from_moments(mean, sd)
    z = stats.norm.ppf(q)
    return float(np.exp(mu + sig ** 2 / 2) * stats.norm.cdf(sig - z) / (1 - q))


TRIANGLE_FILES = {"paid": "03_development_triangle.csv",
                  "incurred": "22_incurred_triangle.csv"}   # R2: incurred = paid + open case reserves


def load_triangle(variant="paid"):
    """Cumulative development triangle → observed-cell mask + cumulative array.
    variant: 'paid' (03, paid) | 'incurred' (22, paid + year-end open case reserves; R2)."""
    df = pd.read_csv(OUTPUTS_DIR / TRIANGLE_FILES[variant], index_col=0)
    df.columns = [int(c) for c in df.columns]
    C = df.values.astype(float)             # cumulative (AY × dev)
    obs = ~np.isnan(C)                       # observed upper triangle
    return df.index.values.astype(int), np.array(df.columns), C, obs


def cl_factors(C, obs):
    """Volume-weighted age-to-age factors f_j (from observed pairs)."""
    n_ay, n_dev = C.shape
    f = np.ones(n_dev - 1)
    for j in range(n_dev - 1):
        num = den = 0.0
        for i in range(n_ay):
            if obs[i, j] and obs[i, j + 1] and C[i, j] > 0:
                num += C[i, j + 1]; den += C[i, j]
        f[j] = num / den if den > 0 else 1.0
    return f


def project_reserve(C, obs, f):
    """Project the lower triangle by CL from the last observed diagonal; reserve =
    Σ(ultimate − last observed cumulative)."""
    n_ay, n_dev = C.shape
    reserve = 0.0
    ults = np.full(n_ay, np.nan)
    for i in range(n_ay):
        obs_j = np.where(obs[i])[0]
        if len(obs_j) == 0:
            continue
        last = obs_j.max()
        c = C[i, last]
        for j in range(last, n_dev - 1):
            c = c * f[j]
        ults[i] = c
        reserve += c - C[i, last]
    return reserve, ults


def fitted_incrementals(C, obs, f):
    """England-Verrall: expected cumulative Ĉ by backward recursion from the last
    diagonal, then expected incrementals m_ij (observed cells only)."""
    n_ay, n_dev = C.shape
    Chat = np.full_like(C, np.nan)
    for i in range(n_ay):
        obs_j = np.where(obs[i])[0]
        if len(obs_j) == 0:
            continue
        last = obs_j.max()
        Chat[i, last] = C[i, last]
        for j in range(last - 1, -1, -1):     # backwards: Ĉ_{j} = Ĉ_{j+1}/f_j
            Chat[i, j] = Chat[i, j + 1] / f[j] if f[j] > 0 else C[i, j]
    m = np.full_like(C, np.nan)
    for i in range(n_ay):
        for j in range(n_dev):
            if not obs[i, j]:
                continue
            m[i, j] = Chat[i, j] if j == 0 else Chat[i, j] - Chat[i, j - 1]
    return m


def bootstrap_reserve(C, obs, B, process_error, seed=GLOBAL_SEED):
    """ODP bootstrap CL predictive distribution (England-Verrall 1999)."""
    rng = np.random.default_rng(seed)
    n_ay, n_dev = C.shape
    f0 = cl_factors(C, obs)
    m = fitted_incrementals(C, obs, f0)
    # observed incrementals
    I = np.full_like(C, np.nan)
    for i in range(n_ay):
        for j in range(n_dev):
            if obs[i, j]:
                I[i, j] = C[i, j] if j == 0 else (C[i, j] - C[i, j - 1]
                                                  if obs[i, j - 1] else np.nan)
    cells = [(i, j) for i in range(n_ay) for j in range(n_dev)
             if obs[i, j] and np.isfinite(I[i, j]) and np.isfinite(m[i, j])]
    denom = np.array([np.sqrt(max(abs(m[i, j]), 1e-6)) for i, j in cells])
    resid = np.array([(I[i, j] - m[i, j]) for i, j in cells]) / denom
    n_obs = len(cells); n_par = (n_dev - 1) + n_ay      # number of CL parameters
    dof = max(n_obs - n_par, 1)
    phi = float(np.sum(resid ** 2) / dof)               # scale (dispersion)
    resid_adj = resid * np.sqrt(n_obs / dof)            # dof correction
    resid_adj = resid_adj - resid_adj.mean()

    reserves = np.empty(B)
    for b in range(B):
        # 1) resample residuals → pseudo-incrementals → pseudo-cumulatives
        rs = rng.choice(resid_adj, size=n_obs, replace=True)
        Istar = I.copy()
        for (i, j), r in zip(cells, rs):
            Istar[i, j] = m[i, j] + r * np.sqrt(max(abs(m[i, j]), 1e-6))
        Cstar = np.full_like(C, np.nan)
        for i in range(n_ay):
            obs_j = np.where(obs[i])[0]
            acc = 0.0
            for j in range(obs_j.max() + 1):
                if obs[i, j]:
                    acc += Istar[i, j] if np.isfinite(Istar[i, j]) else 0.0
                    Cstar[i, j] = acc
        # 2) refit CL + project (parameter/estimation error)
        fstar = cl_factors(Cstar, obs)
        _, ults = project_reserve(Cstar, obs, fstar)
        # 3) process error: future incrementals ODP (Gamma, var=φ·mean)
        r_b = 0.0
        for i in range(n_ay):
            obs_j = np.where(obs[i])[0]
            if len(obs_j) == 0:
                continue
            last = obs_j.max()
            cprev = Cstar[i, last] if np.isfinite(Cstar[i, last]) else C[i, last]
            for j in range(last, n_dev - 1):
                inc = cprev * (fstar[j] - 1.0)
                if process_error and inc > 0 and phi > 0:
                    shape = inc / phi
                    inc = rng.gamma(max(shape, 1e-6), phi)
                r_b += inc
                cprev = cprev + inc
        reserves[b] = r_b
    return reserves, phi, f0


def runoff_payments(C, obs, f):
    """Expected CL payments by future calendar year k = 1..n_dev−1 (P_0 ≡ 0).
    Negative expected incrementals (mature-year recoveries) are floored at 0
    in the pattern; their count and sum are reported."""
    n_ay, n_dev = C.shape
    P = np.zeros(n_dev)
    n_neg, neg_sum = 0, 0.0
    for i in range(n_ay):
        obs_j = np.where(obs[i])[0]
        if len(obs_j) == 0:
            continue
        last = obs_j.max()
        c = C[i, last]
        for j in range(last, n_dev - 1):
            inc = c * (f[j] - 1.0)
            if inc < 0:
                n_neg += 1; neg_sum += inc
            P[j - last + 1] += max(inc, 0.0)
            c = c + inc
    return P, n_neg, float(neg_sum)


def coc_risk_margin(best_estimate, scr0, ults, C, obs, f, coc, disc,
                    timing=RISK_MARGIN_TIMING):
    """Solvency II run-off cost-of-capital risk margin.
    RM = CoC·Σ_{t=0}^{T−1} SCR_t/(1+r_f)^{t+1}, SCR_t = SCR_0·O_t/O_0,
    T = n_dev; O_t = expected unpaid-claims path (depending on timing, see
    RISK_MARGIN_TIMING). Returns: (RM, O, detail-dict)."""
    n_dev = C.shape[1]
    P, n_neg, neg_sum = runoff_payments(C, obs, f)
    if timing == "as_submitted":
        O = np.array([P[t:].sum() for t in range(n_dev)])        # O_0 = O_1
    elif timing == "solvency2_standard":
        O = np.array([P[t + 1:].sum() for t in range(n_dev)])    # end of year t
    else:
        raise ValueError(f"unknown risk-margin timing: {timing}")
    O0 = O[0] if O[0] > 0 else best_estimate
    rm = 0.0
    scr_t, dfac, terms = [], [], []
    for t in range(n_dev):
        s_t = scr0 * (O[t] / O0 if O0 > 0 else 0.0)
        d_t = 1.0 / (1.0 + disc) ** (t + 1)
        rm += coc * s_t * d_t
        scr_t.append(float(s_t)); dfac.append(float(d_t))
        terms.append(float(coc * s_t * d_t))
    detail = {
        "timing": timing,
        "expected_payments_by_future_year_P_k": [float(x) for x in P],
        "O_t": [float(x) for x in O],
        "O_t_over_O0": [float(x / O0) for x in O],
        "sum_O_t_over_O0": float(np.sum(O / O0)),
        "SCR_t": scr_t, "discount_factor_t": dfac, "rm_term_t": terms,
        "t_range": [0, n_dev - 1], "T": int(n_dev),
        "n_nonzero_terms": int(np.sum(O > 0)),
        "n_negative_expected_increments_floored": int(n_neg),
        "sum_negative_expected_increments_floored": neg_sum,
    }
    return float(rm), O, detail


# ---------------------------------------------------------------------------
# R1: moment-matched predictive distributions (sensitivity of Tables 8-9)
# ---------------------------------------------------------------------------
def moment_matched(name, mean, sd):
    """(mean, sd) → frozen scipy distribution + parameters (moment matching)."""
    cv2 = (sd / mean) ** 2
    if name == "lognormal":
        mu, sig = lognormal_from_moments(mean, sd)
        return stats.lognorm(sig, scale=np.exp(mu)), {"mu": float(mu), "sigma": float(sig)}
    if name == "gamma":
        k = (mean / sd) ** 2; th = sd ** 2 / mean      # = former k_g, th_g
        return stats.gamma(k, scale=th), {"shape_k": k, "scale_theta": th}
    if name == "weibull":
        # CV² matching: Γ(1+2/c)/Γ(1+1/c)² = 1 + CV² (Brent on the log scale)
        g = lambda c: (special.gammaln(1 + 2 / c) - 2 * special.gammaln(1 + 1 / c)
                       - np.log1p(cv2))
        c = optimize.brentq(g, *WEIBULL_SHAPE_BRACKET, xtol=WEIBULL_SHAPE_XTOL,
                            rtol=4 * np.finfo(float).eps, maxiter=500)
        lam = mean / np.exp(special.gammaln(1 + 1 / c))
        return stats.weibull_min(c, scale=lam), {"shape_c": float(c),
                                                 "scale_lambda": float(lam)}
    if name == "invgauss":
        lam = mean ** 3 / sd ** 2                       # IG(μ=mean, λ=mean³/sd²)
        return (stats.invgauss(mean / lam, scale=lam),
                {"mean_mu": mean, "shape_lambda": float(lam)})
    raise ValueError(name)


def dist_quantile(name, dist, mean, sd, q):
    """Primary closed form for the lognormal (ln_quantile) — identical to Table 7."""
    return ln_quantile(mean, sd, q) if name == "lognormal" else float(dist.ppf(q))


def dist_cdf(name, dist, mean, sd, x):
    if name == "lognormal":
        mu, sig = lognormal_from_moments(mean, sd)
        return float(stats.norm.cdf((np.log(x) - mu) / sig))
    return float(dist.cdf(x))


def dist_tvar(name, dist, par, mean, sd, q):
    """Closed-form TVaR_q = E[X | X > VaR_q] (continuous distribution)."""
    if name == "lognormal":
        return ln_tvar(mean, sd, q)
    v = float(dist.ppf(q))
    if name == "gamma":          # E[X;X>v] = m·S_{k+1,θ}(v)
        return float(mean * stats.gamma.sf(v, par["shape_k"] + 1,
                                           scale=par["scale_theta"]) / (1 - q))
    if name == "weibull":        # E[X;X>v] = m·Q(1+1/c, (v/λ)^c)
        c, lam = par["shape_c"], par["scale_lambda"]
        return float(mean * special.gammaincc(1 + 1 / c, (v / lam) ** c) / (1 - q))
    if name == "invgauss":       # E[X;X≤v] = μ[Φ(a(v/μ−1)) − e^{2λ/μ}Φ(−a(v/μ+1))], a=√(λ/v)
        mu, la = mean, par["shape_lambda"]
        a = np.sqrt(la / v)
        e_below = mu * (stats.norm.cdf(a * (v / mu - 1))
                        - np.exp(2 * la / mu + stats.norm.logcdf(-a * (v / mu + 1))))
        return float((mu - e_below) / (1 - q))
    raise ValueError(name)


def tvar_numeric(dist, q, scale):
    """Independent numerical check: TVaR_q = VaR_q + (1−q)⁻¹ ∫_{VaR_q}^∞ S(x) dx."""
    v = float(dist.ppf(q)); hi = float(dist.isf(1e-14))
    integ = integrate.quad(lambda u: dist.sf(u * scale), v / scale, hi / scale,
                           epsabs=0.0, epsrel=1e-11, limit=1000)[0] * scale
    return v + integ / (1 - q)


def main(variant="paid"):
    global log
    SFX = "" if variant == "paid" else f"_{variant}"
    log = setup_logger(mode="w" if variant == "paid" else "a"); set_style()
    log.info("=" * 78)
    log.info(f"SCRIPT {SCRIPT_NAME} — Optimal Reserving Strategy [triangle: {variant}]")
    log.info("=" * 78)
    outfiles = []

    ay, dev, C, obs = load_triangle(variant)
    log.info(f"\n[1/5] Triangle ({variant}): {C.shape[0]} accident years × {C.shape[1]} development years")
    f0 = cl_factors(C, obs)
    be_point, ults0 = project_reserve(C, obs, f0)
    log.info(f"  CL point reserve (ultimate − last observed cumulative) = {be_point:,.0f} {CURRENCY}")
    # R2: the expected PAYMENT run-off pattern (O_t of the risk margin) always comes
    # from the paid triangle; for the incurred variant the incurred CL gives the
    # level (outstanding = ultimate − paid to date) and the Mack s.e.
    ayP, devP, CP, obsP = load_triangle("paid")
    fP = cl_factors(CP, obsP)
    _, ultsP = project_reserve(CP, obsP, fP)

    # PRIMARY predictive distribution: lognormal with the Mack (1993) total prediction error.
    # This triangle is almost fully developed and contains recoveries (negative
    # incrementals) in mature years; the ODP bootstrap inflates φ because of its
    # positive-increment assumption. The Mack model works on the cumulative scale and is
    # unaffected by this wrinkle → primary distribution is the Mack-moment lognormal; ODP bootstrap as SENSITIVITY.
    log.info("\n[2/5] Predictive distribution — Mack (1993) moment lognormal (primary)")
    mack_total_se = None
    if variant == "incurred":
        m22 = json.load(open(OUTPUTS_DIR / "22_manifest.json", encoding="utf-8"))
        be_point = float(m22["incurred_CL"]["outstanding"])      # ultimate − paid to date
        mack_total_se = float(m22["incurred_CL"]["mack_total_se"])
        log.info(f"  Incurred CL (22): outstanding={be_point:,.0f}, Mack s.e.={mack_total_se:,.0f}; "
                 f"open case reserves={m22['incurred_CL']['case_reserve_at_valuation']:,.0f}")
    else:
        try:
            mm = json.load(open(OUTPUTS_DIR / "03_manifest.json", encoding="utf-8"))
            mack_total_se = float(mm["mack"]["total_se"])
            be_point = float(mm["chain_ladder"]["ibnr"])       # reserve consistent with 03
        except Exception:
            mack = pd.read_csv(OUTPUTS_DIR / "03_mack_se.csv")
            mack_total_se = float(np.sqrt(np.sum(mack["se"].values ** 2)))
    be = be_point
    sd = mack_total_se
    cv = sd / be if be > 0 else np.nan
    if not (be > 0 and sd > 0):
        # R2 robustness guard: a non-positive best estimate (possible on synthetic or
        # degenerate triangles, e.g. incurred CL below paid to date) has no lognormal
        # predictive distribution; the variant is skipped with a minimal manifest.
        log.warning(f"  Reserve basis '{variant}': BE={be:,.0f}, s.e.={sd:,.0f} → predictive distribution "
                    "undefined; this variant SKIPPED (minimal manifest written)")
        with open(OUTPUTS_DIR / f"19{SFX}_manifest.json", "w", encoding="utf-8") as f:
            json.dump({"script": SCRIPT_NAME, "triangle_variant": variant, "skipped": True,
                       "reason": "non-positive best estimate or standard error",
                       "cl_point_reserve": float(be_point), "mack_total_se": float(sd),
                       "run_timestamp": datetime.now().isoformat(timespec="seconds")}, f, indent=2)
        log.info("\n" + "=" * 78); log.info("SCRIPT 19 COMPLETED"); log.info("=" * 78)
        return
    log.info(f"  Reserve={be:,.0f}, Mack total s.e.={sd:,.0f} (CV={cv:.1%}) → "
             f"lognormal predictive distribution")

    # SENSITIVITY: England-Verrall ODP bootstrap (mature-year recovery caveat)
    log.info(f"\n[2b/5] Sensitivity — ODP bootstrap (B={RESERVE_BOOTSTRAP_N})")
    try:
        reserves_bs, phi, _ = bootstrap_reserve(C, obs, RESERVE_BOOTSTRAP_N,
                                                RESERVE_PROCESS_ERROR)
        odp_ok = True
    except Exception as _e:   # R2: incurred triangle has negative increments
        log.warning(f"  ODP bootstrap could not be computed on this triangle ({_e}); NaN reported")
        reserves_bs = np.full(RESERVE_BOOTSTRAP_N, np.nan); phi = np.nan; odp_ok = False
    bs_mean = float(np.mean(reserves_bs)); bs_se = float(np.std(reserves_bs))
    log.info(f"  ODP bootstrap mean={bs_mean:,.0f}, s.e.={bs_se:,.0f}, φ={phi:,.0f} "
             f"(mature-year recoveries inflate φ → upper-bound sensitivity only)")

    log.info("\n[3/5] Reserving strategies (VaR / TVaR / CoC risk margin)")
    strat = [{"strategy": "best_estimate", "level": None, "reserve": be,
              "margin_over_be": 0.0}]
    for q in RESERVE_CONF_LEVELS:
        var_q = ln_quantile(be, sd, q)
        tv = ln_tvar(be, sd, q)
        strat.append({"strategy": f"VaR_{q}", "level": q, "reserve": var_q,
                      "margin_over_be": var_q - be})
        strat.append({"strategy": f"TVaR_{q}", "level": q, "reserve": tv,
                      "margin_over_be": tv - be})
    # SCR_0 = VaR_0.995(R) − BE (Mack-lognormal; ULTIMATE-view proxy of the
    # one-year reserve-risk SCR)
    scr0 = ln_quantile(be, sd, SOLVENCY_ALPHA) - be
    rm, O, rm_detail = coc_risk_margin(be, scr0, ultsP, CP, obsP, fP,
                                       RESERVE_COC_RATE, RESERVE_DISCOUNT_RATE,
                                       timing=RISK_MARGIN_TIMING)
    tp_coc = be + rm
    strat.append({"strategy": "coc_margin", "level": RESERVE_COC_RATE,
                  "reserve": tp_coc, "margin_over_be": rm})
    # R1: × BE ratio (last column of Table 7; 2 decimals)
    for s in strat:
        s["ratio_to_be"] = float(s["reserve"] / be)
        s["ratio_to_be_2dp"] = round(s["reserve"] / be, 2)
    for s in strat:
        log.info(f"  {s['strategy']:16s} reserve={s['reserve']:,.0f} "
                 f"(margin +{s['margin_over_be']:,.0f}; ×BE={s['ratio_to_be']:.4f}"
                 f" → {s['ratio_to_be_2dp']:.2f})")
    log.info(f"  Solvency II CoC risk margin = {rm:,.0f} → technical provision = {tp_coc:,.0f}")
    log.info(f"  RM definition: r_f={RESERVE_DISCOUNT_RATE}, T={rm_detail['T']} "
             f"(t=0..{rm_detail['T']-1}), timing={RISK_MARGIN_TIMING}, "
             f"ΣO_t/O_0={rm_detail['sum_O_t_over_O0']:.4f}, "
             f"O_t/O_0={np.round(rm_detail['O_t_over_O0'], 4).tolist()}")
    alt_timing = ("solvency2_standard" if RISK_MARGIN_TIMING == "as_submitted"
                  else "as_submitted")
    rm_alt, _, rm_alt_detail = coc_risk_margin(be, scr0, ultsP, CP, obsP, fP,
                                               RESERVE_COC_RATE,
                                               RESERVE_DISCOUNT_RATE,
                                               timing=alt_timing)
    tp_alt = be + rm_alt
    log.info(f"  [sensitivity] RM timing '{alt_timing}': RM={rm_alt:,.0f}, "
             f"TP={tp_alt:,.0f} (×BE {tp_alt/be:.2f}), "
             f"ΣO_t/O_0={rm_alt_detail['sum_O_t_over_O0']:.4f}")

    sdf = pd.DataFrame(strat)
    sdf.to_csv(OUTPUTS_DIR / f"19{SFX}_reserve_strategies.csv", index=False)
    outfiles.append(str((OUTPUTS_DIR / f"19{SFX}_reserve_strategies.csv").resolve()))

    # ---------------------------------------------------------------------
    # R1: four moment-matched predictive distributions (Tables 8-9 sensitivity)
    # ---------------------------------------------------------------------
    log.info("\n[3b/5] Predictive-distribution sensitivity: lognormal / gamma / Weibull / "
             "inverse-Gaussian (same mean and SD) + ODP")
    D, PAR, moment_check, tvar_check = {}, {}, {}, {}
    for nm in PRED_DISTS:
        D[nm], PAR[nm] = moment_matched(nm, be, sd)
        m_, s_ = float(D[nm].mean()), float(D[nm].std())
        moment_check[nm] = {"mean": m_, "sd": s_,
                            "rel_err_mean": abs(m_ / be - 1),
                            "rel_err_sd": abs(s_ / sd - 1)}
        assert moment_check[nm]["rel_err_mean"] < 1e-9 and \
            moment_check[nm]["rel_err_sd"] < 1e-9, f"moment match failed: {nm}"
        log.info(f"  {nm:9s} params={ {k: round(v, 6) for k, v in PAR[nm].items()} } "
                 f"mean={m_:,.2f} sd={s_:,.2f}")
    k_g, th_g = PAR["gamma"]["shape_k"], PAR["gamma"]["scale_theta"]

    var_tab = {q: {nm: dist_quantile(nm, D[nm], be, sd, q) for nm in PRED_DISTS}
               for q in RESERVE_CONF_LEVELS}
    tvar_tab = {}
    for q in RESERVE_CONF_LEVELS:
        tvar_tab[q] = {}
        for nm in PRED_DISTS:
            tv_cf = dist_tvar(nm, D[nm], PAR[nm], be, sd, q)
            tv_nu = tvar_numeric(D[nm], q, be)
            err = abs(tv_cf / tv_nu - 1)
            tvar_check[f"{nm}_{q}"] = err
            assert err < TVAR_CHECK_RTOL, f"TVaR check failed {nm} q={q}: {err:.2e}"
            tvar_tab[q][nm] = tv_cf
    bs_var = {q: float(np.quantile(reserves_bs, q)) for q in RESERVE_CONF_LEVELS}
    bs_tvar = {q: float(np.mean(reserves_bs[reserves_bs > bs_var[q]]))
               for q in RESERVE_CONF_LEVELS}
    log.info(f"  TVaR closed form vs numerical integral: max rel. diff = "
             f"{max(tvar_check.values()):.2e}")

    dist_cmp = []
    for q in RESERVE_CONF_LEVELS:
        ln_q, ga_q = var_tab[q]["lognormal"], var_tab[q]["gamma"]
        wb_q, ig_q = var_tab[q]["weibull"], var_tab[q]["invgauss"]
        dist_cmp.append({"q": q, "lognormal": ln_q, "gamma": ga_q,
                         "odp_bootstrap": bs_var[q],
                         "gamma_over_ln": float(ga_q / ln_q),
                         "weibull": wb_q, "invgauss": ig_q,
                         "weibull_over_ln": float(wb_q / ln_q),
                         "invgauss_over_ln": float(ig_q / ln_q),
                         "tvar": {nm: tvar_tab[q][nm] for nm in PRED_DISTS}
                                 | {"odp_bootstrap": bs_tvar[q]}})
        log.info(f"  q={q}: VaR LN={ln_q:,.0f} | G={ga_q:,.0f} ({ga_q/ln_q:.3f}) | "
                 f"W={wb_q:,.0f} ({wb_q/ln_q:.3f}) | IG={ig_q:,.0f} ({ig_q/ln_q:.3f}) | "
                 f"ODP={bs_var[q]:,.0f}")
        log.info(f"         TVaR LN={tvar_tab[q]['lognormal']:,.0f} | "
                 f"G={tvar_tab[q]['gamma']:,.0f} | W={tvar_tab[q]['weibull']:,.0f} | "
                 f"IG={tvar_tab[q]['invgauss']:,.0f} | ODP={bs_tvar[q]:,.0f}")

    # c_u/c_o implied by the CoC technical provision under each distribution
    tp_implied = {}
    for nm in PRED_DISTS:
        p = dist_cdf(nm, D[nm], be, sd, tp_coc)
        p_alt = dist_cdf(nm, D[nm], be, sd, tp_alt)
        tp_implied[nm] = {"F_TP": p, "implied_cu_co": p / (1 - p),
                          "F_TP_alt_timing": p_alt,
                          "implied_cu_co_alt_timing": p_alt / (1 - p_alt)}
        log.info(f"  TP={tp_coc:,.0f}: {nm:9s} F(TP)={p:.4f} → c_u/c_o={p/(1-p):.3f}"
                 f"   [alt. timing TP={tp_alt:,.0f}: F={p_alt:.4f} → "
                 f"{p_alt/(1-p_alt):.3f}]")
    log.info("  → The lognormal/gamma difference is small at central quantiles; at the extreme "
             "quantile the lognormal is heavier-tailed (conservative) — primary choice.")

    log.info("\n[4/5] Optimal reserve — asymmetric cost (critical fractile)")
    # Sample from the lognormal predictive distribution (cost integral + histogram)
    rng = np.random.default_rng(GLOBAL_SEED)
    mu_ln, sig_ln = lognormal_from_moments(be, sd)
    reserves = np.exp(rng.normal(mu_ln, sig_ln, N_PRED_DRAWS))
    opt = []
    for ratio in RESERVE_PENALTY_GRID:
        crit = ratio / (1.0 + ratio)             # c_u/(c_u+c_o)
        Rstar = ln_quantile(be, sd, crit)
        under = float(np.mean(np.maximum(reserves - Rstar, 0.0)))   # E[(U−R)+]
        over = float(np.mean(np.maximum(Rstar - reserves, 0.0)))    # E[(R−U)+]
        exp_cost = ratio * under + over
        Rstar_g = float(stats.gamma.ppf(crit, k_g, scale=th_g))
        Rstar_w = dist_quantile("weibull", D["weibull"], be, sd, crit)
        Rstar_ig = dist_quantile("invgauss", D["invgauss"], be, sd, crit)
        allR = [Rstar, Rstar_g, Rstar_w, Rstar_ig]
        opt.append({"penalty_ratio_cu_co": ratio, "critical_fractile": crit,
                    "optimal_reserve": Rstar, "implied_confidence": crit,
                    "optimal_reserve_gamma": Rstar_g,
                    "gamma_over_ln": float(Rstar_g / Rstar),
                    "margin_over_be": Rstar - be, "exp_cost_per_unit_co": exp_cost,
                    # --- R1 additions ---
                    "optimal_reserve_weibull": Rstar_w,
                    "weibull_over_ln": float(Rstar_w / Rstar),
                    "optimal_reserve_invgauss": Rstar_ig,
                    "invgauss_over_ln": float(Rstar_ig / Rstar),
                    "pct_diff_gamma_vs_ln": float(100 * (Rstar_g / Rstar - 1)),
                    "pct_diff_weibull_vs_ln": float(100 * (Rstar_w / Rstar - 1)),
                    "pct_diff_invgauss_vs_ln": float(100 * (Rstar_ig / Rstar - 1)),
                    "spread_max_over_min": float(max(allR) / min(allR)),
                    "optimal_reserve_odp_bootstrap": float(np.quantile(reserves_bs,
                                                                       crit))})
        log.info(f"  c_u/c_o={ratio:>4.0f} → critical fractile={crit:.3f}, "
                 f"R*={Rstar:,.0f} (gamma {Rstar_g:,.0f}; ratio {Rstar_g/Rstar:.2f}) "
                 f"(margin {Rstar-be:+,.0f}, {100*(Rstar/be-1):+.1f}%)")
        log.info(f"        Weibull {Rstar_w:,.0f} ({100*(Rstar_w/Rstar-1):+.2f}%), "
                 f"IG {Rstar_ig:,.0f} ({100*(Rstar_ig/Rstar-1):+.2f}%), "
                 f"gamma ({100*(Rstar_g/Rstar-1):+.2f}%); max/min={max(allR)/min(allR):.4f}")
    odf = pd.DataFrame(opt)
    odf.to_csv(OUTPUTS_DIR / f"19{SFX}_optimal_reserve.csv", index=False)
    outfiles.append(str((OUTPUTS_DIR / f"19{SFX}_optimal_reserve.csv").resolve()))

    # min/max of the newsvendor % differences over the grid (text: "gamma +2..+9%")
    ns_summary = {}
    for nm in ("gamma", "weibull", "invgauss"):
        col = odf[f"pct_diff_{nm}_vs_ln"]
        ns_summary[nm] = {
            "pct_vs_ln_min": float(col.min()),
            "pct_vs_ln_min_at_cu_co": float(odf.loc[col.idxmin(), "penalty_ratio_cu_co"]),
            "pct_vs_ln_max": float(col.max()),
            "pct_vs_ln_max_at_cu_co": float(odf.loc[col.idxmax(), "penalty_ratio_cu_co"]),
            "pct_vs_ln_range_rounded": [int(np.round(col.min())), int(np.round(col.max()))],
        }
        log.info(f"  newsvendor {nm:8s} vs LN on grid: {col.min():+.2f}% .. {col.max():+.2f}% "
                 f"(rounded {ns_summary[nm]['pct_vs_ln_range_rounded']})")
    i_sp = int(odf["spread_max_over_min"].idxmax())
    ns_summary["max_abs_shift_vs_ln_pct"] = float(max(
        odf[[f"pct_diff_{nm}_vs_ln" for nm in ("gamma", "weibull", "invgauss")]]
        .abs().max()))
    ns_summary["max_spread_across_four_pct"] = float(
        100 * (odf.loc[i_sp, "spread_max_over_min"] - 1))
    ns_summary["max_spread_at_cu_co"] = float(odf.loc[i_sp, "penalty_ratio_cu_co"])
    q_top = max(RESERVE_CONF_LEVELS)
    alts_top = {nm: var_tab[q_top][nm] for nm in ("gamma", "weibull", "invgauss")}
    nm_low = min(alts_top, key=alts_top.get)
    ns_summary["q_top"] = q_top
    ns_summary["ln_is_max_at_q_top"] = bool(var_tab[q_top]["lognormal"] >= max(alts_top.values()))
    ns_summary["ln_over_lowest_alt_at_q_top"] = float(var_tab[q_top]["lognormal"] / alts_top[nm_low])
    ns_summary["lowest_alt_at_q_top"] = nm_low
    log.info(f"  max |shift| vs LN = {ns_summary['max_abs_shift_vs_ln_pct']:.2f}%; "
             f"max spread across four = {ns_summary['max_spread_across_four_pct']:.2f}% "
             f"(c_u/c_o={ns_summary['max_spread_at_cu_co']:.0f}); "
             f"q{q_top}: LN/{nm_low} = {ns_summary['ln_over_lowest_alt_at_q_top']:.4f}")

    # 19_distribution_sensitivity.csv (Table 9 + Table 8 extension + TP)
    rows = []
    def _row(measure, level, crit, vals, odp):
        r = {"measure": measure, "level": level, "critical_fractile": crit}
        r.update({nm: vals[nm] for nm in PRED_DISTS})
        r["odp_bootstrap"] = odp
        for nm in ("gamma", "weibull", "invgauss"):
            r[f"{nm}_over_ln"] = vals[nm] / vals["lognormal"]
        rows.append(r)
    for q in RESERVE_CONF_LEVELS:
        _row("VaR", q, q, var_tab[q], bs_var[q])
    for q in RESERVE_CONF_LEVELS:
        _row("TVaR", q, q, tvar_tab[q], bs_tvar[q])
    for o in opt:
        _row("newsvendor_Rstar", o["penalty_ratio_cu_co"], o["critical_fractile"],
             {"lognormal": o["optimal_reserve"], "gamma": o["optimal_reserve_gamma"],
              "weibull": o["optimal_reserve_weibull"],
              "invgauss": o["optimal_reserve_invgauss"]},
             o["optimal_reserve_odp_bootstrap"])
    _row("CoC_TP_cdf", tp_coc, np.nan,
         {nm: tp_implied[nm]["F_TP"] for nm in PRED_DISTS},
         float(np.mean(reserves_bs <= tp_coc)))
    _row("CoC_TP_implied_cu_co", tp_coc, np.nan,
         {nm: tp_implied[nm]["implied_cu_co"] for nm in PRED_DISTS}, np.nan)
    dsens = pd.DataFrame(rows)
    dsens.to_csv(OUTPUTS_DIR / f"19{SFX}_distribution_sensitivity.csv", index=False)
    outfiles.append(str((OUTPUTS_DIR / f"19{SFX}_distribution_sensitivity.csv").resolve()))

    log.info("\n[5/5] Figures + saving")
    fmt_int = thousands_formatter(0)
    # ---- Figure 10: reserve predictive distribution + strategies -----------
    x_max = ln_quantile(be, sd, FIG10_XMAX_Q)
    n_beyond = int(np.sum(reserves > x_max))
    fig, ax = plt.subplots(figsize=(FIG_W_TEXT, 3.4))
    bins = np.linspace(0.0, x_max / 1e6, FIG10_N_BINS + 1)
    ax.hist(reserves / 1e6, bins=bins, color="#B8C4D6", edgecolor="white",
            linewidth=0.4, zorder=2)
    marks = [("Best estimate (BE)", be, "#000000", "-"),
             ("VaR 75%", var_tab[0.75]["lognormal"], "#0072B2", "--"),
             ("CoC technical provision", tp_coc, "#009E73", "-."),
             ("VaR 95%", var_tab[0.95]["lognormal"], "#E69F00", (0, (5, 1.5, 1, 1.5, 1, 1.5))),
             ("VaR 99.5%", var_tab[0.995]["lognormal"], "#D55E00", ":")]
    for lbl, xv, col, ls in marks:
        ax.axvline(xv / 1e6, color=col, ls=ls, lw=1.3, zorder=3,
                   label=f"{lbl}: {xv:,.0f}")
    ax.set_xlim(0.0, x_max / 1e6)
    ax.set_xlabel("Outstanding amount (TRY million, real)" if variant == "incurred"
                  else "Total reserve (TRY million, real)")
    ax.set_ylabel("Number of draws")
    ax.yaxis.set_major_formatter(fmt_int)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.1f}"))
    ax.grid(axis="x", visible=False)
    # legend below the axes: lines span the full height, so an in-axes legend
    # would be crossed by the VaR 95% / 99.5% lines
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2,
              title="Marked levels (real TRY)", title_fontsize=7,
              handlelength=2.8, columnspacing=1.8, borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"19{SFX}_reserve_distribution.png")
    fig.savefig(FIGURES_DIR / f"19{SFX}_reserve_distribution.pdf"); plt.close(fig)
    outfiles.append(str((FIGURES_DIR / f"19{SFX}_reserve_distribution.png").resolve()))
    outfiles.append(str((FIGURES_DIR / f"19{SFX}_reserve_distribution.pdf").resolve()))
    # Data S1 (R1 final): bin counts of the Figure 10 histogram (the same
    # np.histogram call as ax.hist) — a NEW file only; the draws, the seed and
    # the existing outputs are NOT AFFECTED (synthetic lognormal draws)
    h10, _e10 = np.histogram(reserves / 1e6, bins=bins)
    fp10 = OUTPUTS_DIR / f"19{SFX}_fig10_reserve_histogram.csv"
    pd.DataFrame({"bin": np.arange(1, len(h10) + 1),
                  "bin_left_TRY_million": bins[:-1],
                  "bin_right_TRY_million": bins[1:],
                  "n_draws": h10}).to_csv(fp10, index=False)
    outfiles.append(str(fp10.resolve()))

    # ---- Figure 11: optimal confidence level vs penalty ratio --------------
    fig, ax = plt.subplots(figsize=(FIG_W_TEXT, 3.0))
    xr = odf["penalty_ratio_cu_co"].values
    ax.plot(xr, 100 * odf["critical_fractile"], "o-", color="#0072B2",
            ms=4.5, lw=1.2, zorder=3)
    for _, r in odf.iterrows():
        ax.annotate(f"{r['optimal_reserve']:,.0f}",
                    (r["penalty_ratio_cu_co"], 100 * r["critical_fractile"]),
                    fontsize=7, xytext=(5, -5), textcoords="offset points",
                    ha="left", va="top")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(list(xr)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(0.85, 16.0)
    ax.set_ylim(45, 97)
    ax.set_xlabel("Penalty ratio $c_u/c_o$ (log scale)")
    ax.set_ylabel("Optimal confidence level (%)")
    ax.text(0.02, 0.97, "Point labels: optimal reserve $R^*$\n"
            "(lognormal, real TRY)", transform=ax.transAxes, fontsize=7,
            ha="left", va="top")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"19{SFX}_optimal_confidence.png")
    fig.savefig(FIGURES_DIR / f"19{SFX}_optimal_confidence.pdf"); plt.close(fig)
    outfiles.append(str((FIGURES_DIR / f"19{SFX}_optimal_confidence.png").resolve()))
    outfiles.append(str((FIGURES_DIR / f"19{SFX}_optimal_confidence.pdf").resolve()))

    table7_labels = {"best_estimate": "Best estimate (chain-ladder)",
                     "VaR_0.75": "VaR 75%", "TVaR_0.75": "TVaR 75%",
                     "VaR_0.95": "VaR 95%", "TVaR_0.95": "TVaR 95%",
                     "VaR_0.995": "VaR 99.5% (Solvency-II-style)",
                     "TVaR_0.995": "TVaR 99.5%",
                     "coc_margin": "BE + CoC risk margin (6%)"}
    scr_t_def = (
        "SCR_t = SCR_0 * O_t / O_0, t = 0..T-1 (T = number of development years "
        f"= {rm_detail['T']}); SCR_0 = VaR_{SOLVENCY_ALPHA}(R) - BE from the "
        "Mack-moment lognormal predictive distribution of the total reserve R "
        "(an ULTIMATE-view proxy for the one-year Solvency II reserve-risk SCR). "
        "O_t is the expected unpaid-claims run-off from the chain-ladder "
        "projection: P_k = expected payment in future calendar year k (k = 1.."
        f"{rm_detail['T']-1}; negative expected increments floored at 0). "
        + ("Timing 'as_submitted': O_t = sum_{k>=t} P_k with P_0 = 0, so "
           "O_0 = O_1 = total expected (positive) future payments, i.e. the SCR "
           "follows the payment pattern with a one-year lag (full SCR_0 is held "
           "in both t = 0 and t = 1)."
           if RISK_MARGIN_TIMING == "as_submitted" else
           "Timing 'solvency2_standard': O_t = sum_{k>=t+1} P_k = expected "
           "outstanding at the end of run-off year t (O_0 = total).")
        + " Discounting: (1 + r_f)^(t+1) with r_f = RESERVE_DISCOUNT_RATE.")

    manifest = {
        "script": SCRIPT_NAME,
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "triangle_variant": variant,
        "triangle_file": TRIANGLE_FILES[variant],
        "level_and_se_source": ("22_manifest incurred_CL (outstanding = ultimate - paid to date; Mack s.e. of the incurred CL)"
                                if variant == "incurred" else "03_manifest chain_ladder.ibnr / mack.total_se"),
        "runoff_pattern_source": "paid triangle chain-ladder expected payments (both variants)",
        "odp_bootstrap_computed": bool(odp_ok),
        "cl_point_reserve": float(be_point),
        "predictive": {"model": "Mack-moment lognormal", "reserve": be,
                       "mack_total_se": sd, "cv": cv},
        "odp_bootstrap_sensitivity": {"B": int(RESERVE_BOOTSTRAP_N),
                                      "mean": bs_mean, "se": bs_se, "phi": phi,
                                      "process_error": RESERVE_PROCESS_ERROR,
                                      "caveat": "olgun-yıl geri-kazanımı φ'yi şişirir"},
        "predictive_sensitivity": dist_cmp,
        "scr0_reserve_risk": scr0, "coc_risk_margin": rm,
        "technical_provision_coc": tp_coc,
        "strategies": strat, "optimal_newsvendor": opt,
        # ---------------- R1 additions ----------------
        "risk_margin": {
            "formula": "RM = CoC * sum_{t=0}^{T-1} SCR_t / (1 + r_f)^(t+1)",
            "coc_rate": RESERVE_COC_RATE,
            "discount_rate_r_f": RESERVE_DISCOUNT_RATE,
            "discount_note": ("r_f = 0: all amounts are in constant (December-2018) "
                              "prices, i.e. real and undiscounted "
                              "(config.RESERVE_DISCOUNT_RATE)"),
            "scr0_definition": (f"VaR_{SOLVENCY_ALPHA}(R) - BE, Mack-moment lognormal "
                                "(ultimate-view proxy for the one-year reserve-risk SCR)"),
            "scr0": scr0,
            "scr_t_definition": scr_t_def,
            **rm_detail,
            "rm": rm, "technical_provision": tp_coc,
            "alternative_timing": {
                "timing": alt_timing, "rm": rm_alt, "technical_provision": tp_alt,
                "ratio_to_be": tp_alt / be,
                "O_t_over_O0": rm_alt_detail["O_t_over_O0"],
                "sum_O_t_over_O0": rm_alt_detail["sum_O_t_over_O0"],
                "SCR_t": rm_alt_detail["SCR_t"], "rm_term_t": rm_alt_detail["rm_term_t"],
            },
        },
        "table7_ratio_x_be": {table7_labels.get(s["strategy"], s["strategy"]):
                              s["ratio_to_be_2dp"] for s in strat},
        "predictive_distributions": {
            "moment_matching": {"mean": be, "sd": sd, "cv": cv,
                                "source": "mean = CL reserve (03 chain_ladder.ibnr), "
                                          "sd = Mack total s.e. (03 mack.total_se)"},
            "params": PAR, "moment_check": moment_check,
            "methods": {
                "lognormal": "sigma^2 = ln(1+CV^2), mu = ln(mean) - sigma^2/2",
                "gamma": "k = 1/CV^2, theta = sd^2/mean",
                "weibull": ("shape c solves lnG(1+2/c) - 2 lnG(1+1/c) = ln(1+CV^2) "
                            f"(Brent, bracket {WEIBULL_SHAPE_BRACKET}, xtol "
                            f"{WEIBULL_SHAPE_XTOL}); scale = mean/G(1+1/c)"),
                "invgauss": "IG(mu = mean, lambda = mean^3/sd^2); scipy invgauss(mu/lambda, scale=lambda)",
                "tvar": ("closed forms: LN exp(mu+s^2/2)Phi(s-z_q)/(1-q); gamma "
                         "m*S_{k+1}(v)/(1-q); Weibull m*Q(1+1/c,(v/lam)^c)/(1-q); IG "
                         "partial-expectation formula; cross-checked against "
                         "VaR + int_v^inf S(x)dx/(1-q) (quad)"),
            },
            "tvar_check_max_rel_diff": float(max(tvar_check.values())),
        },
        "distribution_sensitivity_summary": ns_summary,
        "coc_tp_implied_cu_co": tp_implied,
        "coc_tp_newsvendor_twin": {
            "cu_co_grid_point": float(odf.loc[(odf["optimal_reserve"] - tp_coc)
                                              .abs().idxmin(), "penalty_ratio_cu_co"]),
            "Rstar_ln_at_twin": float(odf.loc[(odf["optimal_reserve"] - tp_coc)
                                              .abs().idxmin(), "optimal_reserve"]),
            "tp_over_twin": float(tp_coc / odf.loc[(odf["optimal_reserve"] - tp_coc)
                                                   .abs().idxmin(), "optimal_reserve"]),
        },
        "figures": {
            "19_reserve_distribution": {
                "paper": "Figure 10", "n_draws": int(N_PRED_DRAWS), "seed": GLOBAL_SEED,
                "x_range_TRY": [0.0, x_max], "x_range_quantile": FIG10_XMAX_Q,
                "n_bins": FIG10_N_BINS, "draws_beyond_x_range": n_beyond,
                "marked": {lbl: float(xv) for lbl, xv, _c, _l in marks},
                "data_s1_bin_counts_csv": "19_fig10_reserve_histogram.csv"},
            "19_optimal_confidence": {"paper": "Figure 11", "x_scale": "log",
                                      "labels": "lognormal R* (real TRY)"},
        },
        "params": {"conf_levels": RESERVE_CONF_LEVELS,
                   "penalty_grid": RESERVE_PENALTY_GRID,
                   "coc_rate": RESERVE_COC_RATE, "discount": RESERVE_DISCOUNT_RATE,
                   "solvency_alpha": SOLVENCY_ALPHA,
                   "pred_dists": list(PRED_DISTS), "n_pred_draws": N_PRED_DRAWS,
                   "risk_margin_timing": RISK_MARGIN_TIMING,
                   "tvar_check_rtol": TVAR_CHECK_RTOL, "seed": GLOBAL_SEED},
        "output_files": outfiles,
        "environment": env_info(),
    }
    with open(OUTPUTS_DIR / f"19{SFX}_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
    log.info("\n" + "=" * 78); log.info("SCRIPT 19 COMPLETED"); log.info("=" * 78)


if __name__ == "__main__":
    for _v in RESERVE_TRIANGLE_VARIANTS:
        main(_v)
