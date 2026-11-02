"""
================================================================================
SCRIPT 14b — Extended Premium Pricing Models
================================================================================
PURPOSE
    Deepens the pricing side to doctoral level: compares several frequency and
    pure-premium models on the SAME data split and metrics; reports
    discriminatory power (Gini) AND calibration (decile actual-vs-predicted,
    lift, loss ratio, P/O) together.

DATA SPLIT (leakage-free, defensible hyperparameter selection)
    fit   : 2014–2015  (model fitting)
    valid : 2016       (Tweedie var_power selection [profile log-likelihood]
                        + GBM early-stopping)
    GLMs (Poisson/NB/two-stage/Tweedie-final) are refitted on 2014–2016;
    GBMs are fitted on a 200,000 sub-sample of 2014–2015 (severity GBM: all
    2014–2015 policies with claims), 2016 used only for early-stopping.
    test  : 2017–2018  (reported final metrics)

PACKAGE POLICY (academic reproduction)
    All models are implemented with numpy/scipy → external packages are NOT
    REQUIRED. Standard packages (statsmodels/lightgbm/xgboost) run as an extra
    validation IF INSTALLED; OTHERWISE they are skipped with a clean info
    message (NO automatic installation; see requirements.txt / environment.yml).

MODELS COVERED
    Frequency: Poisson GLM, Negative Binomial GLM (alpha estimated FROM DATA)
    Pure-prem: Tweedie GLM (var_power: 1.1→1.9 step 0.05, via the 2016 profile
               log-likelihood),
               Tweedie histogram GBM (early-stopping)
    Optional : statsmodels Tweedie, LightGBM, XGBoost (validation + early-stop)

INPUT : data/proc/analytical_dataset.parquet
OUTPUT: outputs/14b_pricing_models.csv ; outputs/14b_calibration.csv ;
        outputs/14b_recalibration.csv ; outputs/14b_tweedie_profile.csv (R1) ;
        outputs/14b_factorial_effects.csv (R1) ; outputs/14b_calibration_slopes.csv
        (R1) ; outputs/14b_model_specs.csv (R1) ; figures/14b_*.{pdf,png} ;
        outputs/14b_manifest.json ; data/proc/pricing_best_preds.parquet

R1 REVISION (mathematics-4559487, Sep 2026)
    * Tweedie GLM power p: selected by the 2016 validation PROFILE
      log-likelihood (exact Dunn-Smyth series density, ML dispersion) instead
      of the validation deviance, which is not comparable across p.
    * LightGBM/XGBoost: single thread, deterministic, fixed seed; XGBoost now
      predicts with the early-stopped model (was: all trained trees).
    * Manifest documents per-model fit years, n, validation use, tree counts
      and hyperparameters; three defined calibration slopes with CIs; paired
      bootstrap CIs for the 2x2 learner/structure contrasts.
    * Figures 7-8 redrawn at print size (MDPI rules).
    Unchanged: which years each model uses; best-model rule (max TEST Gini).
================================================================================
"""
from __future__ import annotations
import sys
import json
import logging
import importlib
import warnings
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

from config import (
    DATA_PROC_DIR, OUTPUTS_DIR, FIGURES_DIR, LOGS_DIR, GLOBAL_SEED,
    TRAIN_YEARS, TEST_YEARS, FIT_YEARS, VALID_YEARS,
    NUMERIC_COVARS, CATEG_COVARS, TREE_TRAIN_CAP, FIG_DPI,
    TWEEDIE_P_MIN, TWEEDIE_P_MAX, TWEEDIE_P_STEP, TWEEDIE_DEFAULT_P,
    GBM14B_MAX_TREES, GBM14B_LR, GBM14B_MAX_DEPTH, GBM14B_N_BINS,
    GBM14B_PATIENCE, PKG_NUM_BOOST_ROUND, PKG_EARLY_STOPPING,
    LGB_PARAMS_BASE, XGB_PARAMS_BASE,
    TWOSTAGE_GBM_ENABLED, RECALIBRATION_ENABLED, RECAL_N_BINS, PAIRED_GINI_B,
    PRICING_LEVEL_ADJUSTMENT,
    FIG_W_FULL, FIG_W_TEXT, apply_pub_style, thousands_formatter, env_info,
)
from scipy.special import gammaln, logsumexp
from scipy.optimize import minimize_scalar

# --- R1 revision constants ---------------------------------------------------
# (mathematics-4559487, R1 code fix; module-specific design choices, recorded
#  verbatim in outputs/14b_manifest.json -> "r1_design_choices")
# Tweedie GLM variance power: chosen by the validation-year (2016) Tweedie
# PROFILE log-likelihood (exact Dunn-Smyth series density, dispersion profiled
# out by maximum likelihood). Tweedie deviances are NOT comparable across p
# (they fall monotonically in p), so the pre-R1 deviance criterion always
# picked the grid boundary; it is still reported for transparency only.
R1_TWEEDIE_P_CRITERION = "validation_profile_loglik"
R1_TWEEDIE_PHI_METHOD = "ml"        # ML dispersion; Pearson fallback if unstable
R1_TWEEDIE_SERIES_TOL = 37.0        # drop series terms < max term * e^-37 (~1e-16)
R1_TWEEDIE_SERIES_K0 = 16           # initial half-window of j around j_max
R1_TWEEDIE_SERIES_KMAX = 1 << 14    # hard cap on the half-window (never reached)
R1_PHI_LOG_HALFWIDTH = 6.0          # ML search interval: log(phi_Pearson) +/- 6
R1_PHI_XATOL = 1e-7                 # bounded-Brent tolerance on log(phi)
R1_LOGPDF_TOL = 1e-6                # validation tolerance (abs. log-density)
R1_LOGPDF_TOL_LWB = 1e-5            # vs scipy log_wright_bessel (its asymptotic
                                    # branch is accurate to ~1e-6 at x > 1e38)
# Package determinism (single thread, fixed seed; explicit hist method so a
# changed default in a later release cannot silently change the result).
R1_XGB_DETERMINISM = {"nthread": 1, "seed": GLOBAL_SEED, "tree_method": "hist"}
R1_LGB_DETERMINISM = {"num_threads": 1, "deterministic": True,
                      "force_row_wise": True, "seed": GLOBAL_SEED}
# In-house histogram-tree constants (pre-R1 values, unchanged; now named)
R1_TREE_L2_LAMBDA = 1.0             # L2 penalty on leaf values
R1_TREE_MIN_NODE_SPLIT = 50         # a node with < 50 obs is not split
R1_TREE_MIN_CHILD = 20              # each child of a split needs >= 20 obs
# 2x2 factorial contrasts (paired bootstrap on the same resamples as Table 6)
R1_FACTORIAL_CONTRASTS = [
    ("learner_direct", "Tweedie GBM", "Tweedie GLM",
     "Learner effect, direct structure: Tweedie GBM - Tweedie GLM"),
    ("learner_twostage", "Two-stage GBM", "Two-stage GLM",
     "Learner effect, two-stage structure: two-stage GBM - two-stage GLM"),
    ("structure_GLM", "Two-stage GLM", "Tweedie GLM",
     "Structure effect, GLM learner: two-stage GLM - Tweedie GLM"),
    ("structure_GBM", "Two-stage GBM", "Tweedie GBM",
     "Structure effect, GBM learner: two-stage GBM - Tweedie GBM"),
]
R1_CALIB_N_DEC = 10
# Figure palette (validated reference categorical slots 1-3 + neutral gray)
R1_GROUP_COLORS = {"freq": "#2a78d6", "direct": "#eb6834",
                   "twostage": "#1baf7a", "optional": "#7a7974"}

SCRIPT_NAME = "14b_pricing_models"
warnings.filterwarnings("ignore")
# Tweedie var_power grid is derived from config.py (hard-coding removed)
TWEEDIE_GRID = np.round(np.arange(TWEEDIE_P_MIN,
                                  TWEEDIE_P_MAX + TWEEDIE_P_STEP / 2,
                                  TWEEDIE_P_STEP), 2)


def setup_logger():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(SCRIPT_NAME); lg.setLevel(logging.INFO); lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOGS_DIR / f"{SCRIPT_NAME}.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt); sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    lg.addHandler(fh); lg.addHandler(sh); return lg


log = logging.getLogger(SCRIPT_NAME)


def set_style():
    plt.rcParams.update({"font.family": "serif", "font.size": 9,
        "savefig.dpi": FIG_DPI,
        "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25})


def try_import(name):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def load_proc(stem):
    pq = DATA_PROC_DIR / f"{stem}.parquet"
    return pd.read_parquet(pq) if pq.exists() else pd.read_pickle(DATA_PROC_DIR / f"{stem}.pkl")


def design_stats(df):
    """Standardisation parameters (mean, SD) of the numeric covariates,
    computed on ONE frame (the training years) and then applied unchanged to
    the validation and test years (R1b: previously every split was scaled by
    its own mean/SD, which used the test-year distribution at prediction time)."""
    st = {}
    for c in NUMERIC_COVARS:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce").fillna(0).values.astype(float)
            st[c] = (float(x.mean()), float(x.std()))
    return st


def design(df, ref=None, stats=None):
    parts = {}
    if stats is None:
        stats = design_stats(df)
    for c in NUMERIC_COVARS:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce").fillna(0).values.astype(float)
            m, s = stats[c]
            parts[c] = (x - m) / (s + 1e-9)
    base = pd.DataFrame(parts, index=df.index)
    cats = []
    for c in CATEG_COVARS:
        if c in df.columns:
            cats.append(pd.get_dummies(df[c].astype(str), prefix=c, drop_first=True).astype(float))
    X = pd.concat([base] + cats, axis=1) if cats else base
    if ref is not None:
        X = X.reindex(columns=ref, fill_value=0.0)
    return X


# ----- metrics -----
def gini(y_true, y_pred):
    order = np.argsort(y_pred); ys = np.asarray(y_true)[order]
    cum = np.cumsum(ys) / (np.sum(ys) + 1e-12); n = len(y_true)
    g = 1 - 2 * np.sum(cum) / n
    op = np.argsort(y_true); yp = np.asarray(y_true)[op]
    cp = np.cumsum(yp) / (np.sum(yp) + 1e-12)
    gp = 1 - 2 * np.sum(cp) / n
    return float(g / gp) if gp != 0 else 0.0


def poisson_dev(y, mu):
    mu = np.clip(mu, 1e-9, None)
    return float(2 * np.sum(np.where(y > 0, y*np.log(y/mu), 0.0) - (y - mu)))


def tweedie_dev(y, mu, p=1.5):
    mu = np.clip(mu, 1e-9, None); y = np.clip(y, 0, None)
    a = np.where(y > 0, y**(2-p)/((1-p)*(2-p)), 0.0)
    b = y*mu**(1-p)/(1-p); c = mu**(2-p)/(2-p)
    return float(2 * np.sum(a - b + c))


def calibration_table(y_true, y_pred, n_dec=10):
    """Decile-based actual vs predicted + P/O + loss ratio components."""
    order = np.argsort(y_pred)
    yt = np.asarray(y_true)[order]; yp = np.asarray(y_pred)[order]
    n = len(yt); idx = np.array_split(np.arange(n), n_dec)
    rows = []
    for d, ix in enumerate(idx, 1):
        a = yt[ix].mean(); pr = yp[ix].mean()
        rows.append({"decile": d, "n": len(ix), "actual_mean": float(a),
                     "pred_mean": float(pr),
                     "PO_ratio": float(pr / a) if a > 0 else np.nan})
    return pd.DataFrame(rows)


def calibration_slope(y_true, y_pred):
    """Slope of log(actual) ~ log(pred) (1.0 = ideal calibration)."""
    m = (y_pred > 0) & (y_true > 0)
    if m.sum() < 10:
        return np.nan
    x = np.log(y_pred[m]); y = np.log(y_true[m])
    b = np.polyfit(x, y, 1)[0]
    return float(b)


# ----- pure-numpy GLM (IRLS) -----
def glm_irls(X, y, offset=None, family="poisson", nb_alpha=1.0,
             tw_p=1.5, n_iter=40, ridge=1e-6, w=None, info=None):
    """IRLS (Fisher scoring) for log-link GLMs. `info` (optional dict) receives
    convergence diagnostics; it does not change the fit."""
    n, k = X.shape
    it_used = 0; last_step = float("nan"); converged = False
    beta = np.zeros(k)
    off = np.zeros(n) if offset is None else offset
    ybar = max(np.mean(y[y > 0]) if family == "tweedie" and (y > 0).any()
               else np.mean(y), 1e-3)
    if np.allclose(X[:, 0], 1.0):
        beta[0] = np.log(ybar) - np.mean(off)
    for _ in range(n_iter):
        eta = X @ beta + off
        mu = np.exp(np.clip(eta, -20, 20))
        if family == "poisson":
            W = mu
        elif family == "negbin":
            W = mu / (1 + nb_alpha * mu)
        elif family == "tweedie":
            W = mu ** (2 - tw_p)
        elif family == "gamma":
            # log-link gamma: IRLS weight (dμ/dη)²/V(μ) = μ²/μ² = 1
            W = np.ones(n)
        else:
            W = mu
        if w is not None:
            W = W * w
        z = (y - mu) / np.clip(mu, 1e-9, None)
        H = (X.T * W) @ X + ridge * np.eye(k)
        g = X.T @ (W * z)
        try:
            step = np.clip(np.linalg.solve(H, g), -2.0, 2.0)
        except np.linalg.LinAlgError:
            break
        beta += step
        it_used += 1; last_step = float(np.max(np.abs(step)))
        if np.max(np.abs(step)) < 1e-8:
            converged = True
            break
    if info is not None:
        info.update({"iterations": int(it_used), "max_iter": int(n_iter),
                     "last_max_abs_step": last_step,
                     "converged_1e-8": bool(converged)})
    return beta


# ----- exact Tweedie log-likelihood (1 < p < 2; Dunn & Smyth 2005 series) -----
def _tweedie_log_series(y, phi, p, tol=R1_TWEEDIE_SERIES_TOL):
    """log sum_{j>=1} W_j for y > 0 (vectorised, log-space).

    W_j = z^j / (j! Gamma(-j*alpha)),  alpha = (2-p)/(1-p) < 0,
    z = y^(-alpha) (p-1)^alpha / (phi^(1-alpha) (2-p)).
    log W_j is concave in j (lgamma convex), so the sum is taken over a window
    centred at j_max = y^(2-p)/(phi(2-p)) that is widened (doubling) until both
    window edges are below max_j log W_j - tol, i.e. < 1e-16 relative.
    Returns (log-sum, largest half-window used)."""
    y = np.asarray(y, float)
    out = np.empty(len(y))
    if len(y) == 0:
        return out, 0
    a = (2.0 - p) / (1.0 - p)
    na = -a
    logz = (-a * np.log(y) + a * np.log(p - 1.0)
            - (1.0 - a) * np.log(phi) - np.log(2.0 - p))
    jmax = np.maximum(1.0, np.round(np.exp((2.0 - p) * np.log(y))
                                    / (phi * (2.0 - p))))
    todo = np.arange(len(y)); K = R1_TWEEDIE_SERIES_K0; k_used = 0
    while todo.size:
        ks = np.arange(-K, K + 1, dtype=float)
        J = jmax[todo, None] + ks[None, :]
        ok = J >= 1.0
        Jc = np.where(ok, J, 1.0)
        LW = Jc * logz[todo, None] - gammaln(Jc + 1.0) - gammaln(na * Jc)
        LW = np.where(ok, LW, -np.inf)
        rmax = LW.max(axis=1)
        conv = ((LW[:, -1] < rmax - tol)
                & ((~ok[:, 0]) | (LW[:, 0] < rmax - tol)))
        if K >= R1_TWEEDIE_SERIES_KMAX:
            conv[:] = True
        out[todo[conv]] = logsumexp(LW[conv], axis=1)
        k_used = K
        todo = todo[~conv]; K *= 2
    return out, int(k_used)


def tweedie_logpdf(y, mu, phi, p):
    """Exact Tweedie log-density (1<p<2) w.r.t. counting measure at 0 plus
    Lebesgue measure on y>0:  y=0: -mu^(2-p)/(phi(2-p));
    y>0: -log y + log sum_j W_j + [y mu^(1-p)/(1-p) - mu^(2-p)/(2-p)]/phi."""
    y = np.asarray(y, float)
    mu = np.clip(np.asarray(mu, float), 1e-12, None)
    ll = (y * mu ** (1.0 - p) / (1.0 - p) - mu ** (2.0 - p) / (2.0 - p)) / phi
    pos = y > 0
    if pos.any():
        ls, _ = _tweedie_log_series(y[pos], phi, p)
        ll[pos] += ls - np.log(y[pos])
    return ll


def tweedie_loglik(y, mu, phi, p):
    return float(np.sum(tweedie_logpdf(y, mu, phi, p)))


def tweedie_phi_pearson(y, mu, p, n_params=0):
    mu = np.clip(mu, 1e-12, None)
    return float(np.sum((y - mu) ** 2 / mu ** p) / max(len(y) - n_params, 1))


def tweedie_phi_ml(y, mu, p, phi_start):
    """ML dispersion for fixed (mu, p): bounded Brent on log(phi) over
    log(phi_start) +/- R1_PHI_LOG_HALFWIDTH. Falls back to the Pearson
    estimate `phi_start` if the optimiser fails or stops at a bound.
    Returns (phi, method, loglik)."""
    lo = np.log(phi_start) - R1_PHI_LOG_HALFWIDTH
    hi = np.log(phi_start) + R1_PHI_LOG_HALFWIDTH

    def nll(lphi):
        return -tweedie_loglik(y, mu, float(np.exp(lphi)), p)

    try:
        res = minimize_scalar(nll, bounds=(lo, hi), method="bounded",
                              options={"xatol": R1_PHI_XATOL})
        at_bound = min(res.x - lo, hi - res.x) < 1e-3
        if res.success and np.isfinite(res.fun) and not at_bound:
            return float(np.exp(res.x)), "ml", float(-res.fun)
    except Exception:
        pass
    return (float(phi_start), "pearson_fallback",
            tweedie_loglik(y, mu, float(phi_start), p))


def _cpg_logpdf(y, mu, phi, p, n_max=None):
    """Independent check: compound Poisson-gamma density
    sum_n Pois(n; lam) Gamma(y; n*k, gam), lam = mu^(2-p)/(phi(2-p)),
    k = (2-p)/(p-1), gam = phi(p-1)mu^(p-1)."""
    lam = mu ** (2 - p) / (phi * (2 - p))
    k = (2 - p) / (p - 1); gam = phi * (p - 1) * mu ** (p - 1)
    if y == 0:
        return -lam
    if n_max is None:
        n_max = int(max(200, lam + 60 * np.sqrt(lam) + 200,
                        3 * y / (k * gam) + 200))
    n = np.arange(1, n_max + 1, dtype=float)
    lt = (-lam + n * np.log(lam) - gammaln(n + 1)
          + (n * k - 1) * np.log(y) - y / gam - gammaln(n * k)
          - n * k * np.log(gam))
    return float(logsumexp(lt))


def validate_tweedie_logpdf():
    """Numerical validation of tweedie_logpdf (recorded in the manifest):
    (1) statsmodels Tweedie(eql=False).loglike_obs (Wright-Bessel);
    (2) scipy.special.log_wright_bessel; (3) the compound Poisson-gamma sum;
    (4) quadrature: P(Y=0) + int f = 1, int y f = mu, int (y-mu)^2 f = phi mu^p."""
    from scipy import integrate
    from scipy.special import log_wright_bessel
    sm = try_import("statsmodels.genmod.families")
    grid_p = (1.1, 1.3, 1.5, 1.7, 1.9)
    ys = np.array([0.0, 5.0, 150.0, 1500.0, 8000.0, 60000.0, 4.0e5])
    d_sm, d_lwb, d_cpg = [], [], []
    n_sm_nonfinite = 0; mom = []
    for p in grid_p:
        for mu in (400.0, 2500.0):
            for lam_target in (0.05, 0.3, 2.0):
                phi = mu ** (2 - p) / (lam_target * (2 - p))
                ll = tweedie_logpdf(ys, np.full(len(ys), mu), phi, p)
                a = (2 - p) / (1 - p)
                th = ys * mu ** (1 - p) / (1 - p) - mu ** (2 - p) / (2 - p)
                for i, y in enumerate(ys):
                    d_cpg.append(abs(ll[i] - _cpg_logpdf(y, mu, phi, p)))
                    if y > 0:
                        x = ((p - 1) * phi / y) ** a / ((2 - p) * phi)
                        ref = th[i] / phi - np.log(y) + log_wright_bessel(-a, 0.0, x)
                        if np.isfinite(ref):
                            d_lwb.append(abs(ll[i] - ref))
                if sm is not None:
                    ref_sm = sm.Tweedie(var_power=p, eql=False).loglike_obs(
                        ys, np.full(len(ys), mu), scale=phi)
                    fin = np.isfinite(ref_sm)
                    n_sm_nonfinite += int((~fin).sum())
                    d_sm.extend(np.abs(ll[fin] - ref_sm[fin]).tolist())
        # quadrature moments at a representative (mu, phi)
        for mu, lam_target in ((2000.0, 0.25), (800.0, 1.5)):
            phi = mu ** (2 - p) / (lam_target * (2 - p))
            sd = np.sqrt(phi * mu ** p)
            t_hi = np.log(mu + 400 * sd)
            f = lambda t, m: (np.exp(tweedie_logpdf(np.array([np.exp(t)]),
                                                   np.array([mu]), phi, p)[0]
                                     + t) * m(np.exp(t)))
            opts = dict(limit=800, epsabs=0, epsrel=1e-10,
                        points=[np.log(mu) - 10.0, np.log(mu)])
            p0 = np.exp(-mu ** (2 - p) / (phi * (2 - p)))
            m0 = integrate.quad(f, -700.0, t_hi, args=(lambda v: 1.0,), **opts)[0]
            m1 = integrate.quad(f, -700.0, t_hi, args=(lambda v: v,), **opts)[0]
            m2 = integrate.quad(f, -700.0, t_hi,
                                args=(lambda v: (v - mu) ** 2,), **opts)[0]
            m2 += p0 * mu ** 2   # (0 - mu)^2 * P(Y=0)
            mom.append({"p": p, "mu": mu, "phi": float(phi),
                        "total_mass_minus_1": float(p0 + m0 - 1.0),
                        "mean_rel_err": float(m1 / mu - 1.0),
                        "var_rel_err": float(m2 / (phi * mu ** p) - 1.0)})
    out = {
        "grid_p": list(grid_p), "y_values": ys.tolist(),
        "max_abs_diff_vs_statsmodels": (float(max(d_sm)) if d_sm else None),
        "n_statsmodels_nonfinite": int(n_sm_nonfinite),
        "max_abs_diff_vs_log_wright_bessel": float(max(d_lwb)),
        "max_abs_diff_vs_compound_poisson_gamma": float(max(d_cpg)),
        "quadrature_checks": mom,
        "max_abs_quadrature_err": float(max(max(abs(r["total_mass_minus_1"]),
                                                abs(r["mean_rel_err"]),
                                                abs(r["var_rel_err"]))
                                            for r in mom)),
    }
    out["passed"] = bool(
        out["max_abs_diff_vs_log_wright_bessel"] < R1_LOGPDF_TOL_LWB
        and out["max_abs_diff_vs_compound_poisson_gamma"] < R1_LOGPDF_TOL
        and (out["max_abs_diff_vs_statsmodels"] is None
             or out["max_abs_diff_vs_statsmodels"] < R1_LOGPDF_TOL)
        and out["max_abs_quadrature_err"] < 1e-5)
    return out


# ----- calibration diagnostics (definitions recorded in the manifest) -----
def _ols_slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    xc = x - x.mean()
    b = float(np.sum(xc * (y - y.mean())) / np.sum(xc * xc))
    return b, float(y.mean() - b * x.mean())


def calibration_diagnostics(y_true, y_pred, n_dec=R1_CALIB_N_DEC):
    """Three calibration slopes (1.0 = ideal for all three):
    loglog_claimants: OLS slope of log(actual) on log(predicted) over test
        policies with actual > 0 and predicted > 0 (claimant policies) -- the
        pre-R1 'calibration slope';
    decile_loglog: OLS slope of log(decile mean actual) on log(decile mean
        predicted) across the n_dec equal-count deciles of predicted;
    policy_linear: OLS slope (with intercept) of actual on predicted over ALL
        test policies (zeros included)."""
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    m = (y_pred > 0) & (y_true > 0)
    b_ll, a_ll = _ols_slope(np.log(y_pred[m]), np.log(y_true[m]))
    ct = calibration_table(y_true, y_pred, n_dec=n_dec)
    b_dec, a_dec = _ols_slope(np.log(ct["pred_mean"]), np.log(ct["actual_mean"]))
    b_lin, a_lin = _ols_slope(y_pred, y_true)
    return {"loglog_claimants": b_ll, "loglog_claimants_intercept": a_ll,
            "n_claimants_used": int(m.sum()), "n_policies": int(len(y_true)),
            "decile_loglog": b_dec, "decile_loglog_intercept": a_dec,
            "policy_linear": b_lin, "policy_linear_intercept": a_lin,
            "overall_PO": float(y_pred.sum() / max(y_true.sum(), 1e-9))}


def _calib_slopes_fast(y, pred, n_dec=R1_CALIB_N_DEC):
    """Same three slopes as calibration_diagnostics (bootstrap inner loop)."""
    m = (pred > 0) & (y > 0)
    b_ll = _ols_slope(np.log(pred[m]), np.log(y[m]))[0]
    o = np.argsort(pred)
    ys, ps = y[o], pred[o]
    ix = np.array_split(np.arange(len(y)), n_dec)
    am = np.array([ys[i].mean() for i in ix]); pm = np.array([ps[i].mean() for i in ix])
    b_dec = _ols_slope(np.log(pm), np.log(am))[0] if (am > 0).all() else np.nan
    b_lin = _ols_slope(pred, y)[0]
    return b_ll, b_dec, b_lin


def glm_predict(X, beta, offset=None):
    eta = X @ beta + (0 if offset is None else offset)
    return np.exp(np.clip(eta, -20, 20))


def estimate_nb_alpha(y, mu):
    """Estimate the NB2 overdispersion alpha from moments (data-driven).
    Var = mu + alpha*mu^2  →  alpha ≈ mean[((y-mu)^2 - mu)/mu^2]."""
    mu = np.clip(mu, 1e-6, None)
    a = np.mean(((y - mu)**2 - mu) / mu**2)
    return float(max(a, 1e-3))


# ----- pure-numpy Tweedie histogram GBM (early-stopping) -----
def hist_bin(X, n_bins=32):
    Xb = np.zeros_like(X, dtype=np.int16); edges = []
    for j in range(X.shape[1]):
        q = np.unique(np.quantile(X[:, j], np.linspace(0, 1, n_bins+1)[1:-1]))
        Xb[:, j] = np.digitize(X[:, j], q); edges.append(q)
    return Xb, edges


def apply_bin(X, edges):
    Xb = np.zeros_like(X, dtype=np.int16)
    for j in range(X.shape[1]):
        Xb[:, j] = np.digitize(X[:, j], edges[j])
    return Xb


class _Node:
    __slots__ = ("f", "t", "L", "R", "val")
    def __init__(self): self.f=-1; self.t=0; self.L=None; self.R=None; self.val=0.0


def _fit_tree(Xb, g, h, depth, max_depth, n_bins, lam=R1_TREE_L2_LAMBDA):
    nd = _Node(); G, H = g.sum(), h.sum(); nd.val = -G/(H+lam)
    if depth >= max_depth or len(g) < R1_TREE_MIN_NODE_SPLIT: return nd
    bg, bf, bt = 0.0, -1, 0
    for f in range(Xb.shape[1]):
        for t in range(n_bins):
            m = Xb[:, f] <= t
            if m.sum() < R1_TREE_MIN_CHILD or (~m).sum() < R1_TREE_MIN_CHILD: continue
            GL, HL = g[m].sum(), h[m].sum(); GR, HR = G-GL, H-HL
            gain = GL**2/(HL+lam) + GR**2/(HR+lam) - G**2/(H+lam)
            if gain > bg: bg, bf, bt = gain, f, t
    if bf < 0: return nd
    nd.f, nd.t = bf, bt; m = Xb[:, bf] <= bt
    nd.L = _fit_tree(Xb[m], g[m], h[m], depth+1, max_depth, n_bins, lam)
    nd.R = _fit_tree(Xb[~m], g[~m], h[~m], depth+1, max_depth, n_bins, lam)
    return nd


def _pred_tree(nd, Xb):
    out = np.empty(len(Xb))
    def rec(n, idx):
        if n.f < 0 or n.L is None: out[idx] = n.val; return
        m = Xb[idx, n.f] <= n.t; rec(n.L, idx[m]); rec(n.R, idx[~m])
    rec(nd, np.arange(len(Xb))); return out


def tweedie_gbm(Xtr_b, ytr, Xval_b, yval, Xte_b, p=TWEEDIE_DEFAULT_P,
                max_trees=GBM14B_MAX_TREES, lr=GBM14B_LR,
                max_depth=GBM14B_MAX_DEPTH, n_bins=GBM14B_N_BINS,
                patience=GBM14B_PATIENCE):
    """Tweedie histogram GBM + validation early-stopping.

    Returns: (test prediction, valid prediction, best round) — the valid
    prediction is needed for recalibration (PAVA)."""
    base = np.log(max(ytr[ytr > 0].mean() if (ytr > 0).any() else 1.0, 1e-3))
    Ftr = np.full(len(ytr), base); Fval = np.full(len(yval), base)
    Fte = np.full(Xte_b.shape[0], base)
    best_val = np.inf; best_Fte = Fte.copy(); best_Fval = Fval.copy()
    best_round = 0; bad = 0
    for it in range(max_trees):
        mu = np.exp(np.clip(Ftr, -20, 20))
        grad = mu**(2-p) - ytr * mu**(1-p)
        hess = np.clip((2-p) * mu**(2-p), 1e-6, None)
        nd = _fit_tree(Xtr_b, grad, hess, 0, max_depth, n_bins)
        Ftr += lr * _pred_tree(nd, Xtr_b)
        Fval += lr * _pred_tree(nd, Xval_b)
        Fte += lr * _pred_tree(nd, Xte_b)
        vdev = tweedie_dev(yval, np.exp(np.clip(Fval, -20, 20)), p)
        if vdev < best_val - 1e-6:
            best_val = vdev; best_Fte = Fte.copy(); best_Fval = Fval.copy()
            best_round = it + 1; bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    return (np.exp(np.clip(best_Fte, -20, 20)),
            np.exp(np.clip(best_Fval, -20, 20)), best_round)


def hist_gbm(Xtr_b, ytr, Xval_b, yval, Xte_b, family="poisson", w=None,
             Xb_extra=None, max_trees=GBM14B_MAX_TREES, lr=GBM14B_LR,
             max_depth=GBM14B_MAX_DEPTH, n_bins=GBM14B_N_BINS,
             patience=GBM14B_PATIENCE):
    """General histogram GBM (Poisson / Gamma deviance) + early-stopping.

    This function fits both components of the two-stage GBM structure
    (frequency GBM × severity GBM), so that the 'direct vs two-stage'
    comparison is NOT CONFOUNDED with the learner (GLM vs GBM) (2×2 design).
    w: sample weight (the policy's claim count in the severity stage)."""
    w = np.ones(len(ytr)) if w is None else np.asarray(w, float)
    base = np.log(max(np.average(ytr, weights=w), 1e-3))
    Ftr = np.full(len(ytr), base); Fval = np.full(len(yval), base)
    Fte = np.full(Xte_b.shape[0], base)
    Fex = (np.full(Xb_extra.shape[0], base)
           if Xb_extra is not None else None)
    best_val = np.inf; best_Fte = Fte.copy(); best_Fval = Fval.copy()
    best_Fex = None if Fex is None else Fex.copy()
    best_round = 0; bad = 0

    def dev_val(mu):
        mu = np.clip(mu, 1e-9, None)
        if family == "poisson":
            return float(2*np.sum(np.where(yval > 0,
                                           yval*np.log(yval/mu), 0.0)
                                  - (yval - mu)))
        return float(2*np.sum(-np.log(np.clip(yval, 1e-9, None)/mu)
                              + (yval - mu)/mu))

    for it in range(max_trees):
        mu = np.exp(np.clip(Ftr, -20, 20))
        if family == "poisson":
            grad = (mu - ytr) * w
            hess = np.clip(mu * w, 1e-6, None)
        else:  # gamma (log-link) deviance
            grad = (1.0 - ytr/np.clip(mu, 1e-9, None)) * w
            hess = np.clip(ytr/np.clip(mu, 1e-9, None) * w, 1e-6, None)
        nd = _fit_tree(Xtr_b, grad, hess, 0, max_depth, n_bins)
        Ftr += lr * _pred_tree(nd, Xtr_b)
        Fval += lr * _pred_tree(nd, Xval_b)
        Fte += lr * _pred_tree(nd, Xte_b)
        if Fex is not None:
            Fex += lr * _pred_tree(nd, Xb_extra)
        vdev = dev_val(np.exp(np.clip(Fval, -20, 20)))
        if vdev < best_val - 1e-6:
            best_val = vdev; best_Fte = Fte.copy(); best_Fval = Fval.copy()
            if Fex is not None:
                best_Fex = Fex.copy()
            best_round = it + 1; bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    extra = (None if best_Fex is None
             else np.exp(np.clip(best_Fex, -20, 20)))
    return (np.exp(np.clip(best_Fte, -20, 20)),
            np.exp(np.clip(best_Fval, -20, 20)), best_round, extra)


def pava_isotonic(x, y, n_bins=RECAL_N_BINS):
    """Quantile-binned weighted isotonic regression (PAVA).

    Learns a non-decreasing calibration curve from the validation
    (prediction, actual) pairs; applied to the test predictions via pava_apply.
    Corrects the slope<1 compression in a data-driven way while preserving
    monotonicity (the level is corrected without disturbing the model
    ranking/Gini)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    order = np.argsort(x); xs, ys = x[order], y[order]
    edges = np.unique(np.quantile(xs, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, xs, side="right") - 1,
                  0, len(edges) - 2)
    bx, by, bw = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            bx.append(xs[m].mean()); by.append(ys[m].mean())
            bw.append(float(m.sum()))
    # PAVA pooling
    v = []; w = []; reps = []
    for yi, wi in zip(by, bw):
        v.append(yi); w.append(wi); reps.append(1)
        while len(v) > 1 and v[-2] > v[-1]:
            tot = w[-2] + w[-1]
            v[-2] = (v[-2]*w[-2] + v[-1]*w[-1]) / tot
            w[-2] = tot; reps[-2] += reps[-1]
            v.pop(); w.pop(); reps.pop()
    fitted = np.repeat(v, reps)
    return np.asarray(bx, float), fitted


def pava_apply(bx, fitted, xnew):
    return np.interp(np.asarray(xnew, float), bx, fitted,
                     left=fitted[0], right=fitted[-1])


def _json_default(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main():
    global log
    log = setup_logger(); set_style(); apply_pub_style()
    log.info("=" * 78)
    log.info(f"SCRIPT {SCRIPT_NAME} — Extended Pricing Models")
    log.info("=" * 78)

    outfiles = []
    ana = load_proc("analytical_dataset")
    ana["exposure"] = pd.to_numeric(ana["exposure"], errors="coerce").fillna(1.0)
    ana.loc[ana["exposure"] <= 0, "exposure"] = 1.0

    fit = ana[ana["policy_year"].isin(FIT_YEARS)].copy()
    val = ana[ana["policy_year"].isin(VALID_YEARS)].copy()
    te = ana[ana["policy_year"].isin(TEST_YEARS)].copy()
    # GLMs use the whole train set (fit+valid); Tweedie/GBM select on valid
    tr = ana[ana["policy_year"].isin(TRAIN_YEARS)].copy()
    log.info(f"\n[1/6] Split: fit={len(fit):,} {FIT_YEARS}, "
             f"valid={len(val):,} {VALID_YEARS}, test={len(te):,} {TEST_YEARS}")

    # R1b: (i) ONE training window for every main model: all GLMs and GBMs are
    # fitted on the fit years 2014-2015, 2016 is used only for validation
    # (p, early stopping, model selection, recalibration), 2017-2018 only for
    # the reported test results. (ii) Covariate scaling parameters come from
    # the fit years only and are applied unchanged to 2016 and 2017-2018.
    # The former design (GLMs refitted on 2014-2016, each split scaled by its
    # own mean/SD) is retained as a sensitivity (train-window sensitivity).
    stats_fit = design_stats(fit)
    Xfit = design(fit, stats=stats_fit); cols = list(Xfit.columns)
    Xval = design(val, ref=cols, stats=stats_fit)
    Xte = design(te, ref=cols, stats=stats_fit)
    stats_tr = design_stats(tr)                    # sensitivity: 2014-2016 window
    Xtr_s = design(tr, ref=cols, stats=stats_tr)
    Xte_s = design(te, ref=cols, stats=stats_tr)
    def addconst(D): return np.hstack([np.ones((len(D), 1)), D.values.astype(float)])
    Xfit_v, Xval_v, Xte_v, Xtr_s_v, Xte_s_v = map(addconst, (Xfit, Xval, Xte, Xtr_s, Xte_s))
    Xtr_v = Xfit_v   # GLM training design = fit years (R1b)

    y_tr = fit["n_claims"].values.astype(float)        # GLM training counts (fit years)
    y_tr_s = tr["n_claims"].values.astype(float)       # sensitivity window
    y_te = te["n_claims"].values.astype(float)
    pp_fit = np.clip(fit["ultimate_rbns"].values.astype(float), 0, None)
    pp_val = np.clip(val["ultimate_rbns"].values.astype(float), 0, None)
    pp_tr = pp_fit                                     # GLM training target (fit years)
    pp_tr_s = np.clip(tr["ultimate_rbns"].values.astype(float), 0, None)
    pp_te = np.clip(te["ultimate_rbns"].values.astype(float), 0, None)
    off_tr = np.log(fit["exposure"].values); off_te = np.log(te["exposure"].values)
    off_tr_s = np.log(tr["exposure"].values)
    sev_mean = (fit.loc[fit["n_claims"] > 0, "ultimate_rbns"]
                / fit.loc[fit["n_claims"] > 0, "n_claims"]).mean()
    sev_mean_s = (tr.loc[tr["n_claims"] > 0, "ultimate_rbns"]
                  / tr.loc[tr["n_claims"] > 0, "n_claims"]).mean()

    results = []; preds_for_calib = {}; preds_valid = {}
    pp_val_true = pp_val  # recalibration target (valid actuals)
    off_val = np.log(val["exposure"].values)
    y_val = val["n_claims"].values.astype(float)

    log.info("\n[2/6] Frequency GLM: Poisson & NegBin (alpha FROM DATA)")
    bp = glm_irls(Xtr_v, y_tr, off_tr, family="poisson")
    mu_p = glm_predict(Xte_v, bp, off_te)
    results.append({"model": "Poisson GLM", "tip": "frekans",
                    "test_deviance": poisson_dev(y_te, mu_p),
                    "pp_gini": gini(pp_te, mu_p * sev_mean)})
    preds_for_calib["Poisson GLM"] = mu_p * sev_mean
    preds_valid["Poisson GLM"] = glm_predict(Xval_v, bp, off_val) * sev_mean
    log.info(f"  Poisson GLM: dev={results[-1]['test_deviance']:,.0f}, Gini={results[-1]['pp_gini']:.3f}")
    # Estimate the NB alpha from the Poisson mu, then fit the NB GLM
    mu_tr_p = glm_predict(Xtr_v, bp, off_tr)
    nb_alpha = estimate_nb_alpha(y_tr, mu_tr_p)
    bnb = glm_irls(Xtr_v, y_tr, off_tr, family="negbin", nb_alpha=nb_alpha)
    mu_nb = glm_predict(Xte_v, bnb, off_te)
    results.append({"model": "NegBin GLM", "tip": "frekans",
                    "test_deviance": poisson_dev(y_te, mu_nb),
                    "pp_gini": gini(pp_te, mu_nb * sev_mean), "nb_alpha": nb_alpha})
    preds_for_calib["NegBin GLM"] = mu_nb * sev_mean
    preds_valid["NegBin GLM"] = glm_predict(Xval_v, bnb, off_val) * sev_mean
    log.info(f"  NegBin GLM: alpha(estimated)={nb_alpha:.3f}, dev={results[-1]['test_deviance']:,.0f}, "
             f"Gini={results[-1]['pp_gini']:.3f}")

    # Two-stage GLM (Poisson frequency × Gamma severity): the GLM-based
    # two-stage cell of the 2×2 design — the missing comparison point needed to
    # separate the structure effect (direct vs two-stage) from the learner
    # effect (GLM vs GBM). Frequency: the Poisson GLM above (with exposure
    # offset). Severity: mean claim on policies with claims, Gamma GLM
    # (log-link), weight = claim count. Deterministic IRLS — does NOT consume
    # the rng (seed stream unchanged).
    log.info("\n[2b/6] Two-stage GLM (Poisson frequency × Gamma severity)")
    cl_tr = fit["n_claims"].values > 0
    sev_tr = (fit.loc[cl_tr, "ultimate_rbns"].values
              / fit.loc[cl_tr, "n_claims"].values)
    w_tr = fit.loc[cl_tr, "n_claims"].values.astype(float)
    sev_ok = sev_tr > 0
    b_sev = glm_irls(Xtr_v[cl_tr][sev_ok], sev_tr[sev_ok],
                     family="gamma", w=w_tr[sev_ok])
    sev_te_glm = glm_predict(Xte_v, b_sev)
    sev_val_glm = glm_predict(Xval_v, b_sev)
    pp_2sglm = mu_p * sev_te_glm
    results.append({"model": "Two-stage GLM (Poi×Gam)", "tip": "iki-aşamalı",
                    "test_deviance": tweedie_dev(pp_te, pp_2sglm,
                                                 TWEEDIE_DEFAULT_P),
                    "pp_gini": gini(pp_te, pp_2sglm)})
    preds_for_calib["Two-stage GLM"] = pp_2sglm
    preds_valid["Two-stage GLM"] = (glm_predict(Xval_v, bp, off_val)
                                    * sev_val_glm)
    log.info(f"  Two-stage GLM: dev={results[-1]['test_deviance']:,.0f}, "
             f"Gini={results[-1]['pp_gini']:.3f}")

    log.info(f"\n[3/6] Tweedie GLM (var_power: 2016 profile log-likelihood, "
             f"grid {TWEEDIE_GRID[0]}–{TWEEDIE_GRID[-1]} step {TWEEDIE_P_STEP})")
    # R1: the variance power is chosen by the validation-year PROFILE
    # log-likelihood. For each p: (i) Tweedie GLM fitted on the fit years
    # 2014-2015 (unchanged); (ii) dispersion phi profiled out by ML on the 2016
    # observations given the fit-year mean mu_hat_p; (iii) exact Tweedie
    # log-likelihood of 2016 (Dunn-Smyth series). Log-likelihoods are
    # densities w.r.t. the same dominating measure (atom at 0 + Lebesgue on
    # y>0) for every p, hence comparable across p, unlike deviances.
    # Deterministic (no RNG consumed -> downstream seed stream unchanged).
    ll_check = validate_tweedie_logpdf()
    log.info(f"  log-density validation: max|Δ| statsmodels="
             f"{ll_check['max_abs_diff_vs_statsmodels']:.2e}, "
             f"log_wright_bessel={ll_check['max_abs_diff_vs_log_wright_bessel']:.2e}, "
             f"CPG-series={ll_check['max_abs_diff_vs_compound_poisson_gamma']:.2e}, "
             f"quadrature={ll_check['max_abs_quadrature_err']:.2e} → "
             f"{'PASSED' if ll_check['passed'] else 'FAILED'}")
    if not ll_check["passed"]:
        log.warning("  WARNING: Tweedie log-density validation failed!")
    k_par = Xfit_v.shape[1]
    prof_rows = []; fit_betas = {}
    for vp in TWEEDIE_GRID:
        vp = float(np.round(vp, 2))
        inf_fit = {}
        b = glm_irls(Xfit_v, pp_fit, family="tweedie", tw_p=vp, n_iter=40,
                     info=inf_fit)
        fit_betas[vp] = b
        mu_f = glm_predict(Xfit_v, b); mu_v = glm_predict(Xval_v, b)
        phi_P_fit = tweedie_phi_pearson(pp_fit, mu_f, vp, k_par)
        phi_P_val = tweedie_phi_pearson(pp_val, mu_v, vp, 0)
        # primary: validation profile log-likelihood (phi profiled on 2016)
        phi_val, meth_val, ll_val_prof = tweedie_phi_ml(pp_val, mu_v, vp,
                                                        phi_P_val)
        # sensitivity: phi by ML on the fit years, then scored on 2016
        phi_fit, meth_fit, ll_fit = tweedie_phi_ml(pp_fit, mu_f, vp, phi_P_fit)
        ll_val_fitphi = tweedie_loglik(pp_val, mu_v, phi_fit, vp)
        # refit on the train years (as for the reported model) -> test Gini
        # (REPORTED ONLY as a sensitivity; never used for selection)
        inf_tr = {}
        b_tr_p = glm_irls(Xtr_s_v, pp_tr_s, family="tweedie", tw_p=vp, n_iter=50,
                          info=inf_tr)
        prof_rows.append({
            "p": vp,
            "loglik_valid_profile": ll_val_prof,
            "phi_valid_ml": phi_val, "phi_valid_method": meth_val,
            "phi_valid_pearson": phi_P_val,
            "loglik_valid_at_fit_phi": ll_val_fitphi,
            "phi_fit_ml": phi_fit, "phi_fit_method": meth_fit,
            "phi_fit_pearson": phi_P_fit,
            "loglik_fit_insample_profile": ll_fit,
            "valid_deviance_p_old_criterion": tweedie_dev(pp_val, mu_v, vp),
            "valid_gini_fit_model": gini(pp_val, mu_v),
            "valid_PO_fit_model": float(mu_v.sum() / pp_val.sum()),
            "irls_fit_iterations": inf_fit["iterations"],
            "irls_fit_converged": inf_fit["converged_1e-8"],
            "irls_fit_last_step": inf_fit["last_max_abs_step"],
            "test_gini_refit_train_REPORT_ONLY": gini(pp_te,
                                                      glm_predict(Xte_s_v, b_tr_p)),
            "irls_train_converged": inf_tr["converged_1e-8"],
        })
        log.info(f"  p={vp:.2f}: ℓ_val(profile)={ll_val_prof:,.1f} "
                 f"(φ̂={phi_val:,.1f}, {meth_val}); ℓ_val(φ̂_fit)="
                 f"{ll_val_fitphi:,.1f}; old criterion dev={prof_rows[-1]['valid_deviance_p_old_criterion']:,.0f}")
    prof = pd.DataFrame(prof_rows)
    i_best = int(prof["loglik_valid_profile"].idxmax())
    vp_best = float(prof.loc[i_best, "p"])
    ll_max = float(prof.loc[i_best, "loglik_valid_profile"])
    prof["delta_loglik_valid_profile"] = prof["loglik_valid_profile"] - ll_max
    # approximate 95% profile interval for p: {p: 2(ℓ_max - ℓ(p)) <= 3.84}
    in_ci = prof.loc[2 * (ll_max - prof["loglik_valid_profile"]) <= 3.841459, "p"]
    p_alt = {
        "validation_loglik_with_fit_year_ML_phi": float(
            prof.loc[prof["loglik_valid_at_fit_phi"].idxmax(), "p"]),
        "fit_year_insample_profile_loglik": float(
            prof.loc[prof["loglik_fit_insample_profile"].idxmax(), "p"]),
        "OLD_validation_deviance_NOT_comparable_across_p": float(
            prof.loc[prof["valid_deviance_p_old_criterion"].idxmin(), "p"]),
    }
    prof.to_csv(OUTPUTS_DIR / "14b_tweedie_profile.csv", index=False)
    outfiles.append(str((OUTPUTS_DIR / "14b_tweedie_profile.csv").resolve()))
    # R1b: the reported Tweedie GLM is the fit-year (2014-2015) model at the
    # selected p -- the same training window as the GBMs; the 2014-2016 refit
    # is kept only as the train-window sensitivity below.
    b_tw = fit_betas[vp_best]
    mu_tw = glm_predict(Xte_v, b_tw)
    mu_tw_val_fit = glm_predict(Xval_v, b_tw)
    phi_best = float(prof.loc[i_best, "phi_valid_ml"])
    # cross-check the total validation log-likelihood at the selected p with
    # statsmodels' Wright-Bessel implementation (finite terms only)
    sm_fam = try_import("statsmodels.genmod.families")
    sm_check = None
    if sm_fam is not None:
        ref = sm_fam.Tweedie(var_power=vp_best, eql=False).loglike_obs(
            pp_val, mu_tw_val_fit, scale=phi_best)
        own = tweedie_logpdf(pp_val, mu_tw_val_fit, phi_best, vp_best)
        fin = np.isfinite(ref)
        sm_check = {"n_finite": int(fin.sum()), "n_nonfinite": int((~fin).sum()),
                    "sum_own_finite": float(own[fin].sum()),
                    "sum_statsmodels_finite": float(ref[fin].sum()),
                    "max_abs_obs_diff": float(np.max(np.abs(own[fin] - ref[fin])))}
    tweedie_sel = {
        "criterion": R1_TWEEDIE_P_CRITERION,
        "criterion_definition": (
            "For each p in the grid: Tweedie GLM (log link, no offset, no "
            "weights, IRLS max 40 iterations) fitted on the fit years "
            f"{list(FIT_YEARS)}; dispersion phi profiled out by maximum "
            "likelihood on the validation year given the fit-year mean; "
            "criterion = exact Tweedie log-likelihood of the validation "
            f"year {list(VALID_YEARS)} (Dunn-Smyth series density, 1<p<2; "
            "y=0 term -mu^(2-p)/(phi(2-p))). Selected p maximises it; the "
            "reported Tweedie GLM is the fit-year model at that p (R1b: no "
            f"refit on {list(TRAIN_YEARS)}; that refit is a sensitivity)."),
        "grid": [float(v) for v in TWEEDIE_GRID],
        "selected_p": vp_best,
        "selected_on_grid_boundary": bool(vp_best in (float(TWEEDIE_GRID[0]),
                                                      float(TWEEDIE_GRID[-1]))),
        "loglik_valid_profile_max": ll_max,
        "phi_valid_ml_at_selected_p": phi_best,
        "phi_method_at_selected_p": str(prof.loc[i_best, "phi_valid_method"]),
        "phi_methods_used": sorted(set(prof["phi_valid_method"])
                                   | set(prof["phi_fit_method"])),
        "approx_95pct_profile_interval_p_grid_points": [float(v) for v in in_ci],
        "alternative_criteria_selected_p": p_alt,
        "pre_R1_selected_p": 1.9,
        "n_fit": int(len(pp_fit)), "n_valid": int(len(pp_val)),
        "n_valid_positive": int((pp_val > 0).sum()),
        "profile_table": prof.to_dict(orient="records"),
        "loglik_implementation_validation": ll_check,
        "statsmodels_crosscheck_at_selected_p": sm_check,
        "note": ("test_gini_refit_train_REPORT_ONLY is a sensitivity column "
                 f"(refit on {list(TRAIN_YEARS)}, covariates scaled by that "
                 "window's parameters); it is never used for selection."),
    }
    results.append({"model": f"Tweedie GLM (p={vp_best:g})", "tip": "pure-prem",
                    "test_deviance": tweedie_dev(pp_te, mu_tw, vp_best),
                    "pp_gini": gini(pp_te, mu_tw)})
    preds_for_calib["Tweedie GLM"] = mu_tw
    preds_valid["Tweedie GLM"] = glm_predict(Xval_v, b_tw)
    log.info(f"  Selected var_power (2016 profile log-likelihood)={vp_best:g} "
             f"(old deviance criterion {p_alt['OLD_validation_deviance_NOT_comparable_across_p']:g}); "
             f"~95% profile interval (grid) {list(in_ci)}; "
             f"test Gini={results[-1]['pp_gini']:.4f}")

    # ---- R1b: train-window sensitivity for the GLMs (former main design):
    # the four GLMs refitted on 2014-2016 (covariates scaled by that window's
    # own parameters), scored on the test years. Deterministic IRLS -> the
    # downstream seed stream is unchanged.
    log.info("\n[3b/6] GLM training-window sensitivity (2014-2016 refit)")
    preds_sens = {}
    bp_s = glm_irls(Xtr_s_v, y_tr_s, off_tr_s, family="poisson")
    preds_sens["Poisson GLM"] = glm_predict(Xte_s_v, bp_s, off_te) * sev_mean_s
    nb_alpha_s = estimate_nb_alpha(y_tr_s, glm_predict(Xtr_s_v, bp_s, off_tr_s))
    bnb_s = glm_irls(Xtr_s_v, y_tr_s, off_tr_s, family="negbin", nb_alpha=nb_alpha_s)
    preds_sens["NegBin GLM"] = glm_predict(Xte_s_v, bnb_s, off_te) * sev_mean_s
    cl_s = tr["n_claims"].values > 0
    sev_s = tr.loc[cl_s, "ultimate_rbns"].values / tr.loc[cl_s, "n_claims"].values
    w_s = tr.loc[cl_s, "n_claims"].values.astype(float); ok_s = sev_s > 0
    b_sev_s = glm_irls(Xtr_s_v[cl_s][ok_s], sev_s[ok_s], family="gamma", w=w_s[ok_s])
    preds_sens["Two-stage GLM"] = (glm_predict(Xte_s_v, bp_s, off_te)
                                   * glm_predict(Xte_s_v, b_sev_s))
    b_tw_s = glm_irls(Xtr_s_v, pp_tr_s, family="tweedie", tw_p=vp_best, n_iter=50)
    preds_sens["Tweedie GLM"] = glm_predict(Xte_s_v, b_tw_s)
    for k_s, v_s in preds_sens.items():
        log.info(f"  {k_s:14s} 2014-2016 refit: test Gini={gini(pp_te, v_s):.4f} "
                 f"(main model, 2014-2015: {gini(pp_te, preds_for_calib[k_s]):.4f})")

    log.info("\n[4/6] Tweedie GBM (numpy, early-stopping)")
    rng = np.random.default_rng(GLOBAL_SEED)
    Xfit_raw = Xfit.values.astype(float); Xval_raw = Xval.values.astype(float)
    Xte_raw = Xte.values.astype(float)
    # R2 (Reviewer 1, item 4): tree models on the same full fit sample as the GLMs.
    if TREE_TRAIN_CAP is None or TREE_TRAIN_CAP >= len(Xfit_raw):
        sub = np.arange(len(Xfit_raw))
        sample_txt = (f"all {len(Xfit_raw):,} fit-year policies (no subsampling; "
                      "the same sample as the GLMs)")
    else:
        sub = rng.choice(len(Xfit_raw), TREE_TRAIN_CAP, replace=False)
        sample_txt = (f"random subsample of {len(sub):,} of {len(Xfit_raw):,} "
                      "fit-year policies (seed GLOBAL_SEED)")
    log.info(f"  Tree-model training sample: {sample_txt}")
    Xb_fit, edges = hist_bin(Xfit_raw[sub])
    Xb_val = apply_bin(Xval_raw, edges); Xb_te = apply_bin(Xte_raw, edges)
    mu_gbm, mu_gbm_val, best_round = tweedie_gbm(
        Xb_fit, pp_fit[sub], Xb_val, pp_val, Xb_te, p=TWEEDIE_DEFAULT_P)
    results.append({"model": "Tweedie GBM (numpy)", "tip": "pure-prem",
                    "test_deviance": tweedie_dev(pp_te, mu_gbm,
                                                 TWEEDIE_DEFAULT_P),
                    "pp_gini": gini(pp_te, mu_gbm), "best_round": best_round})
    preds_for_calib["Tweedie GBM"] = mu_gbm
    preds_valid["Tweedie GBM"] = mu_gbm_val
    log.info(f"  Early-stop @ {best_round} trees, test Gini={results[-1]['pp_gini']:.3f}")

    if TWOSTAGE_GBM_ENABLED:
        log.info("\n[4a/6] Two-stage GBM (Poisson frequency × Gamma severity)")
        # The missing cell of the 2×2 design needed to separate the structure
        # (direct vs two-stage) and learner (GLM vs GBM) effects: GBM-based
        # two-stage. Frequency: all policies, Poisson GBM. Severity: mean claim
        # on policies with claims (weight = claim count), Gamma GBM.
        y_fit = fit["n_claims"].values.astype(float)
        mu_f_te, mu_f_val, br_f, _ = hist_gbm(
            Xb_fit, y_fit[sub], Xb_val, y_val, Xb_te, family="poisson")
        cl_fit = fit["n_claims"].values > 0
        cl_val = val["n_claims"].values > 0
        sev_fit = (fit.loc[cl_fit, "ultimate_rbns"].values
                   / fit.loc[cl_fit, "n_claims"].values)
        sev_val = (val.loc[cl_val, "ultimate_rbns"].values
                   / val.loc[cl_val, "n_claims"].values)
        w_fit = fit.loc[cl_fit, "n_claims"].values.astype(float)
        Xb_fit_cl = apply_bin(Xfit_raw[cl_fit], edges)
        Xb_val_cl = apply_bin(Xval_raw[cl_val], edges)
        sev_pos = sev_fit > 0
        # severity GBM: early-stop on valid policies with claims; extra prediction
        # = ALL valid (for the recalibration valid pure premium), test is already Xb_te.
        mu_s_te, _, br_s, mu_s_val_all = hist_gbm(
            Xb_fit_cl[sev_pos], sev_fit[sev_pos], Xb_val_cl,
            np.clip(sev_val, 1e-6, None), Xb_te, family="gamma",
            w=w_fit[sev_pos], Xb_extra=Xb_val)
        pp_2s = mu_f_te * mu_s_te
        pp_2s_val = mu_f_val * mu_s_val_all
        results.append({"model": "Two-stage GBM (Poi×Gam)",
                        "tip": "iki-aşamalı",
                        "test_deviance": tweedie_dev(pp_te, pp_2s,
                                                     TWEEDIE_DEFAULT_P),
                        "pp_gini": gini(pp_te, pp_2s),
                        "best_round": int(br_f)})
        preds_for_calib["Two-stage GBM"] = pp_2s
        preds_valid["Two-stage GBM"] = pp_2s_val
        log.info(f"  Frequency GBM @ {br_f} trees, Severity GBM @ {br_s} trees, "
                 f"test Gini={results[-1]['pp_gini']:.3f}")

    log.info("\n[4b/6] Optional packages (validation + early-stop, if available)")
    avail = []; pkg_info = {}
    lgb = try_import("lightgbm")
    if lgb is not None:
        try:
            dtr = lgb.Dataset(Xfit_raw[sub], label=pp_fit[sub])
            dval = lgb.Dataset(Xval_raw, label=pp_val, reference=dtr)
            # R1: single thread + deterministic=True + fixed seed
            lgb_params = {**LGB_PARAMS_BASE, "objective": "tweedie",
                          "tweedie_variance_power": TWEEDIE_DEFAULT_P,
                          "verbose": -1, **R1_LGB_DETERMINISM}
            m = lgb.train(lgb_params,
                          dtr, num_boost_round=PKG_NUM_BOOST_ROUND,
                          valid_sets=[dval],
                          callbacks=[lgb.early_stopping(PKG_EARLY_STOPPING,
                                                        verbose=False)])
            # Booster.predict uses best_iteration (early-stopped model)
            mu = np.clip(m.predict(Xte_raw, num_iteration=m.best_iteration),
                         1e-6, None)
            preds_valid["LightGBM"] = np.clip(
                m.predict(Xval_raw, num_iteration=m.best_iteration), 1e-6, None)
            results.append({"model": "LightGBM (Tweedie)", "tip": "opsiyonel",
                            "test_deviance": tweedie_dev(pp_te, mu, TWEEDIE_DEFAULT_P),
                            "pp_gini": gini(pp_te, mu), "best_round": m.best_iteration})
            preds_for_calib["LightGBM"] = mu; avail.append("lightgbm")
            _lp = {k: m.params.get(k) for k in sorted(m.params)}
            pkg_info["lightgbm"] = {
                "version": lgb.__version__, "params_passed": lgb_params,
                "params_resolved": _lp,
                "defaults_not_set (LightGBM 4.x documented defaults)": {
                    "lambda_l2": 0.0, "lambda_l1": 0.0, "max_depth": -1,
                    "bagging_fraction": 1.0, "bagging_freq": 0,
                    "feature_fraction": 1.0, "max_bin": 255,
                    "min_sum_hessian_in_leaf": 1e-3},
                "early_stopping_metric": "tweedie (neg. log-lik., p=1.5) on 2016",
                "n_trees_used": int(m.best_iteration),
                "n_trees_in_returned_booster": int(m.current_iteration())}
            log.info(f"  LightGBM: best_iter={m.best_iteration}, Gini={results[-1]['pp_gini']:.4f}")
        except Exception as e:
            log.info(f"  LightGBM skipped: {str(e)[:60]}")
    else:
        log.info("  LightGBM not installed — skipped (numpy GBM is sufficient)")
    xgb = try_import("xgboost")
    if xgb is not None:
        try:
            dtr = xgb.DMatrix(Xfit_raw[sub], label=pp_fit[sub])
            dval = xgb.DMatrix(Xval_raw, label=pp_val); dte = xgb.DMatrix(Xte_raw)
            # R1: single thread (nthread=1), fixed seed, explicit 'hist'
            xgb_params = {**XGB_PARAMS_BASE, "objective": "reg:tweedie",
                          "tweedie_variance_power": TWEEDIE_DEFAULT_P,
                          "verbosity": 0, **R1_XGB_DETERMINISM}
            m = xgb.train(xgb_params, dtr,
                          num_boost_round=PKG_NUM_BOOST_ROUND,
                          evals=[(dval, "valid")],
                          early_stopping_rounds=PKG_EARLY_STOPPING,
                          verbose_eval=False)
            # R1 FIX: xgb.train returns the LAST model (best_iteration + 20
            # rounds); Booster.predict uses all trees unless iteration_range
            # is given. Predict with the early-stopped model (0-based
            # best_iteration -> best_iteration + 1 trees), as the text says.
            n_best = int(m.best_iteration) + 1
            mu_full = np.clip(m.predict(dte), 1e-6, None)   # pre-R1 rule
            mu = np.clip(m.predict(dte, iteration_range=(0, n_best)), 1e-6, None)
            preds_valid["XGBoost"] = np.clip(
                m.predict(dval, iteration_range=(0, n_best)), 1e-6, None)
            results.append({"model": "XGBoost (Tweedie)", "tip": "opsiyonel",
                            "test_deviance": tweedie_dev(pp_te, mu, TWEEDIE_DEFAULT_P),
                            "pp_gini": gini(pp_te, mu), "best_round": m.best_iteration})
            preds_for_calib["XGBoost"] = mu; avail.append("xgboost")
            _cfg = json.loads(m.save_config())
            _tp = (_cfg.get("learner", {}).get("gradient_booster", {})
                   .get("tree_train_param", {}))
            _gb = _cfg.get("learner", {}).get("gradient_booster", {})
            pkg_info["xgboost"] = {
                "version": xgb.__version__, "params_passed": xgb_params,
                "tree_train_param_resolved": {k: _tp.get(k) for k in (
                    "eta", "max_depth", "min_child_weight", "lambda", "alpha",
                    "gamma", "subsample", "colsample_bytree",
                    "colsample_bylevel", "colsample_bynode", "max_bin",
                    "max_leaves", "grow_policy") if k in _tp},
                "updater_resolved": _gb.get("gbtree_train_param", {}).get(
                    "tree_method", None),
                "early_stopping_metric": "tweedie-nloglik@1.5 on 2016",
                "best_iteration_0based": int(m.best_iteration),
                "n_trees_used": n_best,
                "n_trees_trained": int(m.num_boosted_rounds()),
                "test_gini_full_model_pre_R1_prediction_rule": float(
                    gini(pp_te, mu_full))}
            log.info(f"  XGBoost: best_iter={m.best_iteration} ({n_best} trees), "
                     f"Gini={results[-1]['pp_gini']:.4f} (with all {m.num_boosted_rounds()} "
                     f"trees [old rule] {pkg_info['xgboost']['test_gini_full_model_pre_R1_prediction_rule']:.4f})")
        except Exception as e:
            log.info(f"  XGBoost skipped: {str(e)[:60]}")
    else:
        log.info("  XGBoost not installed — skipped (numpy GBM is sufficient)")

    def model_key(name):
        for k in ("Tweedie GLM", "Tweedie GBM", "Two-stage GBM",
                  "Two-stage GLM", "Poisson GLM", "NegBin GLM",
                  "LightGBM", "XGBoost"):
            if name.startswith(k):
                return k
        return name.split(" ")[0]

    log.info("\n[5/6] Calibration + recalibration (PAVA)")
    # select the best pure-premium model for calibration
    pp_models = [r for r in results
                 if r["tip"] in ("pure-prem", "opsiyonel", "iki-aşamalı")]
    # R1b: the carried-forward model is chosen on the VALIDATION year (2016)
    # Gini, never on the test years; test results are then only reported.
    for r in results:
        r["valid_gini"] = float(gini(pp_val, preds_valid[model_key(r["model"])]))
    best_model = max(pp_models, key=lambda r: r["valid_gini"])["model"]
    key = model_key(best_model)
    test_best_model = max(pp_models, key=lambda r: r["pp_gini"])["model"]
    log.info(f"  Model selection (2016 validation Gini): {best_model} "
             f"[valid Gini={max(r['valid_gini'] for r in pp_models):.4f}]; "
             f"test-year maximum: {test_best_model}")
    yhat = preds_for_calib.get(key, mu_tw)
    # Per-policy TEST predictions of the best pure-premium model (18_optimal_pricing
    # consumes them as the risk score): prediction, actual ultimate, exposure, premium.
    best_preds = pd.DataFrame({
        "policy_ref": te["policy_ref"].values,
        "policy_year": te["policy_year"].values,
        "exposure": te["exposure"].values.astype(float),
        "pp_pred": np.asarray(yhat, float),
        "pp_true": np.asarray(pp_te, float),
        "market_premium": pd.to_numeric(
            te.get("Kasko_Prim_TL", pd.Series(np.nan, index=te.index)),
            errors="coerce").values.astype(float),
    })
    best_preds.to_parquet(DATA_PROC_DIR / "pricing_best_preds.parquet", index=False)
    outfiles.append(str((DATA_PROC_DIR / "pricing_best_preds.parquet").resolve()))
    log.info(f"  Best-model per-policy test predictions saved "
             f"({len(best_preds):,} policies) → pricing_best_preds.parquet")
    calib = calibration_table(pp_te, yhat, n_dec=10)
    slope = calibration_slope(pp_te, yhat)
    overall_PO = float(yhat.sum() / max(pp_te.sum(), 1e-9))
    calib.to_csv(OUTPUTS_DIR / "14b_calibration.csv", index=False)
    outfiles.append(str((OUTPUTS_DIR / "14b_calibration.csv").resolve()))
    log.info(f"  Best model (calibration): {best_model}")
    log.info(f"  Calibration slope={slope:.3f} (1.0 ideal), overall P/O={overall_PO:.3f}")
    log.info(f"  Decile P/O range: [{calib['PO_ratio'].min():.2f}, {calib['PO_ratio'].max():.2f}]")

    # R1: three explicitly defined calibration slopes (see calibration_diagnostics)
    cal_before = calibration_diagnostics(pp_te, yhat)
    assert abs(cal_before["loglog_claimants"] - slope) < 1e-9
    log.info(f"  Slopes (1.0 ideal): log-log claimant-policy={cal_before['loglog_claimants']:.3f} "
             f"(n={cal_before['n_claimants_used']:,}); decile log-log="
             f"{cal_before['decile_loglog']:.3f}; policy-level linear="
             f"{cal_before['policy_linear']:.3f}")

    # ---- R2 (Reviewer 1, round 2, point 4b): level ("on-level") adjustment
    # diagnostics for the selected model. Neither factor uses any test-year
    # information:
    #  (a) validation-level factor  = realized / predicted pure premium on the
    #      validation year 2016 (a scalar; leaves the Gini unchanged);
    #  (b) frequency-trend factor   = log-linear trend of the claim frequency
    #      per policy-year over the fit + validation years (2014-2016),
    #      extrapolated from the exposure-weighted centre of the fit years to
    #      the centre of the test years (standard actuarial trending).
    lvl_val = (float(pp_val_true.sum() / max(preds_valid[key].sum(), 1e-9))
               if key in preds_valid else np.nan)
    pre_years = sorted(set(FIT_YEARS) | set(VALID_YEARS))
    freq_by_year = {int(y): float(ana.loc[ana["policy_year"] == y, "n_claims"].sum()
                                  / ana.loc[ana["policy_year"] == y, "exposure"].sum())
                    for y in pre_years}
    freq_test_report_only = {int(y): float(te.loc[te["policy_year"] == y, "n_claims"].sum()
                                           / te.loc[te["policy_year"] == y, "exposure"].sum())
                             for y in TEST_YEARS}
    _yrs = np.array(pre_years, float); _lf = np.log([freq_by_year[int(y)] for y in pre_years])
    b_trend, a_trend = np.polyfit(_yrs, _lf, 1)
    fit_center = float(np.average(fit["policy_year"].values.astype(float),
                                  weights=fit["exposure"].values.astype(float)))
    test_center = float(np.average(te["policy_year"].values.astype(float),
                                   weights=te["exposure"].values.astype(float)))
    lvl_trend = float(np.exp(b_trend * (test_center - fit_center)))
    #  (c) R2b: validation level, then trend from the validation-year centre to
    #      the test centre (standard on-level-to-latest-year + trend forward).
    valid_center = float(np.average(val["policy_year"].values.astype(float),
                                    weights=val["exposure"].values.astype(float)))
    lvl_val_trend = float(lvl_val * np.exp(b_trend * (test_center - valid_center)))
    level_factors = {"none": 1.0,
                     "validation_level": lvl_val,
                     "frequency_trend": lvl_trend,
                     "validation_level_then_trend": lvl_val_trend}
    level_rows = []; onlevel_tables = []
    for _nm, _fac in (("validation_level_factor", lvl_val),
                      ("frequency_trend_factor", lvl_trend),
                      ("validation_level_then_trend_factor", lvl_val_trend)):
        if np.isfinite(_fac):
            _y_adj = yhat * _fac
            _cal_adj = calibration_table(pp_te, _y_adj, n_dec=10)
            _cal_adj.insert(0, "adjustment", _nm)
            onlevel_tables.append(_cal_adj)
            level_rows.append({
                "adjustment": _nm, "factor": float(_fac),
                "overall_PO_before": float(overall_PO),
                "overall_PO_after": float(_y_adj.sum() / max(pp_te.sum(), 1e-9)),
                "decile_PO_min_before": float(calib["PO_ratio"].min()),
                "decile_PO_max_before": float(calib["PO_ratio"].max()),
                "decile_PO_min_after": float(_cal_adj["PO_ratio"].min()),
                "decile_PO_max_after": float(_cal_adj["PO_ratio"].max()),
                "mean_abs_decile_log_PO_before": float(np.abs(np.log(calib["PO_ratio"])).mean()),
                "mean_abs_decile_log_PO_after": float(np.abs(np.log(_cal_adj["PO_ratio"])).mean()),
                "gini_after": float(gini(pp_te, _y_adj))})
    level_df = pd.DataFrame(level_rows)
    level_df.to_csv(OUTPUTS_DIR / "14b_level_adjustment.csv", index=False)
    outfiles.append(str((OUTPUTS_DIR / "14b_level_adjustment.csv").resolve()))
    pd.concat(onlevel_tables, ignore_index=True).to_csv(
        OUTPUTS_DIR / "14b_calibration_onlevel.csv", index=False)
    outfiles.append(str((OUTPUTS_DIR / "14b_calibration_onlevel.csv").resolve()))
    # R2b: the on-levelled predictions are handed to 18_optimal_pricing next to
    # the raw ones (same parquet; the raw column pp_pred is unchanged).
    for _k, _f in level_factors.items():
        if _k != "none" and np.isfinite(_f):
            best_preds[f"pp_pred_onlevel_{_k}"] = np.asarray(yhat, float) * float(_f)
    best_preds.to_parquet(DATA_PROC_DIR / "pricing_best_preds.parquet", index=False)
    level_adj = {
        "model": best_model,
        "validation_level_factor": lvl_val,
        "frequency_trend": {"claim_frequency_per_policy_year": freq_by_year,
                            "slope_log_frequency_per_year": float(b_trend),
                            "fit_center_year_exposure_weighted": fit_center,
                            "test_center_year_exposure_weighted": test_center,
                            "factor": lvl_trend,
                            "test_year_frequency_REPORT_ONLY_not_used": freq_test_report_only},
        "validation_level_then_trend": {"valid_center_year_exposure_weighted": valid_center,
                                        "trend_factor_valid_to_test": float(np.exp(b_trend * (test_center - valid_center))),
                                        "factor": lvl_val_trend},
        "factors": level_factors,
        "rows": level_rows,
        "handed_to_18_as_columns": [f"pp_pred_onlevel_{k}" for k in level_factors if k != "none"],
        "config_PRICING_LEVEL_ADJUSTMENT": PRICING_LEVEL_ADJUSTMENT,
        "note": ("Scalar level factors estimated without test-year data; they "
                 "change the level of the predictions only, not their ranking "
                 "(Gini unchanged). 18_optimal_pricing multiplies the selected "
                 "model's predictions by the factor named in config "
                 "PRICING_LEVEL_ADJUSTMENT (R2b); the raw-level results are "
                 "kept in its manifest as a sensitivity.")}
    log.info("  R2 level (on-level) adjustment — without using any test information:")
    for _r in level_rows:
        log.info(f"    {_r['adjustment']:26s} factor={_r['factor']:.4f}: P/O "
                 f"{_r['overall_PO_before']:.3f} → {_r['overall_PO_after']:.3f}; "
                 f"decile P/O [{_r['decile_PO_min_after']:.2f}, {_r['decile_PO_max_after']:.2f}]; "
                 f"Gini {_r['gini_after']:.4f}")
    log.info(f"    frequency/policy-year {freq_by_year}; log-slope {b_trend:+.4f}/year; "
             f"centres fit {fit_center:.2f} / valid {valid_center:.2f} → test {test_center:.2f}; "
             f"factor handed to 18: {PRICING_LEVEL_ADJUSTMENT} = "
             f"{level_factors.get(PRICING_LEVEL_ADJUSTMENT, np.nan):.4f}")

    recal = None; yhat_rc = None; cal_after = None; calib_rc = None
    if RECALIBRATION_ENABLED and key in preds_valid:
        # PAVA isotonic recalibration: learn on VALID, apply to TEST.
        # The monotone transform preserves the ranking (and in practice the Gini).
        bx, fitted = pava_isotonic(preds_valid[key], pp_val_true)
        yhat_rc = pava_apply(bx, fitted, yhat)
        slope_rc = calibration_slope(pp_te, yhat_rc)
        PO_rc = float(yhat_rc.sum() / max(pp_te.sum(), 1e-9))
        gini_rc = gini(pp_te, yhat_rc)
        calib_rc = calibration_table(pp_te, yhat_rc, n_dec=10)
        calib_rc.to_csv(OUTPUTS_DIR / "14b_recalibration.csv", index=False)
        outfiles.append(str((OUTPUTS_DIR / "14b_recalibration.csv").resolve()))
        cal_after = calibration_diagnostics(pp_te, yhat_rc)
        recal = {"model": best_model, "slope_before": float(slope),
                 "slope_after": float(slope_rc),
                 "PO_before": float(overall_PO), "PO_after": PO_rc,
                 "gini_before": float(gini(pp_te, yhat)),
                 "gini_after": float(gini_rc), "n_bins": int(RECAL_N_BINS),
                 "slope_definition": ("slope_before/slope_after = "
                                      "loglog_claimants (see calibration."
                                      "slope_definitions)"),
                 "fitted_on": f"validation year {list(VALID_YEARS)} "
                              "predictions of the fit-year model",
                 "applied_to": f"test years {list(TEST_YEARS)}, unchanged",
                 "n_distinct_recalibrated_values_test": int(
                     len(np.unique(np.round(yhat_rc, 6)))),
                 "slopes_before": cal_before, "slopes_after": cal_after}
        log.info(f"  PAVA recalibration (fitted on valid → test): "
                 f"slope {slope:.3f} → {slope_rc:.3f}, "
                 f"P/O {overall_PO:.3f} → {PO_rc:.3f}, "
                 f"Gini {recal['gini_before']:.3f} → {gini_rc:.3f}")
        log.info(f"  After recalibration: decile log-log={cal_after['decile_loglog']:.3f}, "
                 f"policy-level linear={cal_after['policy_linear']:.3f}")
    elif RECALIBRATION_ENABLED:
        log.info(f"  Recalibration skipped: no valid prediction for '{key}'")

    log.info("\n[6/6] Paired Gini bootstrap + save + figures")
    # Gini of every model on the same bootstrap indices and the differences
    # relative to the best model. R1: the same resamples also give (i) paired CIs for the
    # four 2x2 factorial contrasts and (ii) CIs for the calibration slopes.
    # No extra random numbers are drawn (resample stream unchanged).
    def _gini_raw(y, pred):
        order = np.argsort(pred); ys = y[order]
        cum = np.cumsum(ys) / (ys.sum() + 1e-12)
        return 1 - 2 * np.sum(cum) / len(y)

    boot_keys = [(r["model"], model_key(r["model"])) for r in results
                 if model_key(r["model"]) in preds_for_calib]
    best_key_boot = key   # R1b: reference = validation-selected model
    rng_b = np.random.default_rng(GLOBAL_SEED)
    n_te = len(pp_te)
    ginis = {k: np.empty(PAIRED_GINI_B) for _, k in boot_keys}
    ginis_s = {k: np.empty(PAIRED_GINI_B) for k in preds_sens}
    cal_boot = {"before": np.empty((PAIRED_GINI_B, 3)),
                "after": np.empty((PAIRED_GINI_B, 3))}
    for b in range(PAIRED_GINI_B):
        idx = rng_b.integers(0, n_te, n_te)
        y_b = pp_te[idx]
        denom = _gini_raw(y_b, y_b) + 1e-12
        for _, k in boot_keys:
            ginis[k][b] = _gini_raw(y_b, preds_for_calib[k][idx]) / denom
        for k in preds_sens:
            ginis_s[k][b] = _gini_raw(y_b, preds_sens[k][idx]) / denom
        cal_boot["before"][b] = _calib_slopes_fast(y_b, yhat[idx])
        if yhat_rc is not None:
            cal_boot["after"][b] = _calib_slopes_fast(y_b, yhat_rc[idx])
    point_gini = {model_key(r["model"]): r["pp_gini"] for r in results}
    for r in results:
        k = model_key(r["model"])
        if k not in ginis:
            continue
        lo, hi = np.percentile(ginis[k], [2.5, 97.5])
        r["gini_lo"], r["gini_hi"] = float(lo), float(hi)
        if k != best_key_boot:
            d = ginis[k] - ginis[best_key_boot]
            dlo, dhi = np.percentile(d, [2.5, 97.5])
            r["delta_vs_best"] = float(d.mean())
            r["delta_point"] = float(point_gini[k] - point_gini[best_key_boot])
            r["delta_lo"], r["delta_hi"] = float(dlo), float(dhi)
            r["delta_significant"] = bool(not (dlo <= 0.0 <= dhi))
    log.info(f"  Paired bootstrap (B={PAIRED_GINI_B}) done; "
             f"reference: {best_key_boot}")
    for r in results:
        if "delta_vs_best" in r:
            log.info(f"    {r['model']:26s} ΔGini(point)={r['delta_point']:+.4f} "
                     f"(boot-mean {r['delta_vs_best']:+.4f}) "
                     f"[{r['delta_lo']:+.4f},{r['delta_hi']:+.4f}] "
                     f"{'DISTINGUISHABLE' if r['delta_significant'] else 'within uncertainty'}")

    # 2x2 factorial contrasts (paired, same resamples)
    fact_rows = []
    for cid, a, b_, desc in R1_FACTORIAL_CONTRASTS:
        if a not in ginis or b_ not in ginis:
            continue
        d = ginis[a] - ginis[b_]
        dlo, dhi = np.percentile(d, [2.5, 97.5])
        fact_rows.append({
            "contrast": cid, "definition": desc, "minuend": a, "subtrahend": b_,
            "gini_minuend": float(point_gini[a]),
            "gini_subtrahend": float(point_gini[b_]),
            "effect_point": float(point_gini[a] - point_gini[b_]),
            "effect_boot_mean": float(d.mean()),
            "ci_lo": float(dlo), "ci_hi": float(dhi),
            "significant_95": bool(not (dlo <= 0.0 <= dhi)),
            "B": int(PAIRED_GINI_B)})
    if all(k in ginis for k in ("Tweedie GLM", "Tweedie GBM",
                                "Two-stage GLM", "Two-stage GBM")):
        d = ((ginis["Two-stage GBM"] - ginis["Two-stage GLM"])
             - (ginis["Tweedie GBM"] - ginis["Tweedie GLM"]))
        dlo, dhi = np.percentile(d, [2.5, 97.5])
        pt = ((point_gini["Two-stage GBM"] - point_gini["Two-stage GLM"])
              - (point_gini["Tweedie GBM"] - point_gini["Tweedie GLM"]))
        fact_rows.append({
            "contrast": "interaction",
            "definition": ("Learner x structure interaction: (two-stage GBM - "
                           "two-stage GLM) - (Tweedie GBM - Tweedie GLM)"),
            "minuend": "learner_twostage", "subtrahend": "learner_direct",
            "gini_minuend": None, "gini_subtrahend": None,
            "effect_point": float(pt), "effect_boot_mean": float(d.mean()),
            "ci_lo": float(dlo), "ci_hi": float(dhi),
            "significant_95": bool(not (dlo <= 0.0 <= dhi)),
            "B": int(PAIRED_GINI_B)})
    # R1b: train-window sensitivity table (paired bootstrap, same resamples):
    # each GLM refitted on 2014-2016 minus the same GLM fitted on 2014-2015,
    # and the learner contrasts recomputed against the 2014-2016 GLMs.
    sens_rows = []
    for k in preds_sens:
        d = ginis_s[k] - ginis[k]; dlo, dhi = np.percentile(d, [2.5, 97.5])
        g_s = gini(pp_te, preds_sens[k])
        sens_rows.append({"quantity": f"{k}: refit {TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]} minus fit {FIT_YEARS[0]}-{FIT_YEARS[-1]}",
                          "model": k, "gini_fit_window": float(point_gini[k]),
                          "gini_train_window": float(g_s),
                          "effect_point": float(g_s - point_gini[k]),
                          "ci_lo": float(dlo), "ci_hi": float(dhi),
                          "significant_95": bool(not (dlo <= 0.0 <= dhi)), "B": int(PAIRED_GINI_B)})
    for cid, a, b_ in (("learner_direct_vs_GLM_2014_2016", "Tweedie GBM", "Tweedie GLM"),
                       ("learner_twostage_vs_GLM_2014_2016", "Two-stage GBM", "Two-stage GLM")):
        if a in ginis and b_ in preds_sens:
            d = ginis[a] - ginis_s[b_]; dlo, dhi = np.percentile(d, [2.5, 97.5])
            g_s = gini(pp_te, preds_sens[b_])
            sens_rows.append({"quantity": f"{a} (fit {FIT_YEARS[0]}-{FIT_YEARS[-1]}) minus {b_} (refit {TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]})",
                              "model": cid, "gini_fit_window": float(point_gini[a]),
                              "gini_train_window": float(g_s),
                              "effect_point": float(point_gini[a] - g_s),
                              "ci_lo": float(dlo), "ci_hi": float(dhi),
                              "significant_95": bool(not (dlo <= 0.0 <= dhi)), "B": int(PAIRED_GINI_B)})
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(OUTPUTS_DIR / "14b_train_window_sensitivity.csv", index=False)
    outfiles.append(str((OUTPUTS_DIR / "14b_train_window_sensitivity.csv").resolve()))
    for sr in sens_rows:
        log.info(f"    sensitivity {sr['quantity'][:70]:70s} {sr['effect_point']:+.4f} "
                 f"[{sr['ci_lo']:+.4f}, {sr['ci_hi']:+.4f}]")

    fact_df = pd.DataFrame(fact_rows)
    fact_df.to_csv(OUTPUTS_DIR / "14b_factorial_effects.csv", index=False)
    outfiles.append(str((OUTPUTS_DIR / "14b_factorial_effects.csv").resolve()))
    for fr in fact_rows:
        log.info(f"    2x2 {fr['contrast']:17s} {fr['effect_point']:+.4f} "
                 f"[{fr['ci_lo']:+.4f}, {fr['ci_hi']:+.4f}] "
                 f"{'significant' if fr['significant_95'] else 'not significant'}")

    # calibration slopes with bootstrap percentile CIs
    slope_names = ("loglog_claimants", "decile_loglog", "policy_linear")
    slope_rows = []
    for stage, cal_d in (("before_recalibration", cal_before),
                         ("after_isotonic_recalibration", cal_after)):
        if cal_d is None:
            continue
        arr = cal_boot["before" if stage.startswith("before") else "after"]
        for j, nm in enumerate(slope_names):
            lo, hi = np.nanpercentile(arr[:, j], [2.5, 97.5])
            slope_rows.append({"stage": stage, "slope": nm,
                               "estimate": float(cal_d[nm]),
                               "ci_lo": float(lo), "ci_hi": float(hi),
                               "B": int(PAIRED_GINI_B)})
    slope_df = pd.DataFrame(slope_rows)
    slope_df.to_csv(OUTPUTS_DIR / "14b_calibration_slopes.csv", index=False)
    outfiles.append(str((OUTPUTS_DIR / "14b_calibration_slopes.csv").resolve()))
    for sr in slope_rows:
        log.info(f"    slope {sr['stage']:29s} {sr['slope']:17s} {sr['estimate']:.3f} "
                 f"[{sr['ci_lo']:.3f}, {sr['ci_hi']:.3f}]")

    # comparable test deviance for every model (pure-premium scale, p = 1.5)
    for r in results:
        k = model_key(r["model"])
        r["test_tweedie_dev_p1.5"] = tweedie_dev(pp_te, preds_for_calib[k],
                                                 TWEEDIE_DEFAULT_P)
        r["test_deviance_metric"] = (
            "Poisson deviance of claim counts" if r["tip"] == "frekans"
            else (f"Tweedie deviance p={vp_best:g} (NOT comparable with p=1.5 rows)"
                  if k == "Tweedie GLM" else "Tweedie deviance p=1.5"))

    df = pd.DataFrame(results).sort_values("pp_gini", ascending=False)
    df.to_csv(OUTPUTS_DIR / "14b_pricing_models.csv", index=False)
    outfiles.append(str((OUTPUTS_DIR / "14b_pricing_models.csv").resolve()))
    for _, r in df.iterrows():
        log.info(f"  {r['model']:24s} Gini={r['pp_gini']:.4f}")

    # ---------------------------------------------------------------- figures
    apply_pub_style()
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    DISPLAY = {"Poisson GLM": "Poisson GLM",
               "NegBin GLM": "Negative binomial GLM",
               "Tweedie GLM": f"Tweedie GLM ($p$ = {vp_best:g})",
               "Tweedie GBM": "Tweedie GBM",
               "Two-stage GLM": "Two-stage GLM (Poisson × gamma)",
               "Two-stage GBM": "Two-stage GBM (Poisson × gamma)",
               "LightGBM": "LightGBM (Tweedie)",
               "XGBoost": "XGBoost (Tweedie)"}
    GROUPS = [("Frequency GLMs", "freq", ["Poisson GLM", "NegBin GLM"]),
              ("Direct pure premium", "direct", ["Tweedie GLM", "Tweedie GBM"]),
              ("Two-stage (frequency × severity)", "twostage",
               ["Two-stage GLM", "Two-stage GBM"]),
              ("Optional packages", "optional", ["LightGBM", "XGBoost"])]
    is_glm = {"Poisson GLM", "NegBin GLM", "Tweedie GLM", "Two-stage GLM"}
    by_key = {model_key(r["model"]): r for r in results}

    # Figure 7: Gini with paired-bootstrap 95% CI, every Table 6 model
    rows_fig = []
    for gname, gcode, keys in GROUPS:
        ks = [k for k in keys if k in by_key]
        if not ks:
            continue
        rows_fig.append(("header", gname, None, None))
        for k in ks:
            rows_fig.append(("model", DISPLAY[k], k, gcode))
    fig, ax = plt.subplots(figsize=(FIG_W_TEXT, 3.5))
    yt, ylab, yweight = [], [], []
    best_g = by_key[best_key_boot]["pp_gini"]
    ax.axvline(best_g, color="#7a7974", lw=0.8, ls=(0, (3, 2)), zorder=1)
    for i, (kind, lab, k, gcode) in enumerate(rows_fig):
        yt.append(i); ylab.append(lab if kind == "header" else "   " + lab)
        yweight.append("bold" if kind == "header" else "normal")
        if kind != "model":
            continue
        r = by_key[k]; g = r["pp_gini"]
        ax.errorbar(g, i, xerr=[[g - r["gini_lo"]], [r["gini_hi"] - g]],
                    fmt="o" if k in is_glm else "s",
                    color=R1_GROUP_COLORS[gcode], ms=4.2, mew=0.8,
                    mfc=R1_GROUP_COLORS[gcode] if k in is_glm else "white",
                    elinewidth=1.0, capsize=2.0, zorder=3)
        ax.annotate(f"{g:.3f} [{r['gini_lo']:.3f}, {r['gini_hi']:.3f}]",
                    xy=(1.02, i), xycoords=("axes fraction", "data"),
                    va="center", ha="left", fontsize=7)
    ax.annotate("Gini [95% CI]", xy=(1.02, -0.9),
                xycoords=("axes fraction", "data"), va="center", ha="left",
                fontsize=7, fontstyle="italic")
    ax.set_yticks(yt); ax.set_yticklabels(ylab)
    for tl, w in zip(ax.get_yticklabels(), yweight):
        tl.set_fontweight(w)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(rows_fig) - 0.4, -1.4)
    lo_all = min(by_key[k]["gini_lo"] for _, _, k, _ in rows_fig if k)
    hi_all = max(by_key[k]["gini_hi"] for _, _, k, _ in rows_fig if k)
    x_lo = np.floor((lo_all - 0.006) * 100) / 100
    ax.set_xlim(x_lo, hi_all + 0.004)
    ax.set_xticks(np.arange(x_lo, hi_all + 0.004 + 1e-9, 0.02))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Pure-premium Gini index, test years 2017–2018")
    from matplotlib.lines import Line2D
    leg_h = [Line2D([], [], marker="o", ls="", color="#52514e", ms=4.2,
                    label="GLM"),
             Line2D([], [], marker="s", ls="", color="#52514e", mfc="white",
                    ms=4.2, label="Gradient boosting"),
             Line2D([], [], ls=(0, (3, 2)), color="#7a7974", lw=0.8,
                    label="Best model")]
    ax.legend(handles=leg_h, loc="upper center", bbox_to_anchor=(0.5, -0.17),
              ncol=3, frameon=False, handletextpad=0.4, columnspacing=1.4)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "14b_pricing_comparison.png")
    fig.savefig(FIGURES_DIR / "14b_pricing_comparison.pdf"); plt.close(fig)
    outfiles.append(str((FIGURES_DIR / "14b_pricing_comparison.png").resolve()))
    outfiles.append(str((FIGURES_DIR / "14b_pricing_comparison.pdf").resolve()))

    # Figure 8: (a) decile actual vs predicted with 45-degree line; (b) lift
    best_disp = DISPLAY.get(key, best_model)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIG_W_FULL, 2.6),
                                 gridspec_kw={"width_ratios": [1.0, 1.3]})
    vals = np.r_[calib["pred_mean"], calib["actual_mean"]]
    if calib_rc is not None:
        vals = np.r_[vals, calib_rc["pred_mean"], calib_rc["actual_mean"]]
    lo_v, hi_v = float(vals.min() * 0.88), float(vals.max() * 1.10)
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.plot([lo_v, hi_v], [lo_v, hi_v], ls=(0, (3, 2)), color="#7a7974",
            lw=0.9, label="45° line", zorder=1)
    a1.plot(calib["pred_mean"], calib["actual_mean"], "o-", color="#2a78d6",
            ms=4, lw=1.0, label=best_disp.replace(" (Poisson × gamma)", ""),
            zorder=3)
    if calib_rc is not None:
        a1.plot(calib_rc["pred_mean"], calib_rc["actual_mean"], "s--",
                color="#eb6834", mfc="white", ms=3.6, lw=0.9,
                label="After isotonic recalibration", zorder=2)
    for d_, dx, dy, ha_ in ((1, 6, -1, "left"), (10, -7, 0, "right")):
        rr = calib.loc[calib["decile"] == d_].iloc[0]
        a1.annotate(f"decile {d_}", xy=(rr["pred_mean"], rr["actual_mean"]),
                    xytext=(dx, dy), textcoords="offset points", fontsize=7,
                    ha=ha_, va="center", color="#52514e")
    a1.set_xlim(lo_v, hi_v); a1.set_ylim(lo_v, hi_v)
    a1.set_aspect("equal", adjustable="box")
    from matplotlib.ticker import FixedLocator, NullLocator
    _tk = [t for t in (1000, 2000, 3000, 4000) if lo_v <= t <= hi_v]
    for axis_ in (a1.xaxis, a1.yaxis):
        axis_.set_major_locator(FixedLocator(_tk))
        axis_.set_minor_locator(NullLocator())
        axis_.set_major_formatter(thousands_formatter())
    a1.set_xlabel("Mean predicted pure premium\n(real TRY, log scale)")
    a1.set_ylabel("Mean actual pure premium\n(real TRY, log scale)")
    a1.set_title("(a) Actual vs predicted, by decile", loc="left")
    a1.legend(loc="upper left", frameon=False, handlelength=1.8)
    overall = pp_te.mean()
    lift = calib["actual_mean"] / (overall + 1e-9)
    a2.bar(calib["decile"], lift, color="#2a78d6", width=0.72, zorder=2)
    a2.axhline(1.0, color="#7a7974", ls=(0, (3, 2)), lw=0.9, zorder=3)
    a2.set_xticks(calib["decile"])
    a2.set_xlabel("Decile of predicted pure premium (1 = lowest)")
    a2.set_ylabel("Lift (decile mean / overall mean)")
    a2.set_title("(b) Lift by predicted decile", loc="left")
    a2.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIGURES_DIR / "14b_calibration.png")
    fig.savefig(FIGURES_DIR / "14b_calibration.pdf"); plt.close(fig)
    outfiles.append(str((FIGURES_DIR / "14b_calibration.png").resolve()))
    outfiles.append(str((FIGURES_DIR / "14b_calibration.pdf").resolve()))

    # ------------------------------------------------ model documentation (R1)
    n_sub = int(len(sub))
    gbm_common = {
        "learning_rate": GBM14B_LR, "max_depth": GBM14B_MAX_DEPTH,
        "n_bins": GBM14B_N_BINS,
        "binning": f"quantile bins computed on the {n_sub:,} fit-year policies used for training",
        "max_trees": GBM14B_MAX_TREES,
        "early_stopping_patience": GBM14B_PATIENCE,
        "l2_leaf_penalty_lambda": R1_TREE_L2_LAMBDA,
        "min_obs_to_split_a_node": R1_TREE_MIN_NODE_SPLIT,
        "min_obs_per_child": R1_TREE_MIN_CHILD,
        "leaf_values": "second-order (Newton) -G/(H+lambda), shrunk by learning rate",
        "row_subsampling_per_tree": "none (rate 1.0)",
        "column_subsampling": "none (rate 1.0)",
        "exposure_offset": "none",
        "tuned": "only the number of trees (early stopping); all else fixed a priori"}
    specs = {
        "Poisson GLM": {
            "learner": "GLM, log link, IRLS (max 40 iterations, ridge 1e-6)",
            "structure": "frequency only; pure premium = predicted count x "
                         "training mean severity",
            "fit_years": list(FIT_YEARS), "n_train": int(len(fit)),
            "validation_use": "none (no tuning parameter)",
            "offset": "log exposure", "weights": None,
            "training_mean_severity_real_TRY": float(sev_mean),
            "training_mean_severity_definition": (
                "mean over claimant policies of the fit years of "
                "ultimate/n_claims (policy-level average, unweighted)")},
        "NegBin GLM": {
            "learner": "GLM (NB2), log link, IRLS (max 40 iterations)",
            "structure": "frequency only; x training mean severity",
            "fit_years": list(FIT_YEARS), "n_train": int(len(fit)),
            "validation_use": "none", "offset": "log exposure",
            "nb_alpha": float(nb_alpha),
            "nb_alpha_estimation": ("moment estimator mean[((y-mu)^2-mu)/mu^2] "
                                    "from the Poisson GLM fit on the fit years")},
        "Two-stage GLM": {
            "learner": "GLM", "structure": "two-stage: Poisson frequency GLM x "
                                           "gamma severity GLM",
            "fit_years": list(FIT_YEARS), "n_train_frequency": int(len(fit)),
            "n_train_severity": int(sev_ok.sum()),
            "severity_model": ("gamma GLM, log link, on claimant policies' mean "
                               "severity (ultimate/n_claims > 0), weights = "
                               "claim count, no offset, IRLS max 40"),
            "frequency_model": "the Poisson GLM above (log-exposure offset)",
            "validation_use": "none"},
        "Tweedie GLM": {
            "learner": "GLM, log link, IRLS", "structure": "direct pure premium",
            "p_selection_fit_years": list(FIT_YEARS), "n_fit": int(len(fit)),
            "validation_use": ("variance power p selected by the 2016 profile "
                               "log-likelihood (see tweedie_p_selection)"),
            "fit_years": list(FIT_YEARS), "n_train": int(len(fit)),
            "tweedie_power": vp_best, "offset": None, "weights": None,
            "n_iter_max": 40},
        "Tweedie GBM": {
            "learner": "in-house histogram GBM", "structure": "direct pure premium",
            "loss": f"Tweedie deviance, fixed p = {TWEEDIE_DEFAULT_P}",
            "fit_years": list(FIT_YEARS), "n_train": n_sub,
            "sample": sample_txt,
            "validation_use": ("early stopping on 2016 Tweedie deviance "
                               f"(p = {TWEEDIE_DEFAULT_P}), all "
                               f"{len(val):,} policies"),
            "n_trees_selected": int(best_round), **gbm_common},
        "Two-stage GBM": {
            "learner": "in-house histogram GBM",
            "structure": "two-stage: Poisson frequency GBM x gamma severity GBM",
            "fit_years": list(FIT_YEARS),
            "frequency": {"loss": "Poisson deviance (claim counts, no exposure offset)",
                          "n_train": n_sub, "sample": sample_txt,
                          "validation_use": f"early stopping on 2016 Poisson deviance, all {len(val):,} policies",
                          "n_trees_selected": int(br_f)} if TWOSTAGE_GBM_ENABLED else None,
            "severity": {"loss": "gamma deviance, weights = claim count",
                         "n_train": int(sev_pos.sum()),
                         "sample": ("ALL 2014-2015 claimant policies with positive "
                                    "mean severity (not subsampled)"),
                         "validation_use": ("early stopping on 2016 claimant policies "
                                            "(unweighted gamma deviance), n = "
                                            f"{int(cl_val.sum()):,}"),
                         "n_trees_selected": int(br_s)} if TWOSTAGE_GBM_ENABLED else None,
            **gbm_common},
    }
    for pk, nm in (("lightgbm", "LightGBM"), ("xgboost", "XGBoost")):
        if pk in pkg_info:
            specs[nm] = {"learner": f"{nm} {pkg_info[pk]['version']}",
                         "structure": "direct pure premium",
                         "loss": f"Tweedie, fixed p = {TWEEDIE_DEFAULT_P}",
                         "fit_years": list(FIT_YEARS), "n_train": n_sub,
                         "sample": sample_txt,
                         "validation_use": (f"early stopping (patience "
                                            f"{PKG_EARLY_STOPPING}, max "
                                            f"{PKG_NUM_BOOST_ROUND} rounds) on "
                                            f"all {len(val):,} 2016 policies"),
                         "n_trees_selected": pkg_info[pk]["n_trees_used"],
                         "row_subsampling_per_tree": "none (rate 1.0)",
                         "column_subsampling": "none (rate 1.0)",
                         "exposure_offset": "none",
                         **pkg_info[pk]}
    spec_df = pd.DataFrame([{"model": k, "spec_json": json.dumps(v, ensure_ascii=False,
                                                                  default=str)}
                            for k, v in specs.items()])
    spec_df.to_csv(OUTPUTS_DIR / "14b_model_specs.csv", index=False)
    outfiles.append(str((OUTPUTS_DIR / "14b_model_specs.csv").resolve()))

    slope_defs = {
        "loglog_claimants": ("OLS slope of log(actual pure premium) on "
                             "log(predicted) over test policies with actual > 0 "
                             "(and predicted > 0), i.e. claimant policies only; "
                             "this is the pre-R1 'calibration slope' "
                             "(key calibration.slope)"),
        "decile_loglog": ("OLS slope of log(decile mean actual) on log(decile "
                          "mean predicted) across the 10 equal-count deciles of "
                          "predicted pure premium (all test policies)"),
        "policy_linear": ("OLS slope (with intercept) of actual pure premium "
                          "on predicted pure premium over ALL test policies, "
                          "zeros included"),
        "interpretation": ("1 = ideal for all three; <1 means realized losses "
                           "vary less than predictions across the predicted "
                           "range (predictions too spread out); >1 means "
                           "predictions are compressed"),
        "CIs": (f"bootstrap percentile 95% CIs over the same B = "
                f"{PAIRED_GINI_B} test-policy resamples as Table 6 "
                "(recalibration map held fixed)"),
    }

    manifest = {"script": SCRIPT_NAME, "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "split": {"fit": list(FIT_YEARS), "valid": list(VALID_YEARS), "test": list(TEST_YEARS),
                  "fit_window_for_all_main_models": list(FIT_YEARS),
                  "train_window_sensitivity_glm_refit": list(TRAIN_YEARS),
                  "n_fit": int(len(fit)), "n_valid": int(len(val)),
                  "n_test": int(len(te)), "n_train_sensitivity": int(len(tr))},
        "covariate_scaling": {
            "rule": ("numeric covariates standardised with the mean and SD of the "
                     f"fit years {list(FIT_YEARS)} only; the same parameters are applied "
                     "to the validation and test years (R1b; previously each split "
                     "used its own mean/SD)"),
            "parameters_fit_years": {c: {"mean": m, "sd": s} for c, (m, s) in stats_fit.items()},
            "sensitivity_window_parameters": {c: {"mean": m, "sd": s} for c, (m, s) in stats_tr.items()}},
        "models": results,
        "best_model": best_model, "best_gini": float(gini(pp_te, yhat)),
        "best_valid_gini": float(max(r["valid_gini"] for r in pp_models)),
        "test_best_model_REPORT_ONLY": test_best_model,
        "best_model_rule": ("model carried forward (calibration, recalibration, "
                            "pricing_best_preds.parquet -> 18_optimal_pricing, and "
                            "reference for the paired deltas) = highest VALIDATION-"
                            f"year {list(VALID_YEARS)} pure-premium Gini among the "
                            "direct pure-premium, two-stage and optional-package "
                            "models (R1b); test-year results are reported only"),
        "train_window_sensitivity": sens_rows,
        "tweedie_best_power": float(vp_best), "nb_alpha_estimated": nb_alpha,
        "tweedie_p_selection": tweedie_sel,
        "model_specs": specs,
        "gbm_fixed_tweedie_power": float(TWEEDIE_DEFAULT_P),
        "delta_conventions": {
            "delta_vs_best": "bootstrap MEAN of paired Gini differences (pre-R1 key)",
            "delta_point": "point difference of full-sample test Ginis",
            "delta_lo/delta_hi": "2.5/97.5 percentiles of the paired bootstrap differences",
            "B": int(PAIRED_GINI_B)},
        "factorial_2x2": fact_rows,
        "calibration": {"best_model": best_model, "slope": slope, "overall_PO": overall_PO,
                        "level_adjustment": level_adj,
                        "slope_definition": slope_defs["loglog_claimants"],
                        "slope_definitions": slope_defs,
                        "slopes": cal_before,
                        "slopes_ci": [r for r in slope_rows
                                      if r["stage"] == "before_recalibration"],
                        "decile_table": calib.to_dict(orient="records")},
        "recalibration": (None if recal is None else
                          {**recal, "slopes_ci": [r for r in slope_rows
                                                  if r["stage"].startswith("after")],
                           "decile_table": calib_rc.to_dict(orient="records")}),
        "paired_gini_B": int(PAIRED_GINI_B),
        "twostage_gbm_enabled": bool(TWOSTAGE_GBM_ENABLED),
        "optional_packages_used": avail,
        "optional_packages": pkg_info,
        "r1_design_choices": {
            "tweedie_p_criterion": R1_TWEEDIE_P_CRITERION,
            "tweedie_phi_method": R1_TWEEDIE_PHI_METHOD,
            "tweedie_series_tol_log": R1_TWEEDIE_SERIES_TOL,
            "tweedie_series_k0": R1_TWEEDIE_SERIES_K0,
            "phi_ml_search_log_halfwidth": R1_PHI_LOG_HALFWIDTH,
            "phi_ml_xatol_log": R1_PHI_XATOL,
            "tweedie_grid": [float(v) for v in TWEEDIE_GRID],
            "xgboost_determinism": R1_XGB_DETERMINISM,
            "lightgbm_determinism": R1_LGB_DETERMINISM,
            "xgboost_prediction": "early-stopped model (iteration_range=(0, best_iteration+1))",
            "lightgbm_prediction": "early-stopped model (num_iteration=best_iteration)",
            "factorial_contrasts": [c[:3] for c in R1_FACTORIAL_CONTRASTS],
            "calibration_deciles": R1_CALIB_N_DEC,
            "R1b_single_training_window": f"all main models fitted on {list(FIT_YEARS)}; {list(VALID_YEARS)} validation only",
            "R1b_model_selection": "validation-year Gini (test years never used for selection)",
            "R1b_covariate_scaling": "fit-year parameters applied to validation and test"},
        "note": "Tüm modeller numpy/scipy; statsmodels/lightgbm/xgboost OPSİYONEL (otomatik kurulum YOK).",
        "output_files": outfiles,
        "environment": env_info()}
    with open(OUTPUTS_DIR / "14b_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=_json_default)
    log.info("\n" + "=" * 78); log.info("SCRIPT 14b COMPLETED"); log.info("=" * 78)


if __name__ == "__main__":
    main()
