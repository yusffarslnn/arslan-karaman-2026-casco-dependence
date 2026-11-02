"""
================================================================================
SCRIPT 11 — Copula Modelling and Selection
================================================================================
PURPOSE
    Models the dependence with copula families on TWO SEPARATE axes:
      axis_FS  : FREQUENCY-SEVERITY (policy N × per-claim severity) — the PRIMARY
                 axis for pricing and compound CAPITAL; 13_capital consumes it.
      axis_lag : settlement-LAG × severity (settlement-lag) — for RESERVE risk;
                 the pkl backward-compatibility alias "axis_C" refers to this axis.
    Candidate families (Joe, Frank, Gumbel, Clayton, Survival-Clayton, Gaussian)
    are fitted via the one-parameter Kendall-τ mapping and selected by AIC; a
    bootstrap confidence interval is given for the parameter of the best family.

METHOD
    • PITs are read from Section 7: axis_A_pit (frequency-severity), axis_C_pit
      (settlement-lag). The same procedure is applied to each axis.
    • For each family θ is estimated from the observed Kendall τ by a closed-form/
      numerical mapping; the log-likelihood and AIC are computed.
    • Tail dependence coefficients (λ_L, λ_U) are reported.
    • [R1] CI for the selected family's θ: FULL-SIZE POLICY-CLUSTER bootstrap.
      Policies are drawn with replacement (all claims of a policy come together),
      PITs are recomputed in each replication with the SAME rule as 07 (rank + the
      same tie-jitter convention), the MPLE is refitted; percentile 95%.
      The old CI was taken, unscaled, from an m-out-of-n sub-sample of 5,000 pairs
      (≈√(n/m)≈6.5 times too wide); it is kept for diagnostic purposes only.
    • [R1] The Kendall τ implied by the selected θ (Gauss: (2/π)·arcsin θ; Joe:
      closed-form digamma) is reported with a CI from the same bootstrap replications.

NOTE
    Since τ≈0 on both axes, θ≈independence boundary (Joe θ≈1, Frank θ≈0);
    this quantifies that the copula is practically unnecessary on these data.

INPUT  : data/proc/{axis_A_pit,axis_C_pit,paid_long_real}.parquet
        (paid_long_real only for the policy-cluster identity; the claim table of
        07 is rebuilt exactly and its equality with the PITs is verified)
OUTPUT : data/proc/copula_fits.pkl ; outputs/11_copulas.csv ;
        outputs/11_manifest.json
================================================================================
"""
from __future__ import annotations
import sys
import json
import logging
import pickle
import time
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
from scipy import stats
from scipy.optimize import brentq
from scipy.special import digamma

from config import (
    DATA_PROC_DIR, OUTPUTS_DIR, LOGS_DIR, GLOBAL_SEED, GOF_BOOTSTRAP_CAP,
    PICKLE_PROTOCOL, COPULA_N_BOOTSTRAP, COPULA_TAU_CLAMP, COPULA_THETA_MAX,
    CLAYTON_THETA_MAX, COPULA_FIT_METHOD, PIT_JITTER_SCALE, env_info,
)

SCRIPT_NAME = "11_copulas"
# Bootstrap replications and τ→θ numerical bounds are read from config.py
N_BOOTSTRAP = COPULA_N_BOOTSTRAP

# --- R1 revision constants ---
# FULL-SIZE POLICY-CLUSTER bootstrap for the selected family's MPLE θ (primary CI).
CLUSTER_BOOT_B = COPULA_N_BOOTSTRAP     # 500 replications (would have been reduced
                                        # to 200 if the runtime exceeded 20 min; not
                                        # needed — the runtime is written to the manifest)
CLUSTER_BOOT_ALPHA = 0.05               # percentile 95% CI
CLUSTER_BOOT_STREAM = 1101              # SeedSequence stream tag (this bootstrap)
CLUSTER_BOOT_AXIS_TAG = {"axis_FS": 1, "axis_lag": 2}
CLUSTER_BOOT_LOG_EVERY = 100
BOOT_METHOD = ("policy_cluster_full_size_percentile: policies resampled with "
               "replacement (all claims of a drawn policy enter; resample size "
               "random), rank PITs recomputed per replicate with the 07 tie-"
               "jitter convention (sigma = PIT_JITTER_SCALE*std, one seed shared "
               "by both margins), MPLE refit of the selected family (tau-b start)")
# Old (R0) CI: iid m-out-of-n sub-sample of GOF_BOOTSTRAP_CAP=5,000 pairs,
# NO √(m/n) scaling → ≈6.5 times too wide. Reproduced for DIAGNOSTIC purposes only
# (same RNG stream → old numbers identical); it is NOT the report/table CI.
LEGACY_MOFN_DIAGNOSTIC = True
# Tie-jitter sensitivity (DIAGNOSTIC): MPLE of the selected family on the full
# sample with K different jitter seeds. The point estimate is a single jitter draw
# (07's GLOBAL_SEED); since the bootstrap uses fresh jitter in every replication,
# the CI also contains the jitter variability. This diagnostic gives the position
# of the reported θ̂ within the jitter distribution and the jitter-averaged θ
# (the point estimate is NOT CHANGED).
JITTER_SENS_K = 50
JITTER_SENS_STREAM = 1102
# Axis definitions: (PIT file, x column in the 07 claim table)
AXIS_SPECS = {"axis_FS": ("axis_A_pit", "n_claims"),
              "axis_lag": ("axis_C_pit", "delay")}


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


# --- θ → τ mappings (closed forms of Kendall τ) ---
def joe_tau(theta):
    """Kendall τ of the Joe copula: τ = 1 + 2/(2−θ)·[ψ(2) − ψ(2/θ + 1)],
    with limit 2 − π²/6 at θ=2 (Joe 1997; Nelsen 2006). Series equivalent:
    τ = 1 − 4 Σ_k 1/[k(θk+2)(θ(k−1)+2)]."""
    th = float(theta)
    if th <= 1.0:
        return 0.0
    if abs(th - 2.0) < 1e-6:
        return float(2.0 - np.pi**2 / 6.0)
    return float(1.0 + 2.0 / (2.0 - th) * (digamma(2.0) - digamma(2.0 / th + 1.0)))


def _debye1(t):
    from scipy.integrate import quad
    if abs(t) < 1e-8:
        return 1.0
    return quad(lambda x: x/(np.exp(x)-1), 0, t)[0]/t


def tau_from_theta(family, theta):
    """Kendall τ implied by the θ of a one-parameter family."""
    th = float(theta)
    if family == "Gauss":
        return float(2.0 / np.pi * np.arcsin(np.clip(th, -1.0, 1.0)))
    if family == "Joe":
        return joe_tau(th)
    if family == "Gumbel":
        return float(1.0 - 1.0 / th) if th > 1 else 0.0
    if family in ("Clayton", "Surv_Clayton"):
        return float(th / (th + 2.0)) if th > 0 else 0.0
    if family == "Frank":
        return float(1 + 4*(_debye1(th)-1)/th) if abs(th) > 1e-8 else 0.0
    return 0.0


# --- τ → θ mappings ---
def theta_from_tau(family, tau):
    tau = max(min(tau, COPULA_TAU_CLAMP), -COPULA_TAU_CLAMP)
    if family == "Frank":
        from scipy.integrate import quad
        def debye1(t):
            if abs(t) < 1e-8:
                return 1.0
            return quad(lambda x: x/(np.exp(x)-1), 0, t)[0]/t
        if abs(tau) < 1e-4:
            return 0.0
        # Frank θ has the same sign as τ (τ→θ is monotone and passes through the origin);
        # under negative dependence the bracket is moved to the negative side.
        g = lambda th: (1 + 4*(debye1(th)-1)/th) - tau
        try:
            return (brentq(g, 1e-6, COPULA_THETA_MAX) if tau > 0
                    else brentq(g, -COPULA_THETA_MAX, -1e-6))
        except ValueError:
            # R2 robustness (synthetic data, |tau| tiny): the bracket can lose its
            # sign change through cancellation in 1 + 4(D1(th)-1)/th at th -> 0;
            # use the small-tau expansion tau = th/9 + O(th^3). Never reached on
            # the reported run (no output changes).
            return 9.0 * tau
    if family == "Gumbel":
        # Gumbel allows positive dependence only; τ≤0 → independence boundary (θ=1)
        return 1.0 / (1.0 - tau) if tau > 0 else 1.0
    if family in ("Clayton", "Surv_Clayton"):
        # Clayton allows positive dependence only (θ>0); τ≤0 → independence (θ→0)
        return 2*tau/(1-tau) if 0 < tau < 1 else (CLAYTON_THETA_MAX
                                                  if tau >= 1 else 0.0)
    if family == "Joe":
        # [R1] Closed form of Joe τ(θ) (joe_tau). The old series carried a spurious
        # 1/θ factor (τ = 1 − 4 Σ 1/[k(2+kθ)(2+(k−1)θ)θ]) → it inflated τ by
        # ≈2.7 times at θ≈1.05; it affected only the MPLE STARTING value.
        if tau <= 1e-4:
            return 1.0
        try:
            return brentq(lambda th: joe_tau(th) - tau, 1.0001, COPULA_THETA_MAX / 2)
        except Exception:
            return 1.0 + 2*tau
    if family == "Gauss":
        return np.sin(np.pi * tau / 2)
    return 0.0


# --- log-likelihoods (copula density) ---
def copula_loglik(family, u, v, theta):
    u = np.clip(u, 1e-9, 1-1e-9); v = np.clip(v, 1e-9, 1-1e-9)
    try:
        if family == "Frank":
            if abs(theta) < 1e-6:
                return 0.0
            num = theta * (1 - np.exp(-theta)) * np.exp(-theta*(u+v))
            den = (1 - np.exp(-theta) - (1-np.exp(-theta*u))*(1-np.exp(-theta*v)))**2
            return float(np.sum(np.log(np.clip(num/den, 1e-12, None))))
        if family == "Gumbel":
            if theta <= 1:
                return 0.0
            lu, lv = -np.log(u), -np.log(v)
            A = (lu**theta + lv**theta)**(1/theta)
            c = (np.exp(-A) * (lu*lv)**(theta-1) * (A + theta - 1)
                 / (u*v*A**(2*theta-1)))
            return float(np.sum(np.log(np.clip(c, 1e-12, None))))
        if family == "Clayton":
            if theta <= 0:
                return 0.0
            c = ((1+theta)*(u*v)**(-1-theta)
                 * (u**(-theta)+v**(-theta)-1)**(-2-1/theta))
            return float(np.sum(np.log(np.clip(c, 1e-12, None))))
        if family == "Surv_Clayton":
            return copula_loglik("Clayton", 1-u, 1-v, theta)
        if family == "Joe":
            if theta <= 1:
                return 0.0
            a, b = (1-u)**theta, (1-v)**theta
            s = a + b - a*b
            c = (s**(1/theta-2) * (1-u)**(theta-1) * (1-v)**(theta-1)
                 * (theta - 1 + a + b - a*b))
            return float(np.sum(np.log(np.clip(c, 1e-12, None))))
        if family == "Gauss":
            from scipy.stats import norm
            if abs(theta) < 1e-6:
                return 0.0
            zu, zv = norm.ppf(u), norm.ppf(v)
            r = theta
            c = (1/np.sqrt(1-r**2)
                 * np.exp(-(r**2*(zu**2+zv**2)-2*r*zu*zv)/(2*(1-r**2))))
            return float(np.sum(np.log(np.clip(c, 1e-12, None))))
    except Exception:
        return -np.inf
    return 0.0


# θ domain per family (MPLE search bounds)
_THETA_BOUNDS = {"Frank": (-COPULA_THETA_MAX, COPULA_THETA_MAX),
                 "Gumbel": (1.0 + 1e-4, 50.0),
                 "Clayton": (1e-4, 50.0), "Surv_Clayton": (1e-4, 50.0),
                 "Joe": (1.0 + 1e-4, 30.0), "Gauss": (-0.99, 0.99)}


def fit_theta_mple(family, u, v, theta0):
    """Maximum pseudo-likelihood: bounded one-dimensional likelihood
    maximisation around the τ-inversion start. It is a 'pseudo'-likelihood
    because it operates on rank-PITs (Genest et al. 1995)."""
    from scipy.optimize import minimize_scalar
    lo, hi = _THETA_BOUNDS[family]
    # local window containing the start + global bounds
    res = minimize_scalar(lambda th: -copula_loglik(family, u, v, th),
                          bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-5})
    th = float(res.x)
    # degenerate-solution guard: if the MPLE is worse than τ-inversion, keep the inversion
    if copula_loglik(family, u, v, th) < copula_loglik(family, u, v, theta0):
        th = float(theta0)
    return th


def tail_dependence(family, theta):
    """Lower/upper tail dependence (λ_L, λ_U)."""
    if family == "Gumbel":
        return 0.0, 2 - 2**(1/theta) if theta > 1 else 0.0
    if family == "Joe":
        return 0.0, 2 - 2**(1/theta) if theta > 1 else 0.0
    if family == "Clayton":
        return 2**(-1/theta) if theta > 0 else 0.0, 0.0
    if family == "Surv_Clayton":
        return 0.0, 2**(-1/theta) if theta > 0 else 0.0
    return 0.0, 0.0


# ------------------------------------------------------------------------------
# [R1] Policy-cluster bootstrap infrastructure
# ------------------------------------------------------------------------------
def build_claim_table() -> pd.DataFrame:
    """Rebuilds the claim table of 07_conditional EXACTLY (same groupby, same
    severity>0 filter, same order). Sole purpose: attach the policy-cluster
    identity (policy_ref) to the PIT rows. Nothing at row level is logged."""
    paid = load_proc("paid_long_real")
    claim = paid.groupby("claim_file").agg(
        policy_ref=("policy_ref", "first"),
        accident_year=("accident_year", "first"),
        severity=("paid_total_real", "sum"),
        delay=("dev_year", "mean")).reset_index()
    claim = claim[claim["severity"] > 0].reset_index(drop=True)
    freq = (paid.groupby("policy_ref")["claim_file"].nunique()
            .rename("n_claims").reset_index())
    claim = claim.merge(freq, on="policy_ref", how="left")
    return claim.dropna(subset=["n_claims"]).reset_index(drop=True)


def to_pit_jitter(v: np.ndarray, seed: int) -> np.ndarray:
    """The SAME rule as 07_conditional.to_pit(v, jitter=True, seed): normal
    jitter with σ = (std(v)+1e-9)·PIT_JITTER_SCALE, rank/(n+1). 07 calls both
    margins with the SAME seed; the bootstrap does the same."""
    rng = np.random.default_rng(seed)
    v = v + rng.normal(0, (np.std(v) + 1e-9) * PIT_JITTER_SCALE, len(v))
    return stats.rankdata(v) / (len(v) + 1)


def verify_pit_alignment(pit: pd.DataFrame, x: np.ndarray, y: np.ndarray,
                         label: str) -> dict:
    """Verifies that the rebuilt raw (x, y) values and the PITs are exactly
    equal to the 07 output; otherwise the cluster-identity alignment/PIT rule
    is unreliable → explicit error."""
    raw_ok = (len(pit) == len(x)
              and np.array_equal(pit["x_raw"].values.astype(float), x)
              and np.array_equal(pit["y_raw"].values.astype(float), y))
    if not raw_ok:
        raise RuntimeError(
            f"[{label}] the rebuilt claim table is not aligned with the PIT file of "
            "module 07 (x_raw/y_raw differ). Re-run modules 07 and 11 on the "
            "same data.")
    u_ok = np.array_equal(to_pit_jitter(x, GLOBAL_SEED), pit["u"].values)
    v_ok = np.array_equal(to_pit_jitter(y, GLOBAL_SEED), pit["v"].values)
    if not (u_ok and v_ok):
        raise RuntimeError(
            f"[{label}] to_pit_jitter does not reproduce the PITs of module 07 exactly "
            "(the tie-jitter rule may have changed); align 11.to_pit_jitter with "
            "07.to_pit.")
    return {"raw_values_identical": True, "pit_reproduced_exactly": True}


def cluster_index(policy_ids: np.ndarray):
    """Groups rows by policy: (order, starts, counts, G)."""
    codes, uniq = pd.factorize(policy_ids)
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    starts = np.searchsorted(sorted_codes, np.arange(len(uniq)))
    ends = np.append(starts[1:], len(sorted_codes))
    return order, starts, ends - starts, len(uniq)


def cluster_bootstrap_theta(fam, x, y, order, starts, counts, G, B, axis_tag,
                            label):
    """FULL-SIZE policy-cluster bootstrap: G policies are drawn with replacement;
    in each replication the PITs are recomputed with the 07 rule and the MPLE
    of the selected family is refitted (as in the main fit, with a τ-inversion
    start from the raw τ-b). Each replication is fed by its own SeedSequence
    child (deterministic)."""
    ss = np.random.SeedSequence([GLOBAL_SEED, CLUSTER_BOOT_STREAM, axis_tag])
    children = ss.spawn(B)
    thetas = np.full(B, np.nan)
    sizes = np.empty(B, dtype=np.int64)
    t_start = time.time()
    for b, child in enumerate(children):
        rng = np.random.default_rng(child)
        pol = rng.integers(0, G, G)
        lens = counts[pol]
        tot = int(lens.sum())
        first = np.repeat(starts[pol], lens)
        within = np.arange(tot) - np.repeat(np.cumsum(lens) - lens, lens)
        idx = order[first + within]
        xb, yb = x[idx], y[idx]
        jseed = int(rng.integers(0, 2**32))      # seed SHARED by both margins (07)
        ub, vb = to_pit_jitter(xb, jseed), to_pit_jitter(yb, jseed)
        t0 = theta_from_tau(fam, stats.kendalltau(xb, yb)[0])
        try:
            thetas[b] = (fit_theta_mple(fam, ub, vb, t0)
                         if COPULA_FIT_METHOD == "mple" else t0)
        except Exception:
            thetas[b] = np.nan
        sizes[b] = tot
        if (b + 1) % CLUSTER_BOOT_LOG_EVERY == 0:
            log.info(f"    cluster-bootstrap {label}: {b+1}/{B} "
                     f"({time.time()-t_start:.0f} s)")
    return thetas, sizes, time.time() - t_start


def legacy_mofn_ci(fam, u, v):
    """Exact reproduction of the OLD (R0) CI — for DIAGNOSTIC purposes only:
    iid sub-sample of GOF_BOOTSTRAP_CAP pairs, NO √(m/n) scaling."""
    rng = np.random.default_rng(GLOBAL_SEED)
    n = len(u); boot_n = min(n, GOF_BOOTSTRAP_CAP)
    boot_theta = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, boot_n)
        t0 = theta_from_tau(fam, stats.kendalltau(u[idx], v[idx])[0])
        boot_theta.append(fit_theta_mple(fam, u[idx], v[idx], t0)
                          if COPULA_FIT_METHOD == "mple" else t0)
    lo, hi = np.percentile(boot_theta, [2.5, 97.5])
    return float(lo), float(hi), int(boot_n)


def jitter_sensitivity(fam, x, y, theta_hat, axis_tag):
    """MPLE on the full sample with K jitter seeds: (i) the 07 convention (seed
    shared by both margins), (ii) independent seed per margin."""
    ss = np.random.SeedSequence([GLOBAL_SEED, JITTER_SENS_STREAM, axis_tag])
    seeds = [int(z) for z in ss.generate_state(2 * JITTER_SENS_K)]
    t0 = theta_from_tau(fam, stats.kendalltau(x, y)[0])
    fit = (lambda uu, vv: fit_theta_mple(fam, uu, vv, t0)
           if COPULA_FIT_METHOD == "mple" else t0)
    common, indep = [], []
    for k in range(JITTER_SENS_K):
        s1, s2 = seeds[k], seeds[JITTER_SENS_K + k]
        ux = to_pit_jitter(x, s1)
        common.append(fit(ux, to_pit_jitter(y, s1)))
        indep.append(fit(ux, to_pit_jitter(y, s2)))

    def summ(a):
        a = np.asarray(a)
        return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)),
                "min": float(a.min()), "max": float(a.max()),
                "q025": float(np.percentile(a, 2.5)),
                "q975": float(np.percentile(a, 97.5)),
                "tau_implied_of_mean": tau_from_theta(fam, float(a.mean()))}
    common = np.asarray(common)
    return {"K": JITTER_SENS_K, "family": fam,
            "convention_07_common_seed": summ(common),
            "independent_seeds_per_margin": summ(indep),
            "share_below_reported_theta": float(np.mean(common < theta_hat)),
            "seed_scheme": (f"SeedSequence([GLOBAL_SEED, {JITTER_SENS_STREAM}, "
                            f"{axis_tag}]).generate_state({2*JITTER_SENS_K})")}


def fit_axis(axis_key, claim, label):
    """Reads one axis's PIT, fits all families, selects the best by AIC, and
    computes the FULL-SIZE policy-cluster bootstrap CI for the best family's θ.
    Returns: (bundle, df_rows, best, results, ci, tau_obs, boot_info)."""
    pit_stem, xcol = AXIS_SPECS[axis_key]
    pit = load_proc(pit_stem)
    u, v = pit["u"].values, pit["v"].values
    x_raw = claim[xcol].values.astype(float)
    y_raw = claim["severity"].values.astype(float)
    align = verify_pit_alignment(pit, x_raw, y_raw, label)
    tau_obs = stats.kendalltau(pit["x_raw"].values, pit["y_raw"].values)[0]
    log.info(f"\n[{label}] PIT: n={len(u):,}, observed τ={tau_obs:+.4f} "
             f"(PIT reproduced exactly from 07)")
    families = ["Joe", "Frank", "Gumbel", "Clayton", "Surv_Clayton", "Gauss"]
    n_obs = len(u)
    results = {}
    # Independence copula as an EXPLICIT candidate: c(u,v)=1 ⇒ ll=0, k=0
    results["Independence"] = {"theta": 0.0, "loglik": 0.0, "aic": 0.0,
                               "bic": 0.0, "lambda_L": 0.0, "lambda_U": 0.0,
                               "fit": "exact", "tau_implied": 0.0}
    log.info(f"    {'Independence':14s} θ=+0.0000  AIC=0.0  BIC=0.0")
    for fam in families:
        theta0 = theta_from_tau(fam, tau_obs)
        theta = (fit_theta_mple(fam, u, v, theta0)
                 if COPULA_FIT_METHOD == "mple" else theta0)
        ll = copula_loglik(fam, u, v, theta)
        aic = 2*1 - 2*ll
        bic = float(np.log(n_obs) - 2*ll)          # k=1 parameter
        lL, lU = tail_dependence(fam, theta)
        results[fam] = {"theta": float(theta), "theta_tau_inv": float(theta0),
                        "loglik": float(ll), "aic": float(aic), "bic": bic,
                        "lambda_L": float(lL), "lambda_U": float(lU),
                        "fit": COPULA_FIT_METHOD,
                        "tau_implied": tau_from_theta(fam, theta)}
        log.info(f"    {fam:14s} θ={theta:+.4f} (τ-inv {theta0:+.4f})  "
                 f"AIC={aic:.1f}  BIC={bic:.1f}  λ_L={lL:.3f} λ_U={lU:.3f}  "
                 f"τ(θ)={results[fam]['tau_implied']:+.4f}")
    best = min(results, key=lambda f: results[f]["aic"])
    best_bic = min(results, key=lambda f: results[f]["bic"])
    log.info(f"    → AIC selection: {best}; BIC selection: {best_bic}")

    # --- [R1] primary CI: full-size policy-cluster bootstrap -----------------
    fam_for_ci = best if best != "Independence" else "Gauss"
    theta_hat = results[fam_for_ci]["theta"]
    order, starts, counts, G = cluster_index(claim["policy_ref"].values)
    log.info(f"  Cluster-bootstrap ({fam_for_ci}): B={CLUSTER_BOOT_B}, "
             f"G={G:,} policy clusters, n={n_obs:,} claims")
    thetas, sizes, secs = cluster_bootstrap_theta(
        fam_for_ci, x_raw, y_raw, order, starts, counts, G, CLUSTER_BOOT_B,
        CLUSTER_BOOT_AXIS_TAG[axis_key], label)
    ok = np.isfinite(thetas)
    q = [100 * CLUSTER_BOOT_ALPHA / 2, 100 * (1 - CLUSTER_BOOT_ALPHA / 2)]
    lo, hi = (float(z) for z in np.percentile(thetas[ok], q))
    taus_b = np.array([tau_from_theta(fam_for_ci, t) for t in thetas[ok]])
    tlo, thi = (float(z) for z in np.percentile(taus_b, q))
    tau_hat = tau_from_theta(fam_for_ci, theta_hat)
    indep_theta = 0.0 if fam_for_ci in ("Gauss", "Frank") else (
        1.0 if fam_for_ci in ("Joe", "Gumbel") else 0.0)
    share_indep_side = float(np.mean(
        thetas[ok] >= indep_theta if theta_hat < indep_theta
        else thetas[ok] <= indep_theta))
    log.info(f"  → {label} best (AIC): {fam_for_ci} θ={theta_hat:+.4f} "
             f"cluster-CI[{lo:+.4f}, {hi:+.4f}] (SE={np.std(thetas[ok], ddof=1):.4f}); "
             f"implied τ={tau_hat:+.4f} CI[{tlo:+.4f}, {thi:+.4f}]; "
             f"share on the independence side={share_indep_side:.3f}; "
             f"valid {int(ok.sum())}/{CLUSTER_BOOT_B}; runtime {secs/60:.1f} min")
    boot_info = {
        "boot_method": BOOT_METHOD,
        "boot_family": fam_for_ci,
        "boot_B": int(CLUSTER_BOOT_B),
        "boot_B_valid": int(ok.sum()),
        "boot_ci_level": 1 - CLUSTER_BOOT_ALPHA,
        "boot_ci_cluster": [lo, hi],
        "boot_se_cluster": float(np.std(thetas[ok], ddof=1)),
        "boot_mean_cluster": float(np.mean(thetas[ok])),
        "boot_share_independence_side": share_indep_side,
        "boot_independence_value": indep_theta,
        "tau_implied": float(tau_hat),
        "tau_implied_ci_cluster": [tlo, thi],
        "tau_implied_map": ("(2/pi)*arcsin(theta)" if fam_for_ci == "Gauss" else
                            "1+2/(2-theta)*[digamma(2)-digamma(2/theta+1)]"
                            if fam_for_ci == "Joe" else "closed-form tau(theta)"),
        "boot_n_clusters": int(G),
        "boot_resample_claims": {"mean": float(sizes.mean()),
                                 "min": int(sizes.min()),
                                 "max": int(sizes.max())},
        "boot_runtime_sec": float(secs),
        "seed_scheme": (f"SeedSequence([GLOBAL_SEED={GLOBAL_SEED}, "
                        f"{CLUSTER_BOOT_STREAM}, {CLUSTER_BOOT_AXIS_TAG[axis_key]}])"
                        ".spawn(B); per replicate: policy draw, then one jitter "
                        "seed shared by both margins"),
        "pit_alignment_check": align,
    }
    # --- tie-jitter sensitivity (DIAGNOSTIC) --------------------------------
    js = jitter_sensitivity(fam_for_ci, x_raw, y_raw, theta_hat,
                            CLUSTER_BOOT_AXIS_TAG[axis_key])
    boot_info["jitter_sensitivity"] = js
    boot_info["theta_jitter_mean"] = js["convention_07_common_seed"]["mean"]
    boot_info["theta_jitter_sd"] = js["convention_07_common_seed"]["sd"]
    jc, ji = js["convention_07_common_seed"], js["independent_seeds_per_margin"]
    log.info(f"    [diagnostic] jitter sensitivity (K={js['K']}): 07-convention "
             f"θ mean={jc['mean']:+.4f} sd={jc['sd']:.4f} "
             f"[{jc['min']:+.4f}, {jc['max']:+.4f}]; independent-seed mean="
             f"{ji['mean']:+.4f}; share below the reported θ̂="
             f"{js['share_below_reported_theta']:.2f}")
    # --- old m-out-of-n CI: DIAGNOSTIC only (unscaled + √(m/n)-scaled) ------
    if LEGACY_MOFN_DIAGNOSTIC:
        llo, lhi, m = legacy_mofn_ci(fam_for_ci, u, v)
        r = np.sqrt(m / n_obs)
        boot_info["theta_ci_mofn_legacy"] = [llo, lhi]
        boot_info["theta_ci_mofn_rescaled_iid"] = [
            float(theta_hat + r * (llo - theta_hat)),
            float(theta_hat + r * (lhi - theta_hat))]
        boot_info["mofn_m"] = int(m)
        log.info(f"    [diagnostic] old m-out-of-n CI (m={m:,}, unscaled) "
                 f"[{llo:+.4f}, {lhi:+.4f}]; √(m/n)-scaled iid "
                 f"[{boot_info['theta_ci_mofn_rescaled_iid'][0]:+.4f}, "
                 f"{boot_info['theta_ci_mofn_rescaled_iid'][1]:+.4f}]")

    bundle = {"best": {"family": best, "params": [results[best]["theta"]]},
              "results": results, "tau_obs": float(tau_obs),
              # [R1] 'theta_ci' now carries the primary (cluster) CI; the old
              # m-out-of-n value is under 'theta_ci_mofn_legacy' (diagnostic).
              "theta_ci": [lo, hi],
              "best_joe_params": (results["Joe"]["theta"],)}
    bundle.update(boot_info)
    bundle["best_bic_family"] = best_bic
    rows = []
    for f in families:
        sel = (f == fam_for_ci)
        rows.append({"axis": label, "family": f, **results[f],
                     "selected_aic": f == best,
                     "ci_lo_cluster": lo if sel else np.nan,
                     "ci_hi_cluster": hi if sel else np.nan,
                     "tau_implied_ci_lo_cluster": tlo if sel else np.nan,
                     "tau_implied_ci_hi_cluster": thi if sel else np.nan,
                     "boot_B": int(ok.sum()) if sel else np.nan,
                     "boot_method": ("policy_cluster_full_size_percentile"
                                     if sel else ""),
                     "theta_jitter_mean": (boot_info["theta_jitter_mean"]
                                           if sel else np.nan),
                     "theta_jitter_sd": (boot_info["theta_jitter_sd"]
                                         if sel else np.nan),
                     "ci_lo_mofn_legacy": (boot_info.get("theta_ci_mofn_legacy",
                                                         [np.nan])[0]
                                           if sel else np.nan),
                     "ci_hi_mofn_legacy": (boot_info.get("theta_ci_mofn_legacy",
                                                         [np.nan, np.nan])[1]
                                           if sel else np.nan)})
    return bundle, rows, best, results, (lo, hi), float(tau_obs), boot_info


def main() -> None:
    global log
    log = setup_logger()
    log.info("=" * 78)
    log.info(f"SCRIPT {SCRIPT_NAME} — Copula Modelling (frequency-severity + settlement-lag)")
    log.info("=" * 78)
    t_all = time.time()

    # The 07 claim table is rebuilt (once) for the policy-cluster identity
    claim = build_claim_table()

    # PRIMARY axis: frequency-severity (axis_A_pit = policy N × per-claim severity)
    fs_bundle, fs_rows, fs_best, fs_res, fs_ci, fs_tau, fs_boot = fit_axis(
        "axis_FS", claim, "axis_FS (frekans-şiddet)")
    # Secondary axis: settlement-lag (axis_C_pit = settlement lag × severity)
    lag_bundle, lag_rows, lag_best, lag_res, lag_ci, lag_tau, lag_boot = fit_axis(
        "axis_lag", claim, "axis_lag (settlement-lag)")
    del claim

    bundle = {
        "axis_FS": fs_bundle,       # frequency-severity — 13_capital reads this
        "axis_lag": lag_bundle,     # settlement-lag — reserve risk
        "axis_C": lag_bundle,       # backward-compatibility alias (axis_C)
    }
    with open(DATA_PROC_DIR / "copula_fits.pkl", "wb") as f:
        pickle.dump(bundle, f, protocol=PICKLE_PROTOCOL)

    df = pd.DataFrame(fs_rows + lag_rows)
    df.to_csv(OUTPUTS_DIR / "11_copulas.csv", index=False)

    runtime = time.time() - t_all
    manifest = {
        "script": SCRIPT_NAME,
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "axis_FS": {"tau_obs": fs_tau, "best_family": fs_best,
                    "best_theta": fs_res[fs_best]["theta"],
                    "theta_ci": [float(fs_ci[0]), float(fs_ci[1])],
                    **fs_boot,
                    "all_families": fs_res},
        "axis_lag": {"tau_obs": lag_tau, "best_family": lag_best,
                     "best_theta": lag_res[lag_best]["theta"],
                     "theta_ci": [float(lag_ci[0]), float(lag_ci[1])],
                     **lag_boot,
                     "all_families": lag_res},
        # backward compatibility (so that 20_synthesis can read the old keys)
        "tau_obs": lag_tau, "best_family": lag_best,
        "best_theta": lag_res[lag_best]["theta"],
        "design_choices": {
            "fit_method": COPULA_FIT_METHOD,
            "selection": "AIC (BIC reported); independence copula explicit candidate",
            "ci_primary": BOOT_METHOD,
            "ci_B": int(CLUSTER_BOOT_B),
            "ci_B_reduced": False,
            "ci_alpha": CLUSTER_BOOT_ALPHA,
            "ci_percentiles": [100 * CLUSTER_BOOT_ALPHA / 2,
                               100 * (1 - CLUSTER_BOOT_ALPHA / 2)],
            "cluster_unit": "policy_ref (claim-bearing policies of the PIT sample)",
            "pit_rule": ("rankdata(v + N(0, (std(v)+1e-9)*PIT_JITTER_SCALE))/(n+1), "
                         f"PIT_JITTER_SCALE={PIT_JITTER_SCALE}; identical to "
                         "07_conditional.to_pit(jitter=True)"),
            "mple_start": "tau-inversion of raw tau-b of the replicate",
            "joe_tau_map": ("closed form 1+2/(2-theta)*[digamma(2)-digamma(2/theta+1)] "
                            "(R1: replaced a series with a spurious 1/theta factor; "
                            "affects only the Joe tau-inversion start value "
                            "'theta_tau_inv', not the MPLE)"),
            "jitter_sensitivity": (f"diagnostic only: K={JITTER_SENS_K} jitter seeds "
                                   "on the full sample, 07 convention (shared seed) "
                                   "and independent seeds per margin; point "
                                   "estimates remain the GLOBAL_SEED-jitter MPLE"),
            "legacy_mofn_ci": ("diagnostic only: iid m-out-of-n with m=GOF_BOOTSTRAP_CAP="
                               f"{GOF_BOOTSTRAP_CAP}, B={N_BOOTSTRAP}, no rescaling "
                               "(R0 table CI); also sqrt(m/n)-rescaled"),
        },
        "runtime_sec": float(runtime),
        "interpretation": ("Her iki eksende τ≈0 → θ≈bağımsızlık sınırı; frekans-"
                           "şiddet kopulası sermayede, settlement-lag kopulası "
                           "rezervde pratikte gereksiz (niceliksel). [R1] Tam-boyut "
                           "poliçe-küme CI'ları: FS Gauss θ CI sıfırı "
                           + ("İÇERMEZ" if (fs_best == "Gauss" and
                                            (fs_ci[1] < 0 or fs_ci[0] > 0))
                              else "içerir/diğer") + "."),
        "output_files": [str((DATA_PROC_DIR / "copula_fits.pkl").resolve()),
                         str((OUTPUTS_DIR / "11_copulas.csv").resolve()),
                         str((OUTPUTS_DIR / "11_manifest.json").resolve())],
        "environment": env_info(),
    }
    with open(OUTPUTS_DIR / "11_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log.info(f"\nTotal runtime: {runtime/60:.1f} min")
    log.info("\n" + "=" * 78)
    log.info("SCRIPT 11 COMPLETED")
    log.info("=" * 78)


if __name__ == "__main__":
    main()
