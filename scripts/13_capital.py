"""
================================================================================
SCRIPT 13 — Capital proxy under count-coupled frequency-severity dependence
            (paired compound Monte Carlo, claim-level margin held fixed)
================================================================================
PURPOSE
    Simulates the aggregate loss S = sum_i sum_{j<=N_i} Y_ij of a 100,000-policy
    portfolio from the fitted negative binomial count and the fitted body+GPD
    severity splice, and compares VaR/TVaR (95/99/99.5%) and the capital distance
    K = VaR_99.5 - E[S] between
      * a DEPENDENT arm that carries the observed claim-level count-severity
        association (tau_hat_FS = per-claim Kendall tau-b between the policy
        claim count and the claim severity), and
      * an INDEPENDENT arm (N independent of every severity),
    in a PAIRED design (identical count uniforms and identical base severity
    uniforms in both arms).

R1 REDESIGN (audit finding: the pre-R1 dependent arm drew its policy latent
    independently of the count uniform, so N was independent of every severity
    and the "dependent" arm only coupled severities of the same policy, an
    O(rho^2), sign-free effect).  The new dependent arms couple each claim's
    severity to ITS OWN POLICY'S COUNT:

  (A) Gaussian count-coupled scenario (MAIN).  U_i ~ U(0,1), N_i = F_N^{-1}(U_i)
      (capped); for claim j with the shared base uniform p_j,
          z_j = rho * Phi^{-1}(U_i) + sqrt(1-rho^2) * Phi^{-1}(p_j),
          V_j = Phi(z_j),  V'_j = H_rho(V_j),  Y_dep = F_sev^{-1}(V'_j),
      Y_ind = F_sev^{-1}(p_j).  H_rho(v) = E[N 1{V<=v}]/E[N] is the claim-weighted
      CDF of V under the model (normal-score quadrature, monotone interpolation),
      so V' is exactly Uniform(0,1) at claim level: the claim-level severity
      margin is the fitted splice EXACTLY (no count-sum-type mean effect is
      re-imported), and ranks - hence tau(N,Y) - are unchanged by the correction.
      rho* is calibrated so that the SIMULATED claim-level tau-b(N, Y_dep) equals
      tau_hat_FS (slope c = tau_claim / tau_lat from large simulated portfolios).
  (B) Copula-free empirical scenario (family sensitivity).  Pooled mid-rank PIT
      cells of the observed severities; per count class k in {1,2,3,4+} the
      empirical conditional rank distribution R_k (continuous checkerboard
      extension: each claim's mass spread uniformly over its mid-rank cell, tied
      severities over their joint cell).  V_j = R_{k(N_i)}^{-1}(p_j),
      V'_j = H_emp(V_j) with H_emp = sum_k w_k R_k and w_k the claim shares
      implied by the fitted (capped) NB, Y_dep = F_sev^{-1}(V'_j).
  (C) tau_lat sweep (+-0.05, +-0.10, +-0.20, +-0.40) with construction (A).
  (D) Analytic first-order cross-checks of Var(S_i) = E[N s2_N] + Var(N mu_N)
      (empirical class moments, and the model-implied moments of A and B).

DESIGN FEATURES KEPT
    * PAIRED arms; 4 seed blocks (seed + 1000 b); paired-replication bootstrap CI.
    * No shared-uniform leakage (Proposition 2 guard): the count uniform is never
      reused as a severity uniform; it enters severities only through the
      calibrated copula link, and the claim-level margin is re-imposed by H.
    * Split-sample Kupiec coverage check (simulation-internal, not a backtest).

INPUT : data/proc/{marginal_fits,copula_fits}.pkl, data/proc/claim_analytical,
        data/proc/paid_long_real (policy claim counts, as in 06_dependence),
        outputs/06_manifest.json (cluster CI of tau_hat_FS)
OUTPUT: outputs/13_capital.csv, outputs/13_capital_summary.csv,
        outputs/13_tau_sweep.csv, outputs/13_manifest.json,
        figures/13_aggregate.{pdf,png}, logs/13_capital.log
================================================================================
"""
from __future__ import annotations
import os
import sys
import json
import time
import logging
import pickle
from pathlib import Path
from datetime import datetime

# --- Spyder-compatible path setup: look for config.py next to this file or one
# level up (flat scripts/ layout and src/ layout) ---
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
from scipy import stats
from scipy.special import ndtr, ndtri

from config import (
    DATA_PROC_DIR, OUTPUTS_DIR, FIGURES_DIR, LOGS_DIR, GLOBAL_SEED,
    VAR_LEVELS, SOLVENCY_ALPHA,
    N_POLICIES_SIM, N_MC_PAIRED, CAPITAL_NMAX, TAU_FS_FALLBACK,
    SEV_TAIL_Q, CAPITAL_SEED_BLOCKS, CAPITAL_SCR_BOOT,
    FIG_W_FULL, apply_pub_style, env_info,
)

SCRIPT_NAME = "13_capital"

# --- R1 revision constants ---------------------------------------------------
R1_N_WORKERS = min(10, os.cpu_count() or 1)   # multiprocessing workers
R1_CHUNK_REPS = 250                            # paired replications per task
# (A) calibration of the latent Kendall tau to the claim-level tau-b
R1_CALIB_TAU_LAT = (-0.05, -0.10, -0.20, -0.40)
R1_CALIB_N_POL = 5_000_000        # policies per calibration portfolio (CRN seed)
R1_CHECK_N_POL = 10_000_000       # independent-seed check portfolio at rho*
R1_VALID_N_POL = 5_000_000        # large-sample checks of the sweep points (+)
R1_SEED_CALIB = GLOBAL_SEED + 777     # common random numbers for calibration
R1_SEED_CHECK = GLOBAL_SEED + 888     # independent seed for the checks
R1_TAU_BATCHES = 10               # policy batches for the MC s.e. of tau-b
# (A) margin correction H_rho: normal-score grid + Gauss-Legendre quadrature
R1_ZGRID_MAX = 9.0
R1_ZGRID_N = 3601                 # step 0.005 in normal-score space
R1_GL_NODES = 64                  # nodes per count interval
R1_T_UPPER = 10.0                 # truncation of the last count interval
R1_PI_MIN = 1e-15                 # counts with smaller capped-NB mass skipped
# (B) empirical conditional-rank scenario
R1_EMP_N_REP = 20_000             # paired replications (same 4 seed blocks as
R1_EMP_BLOCKS = 4                 # the main run: common random numbers)
R1_EMP_KMAX = 4                   # count classes 1, 2, 3, 4+
# (C) sweep
R1_SWEEP_TAU_LAT = (-0.40, -0.20, -0.10, -0.05, 0.05, 0.10, 0.20, 0.40)
R1_SWEEP_N_REP = 20_000           # paired replications per sweep point (4 seed
R1_SWEEP_BLOCKS = 4               # blocks, common random numbers with main);
                                  # 4,000 gave K-change CIs of +-2-5 points
R1_CURVE_TAU_LAT = tuple(np.unique(np.round(np.r_[
    np.linspace(-0.45, 0.45, 19), np.linspace(-0.03, 0.03, 7)], 4)))
# observed tau_hat_FS policy-cluster bootstrap CI (fallback if 06 is missing)
R1_TAU_FS_CI_FALLBACK = (-0.00979589228422127, -0.0035097581182701497)
# pre-R1 de-attenuated value (13_manifest.json of 27 Sep 2026), comparison only
R1_OLD_TAU_DEATT = -0.011510066485506556
R1_OLD_SCR_BIAS = {"main_gauss_latent": -0.004059094020614218,
                   "ci95": [-0.017227118048432077, 0.028846220246164234],
                   "deattenuated": 0.001053296248931801,
                   "frank": 0.0031150436194726794}


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


def load_pkl(name: str):
    with open(DATA_PROC_DIR / name, "rb") as f:
        return pickle.load(f)


def load_proc(stem: str) -> pd.DataFrame:
    pq = DATA_PROC_DIR / f"{stem}.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    return pd.read_pickle(DATA_PROC_DIR / f"{stem}.pkl")


# =============================================================================
# Margins: capped negative binomial count, body + POT-GPD severity splice
# =============================================================================
def nb_table(r: float, p: float, nmax: int = CAPITAL_NMAX) -> dict:
    """Capped NB count: N = min(F_NB^{-1}(U), nmax).  Returns the capped pmf,
    the CDF used for inversion, and the normal-score interval (a_n, b_n] of
    T = Phi^{-1}(U) that maps to N = n (computed from the survival function for
    tail precision)."""
    n = np.arange(nmax + 1)
    sf = stats.nbinom.sf(n, r, p)                       # P(N_NB > n)
    pi = stats.nbinom.pmf(n, r, p)
    pi[nmax] = sf[nmax - 1]                             # cap mass
    cdf = stats.nbinom.cdf(n, r, p)
    # T = Phi^{-1}(U) in (a_n, b_n]  <=>  N = n;  Phi^{-1}(F(n)) = -Phi^{-1}(sf(n))
    a = np.full(nmax + 1, -np.inf); b = np.full(nmax + 1, np.inf)
    a[1:] = -ndtri(sf[:-1])                             # Phi^{-1}(F(n-1))
    b[:nmax] = -ndtri(sf[:nmax])                        # Phi^{-1}(F(n)); b_nmax=inf
    EN = float(np.sum(n * pi))
    EN2 = float(np.sum(n * (n - 1) * pi))               # E[N(N-1)]
    return {"r": float(r), "p": float(p), "nmax": int(nmax), "n": n, "pi": pi,
            "cdf_cap": cdf[:nmax].copy(), "a": a, "b": b, "EN": EN,
            "ENN1": EN2, "VarN": EN2 + EN - EN ** 2,
            "claim_share": n * pi / EN}


def sample_counts(U: np.ndarray, nbt: dict) -> np.ndarray:
    """min(F_NB^{-1}(U), nmax) by inversion of the tabulated CDF (identical to
    np.minimum(stats.nbinom.ppf(U, r, p), nmax), but vectorised and fast)."""
    return np.searchsorted(nbt["cdf_cap"], U, side="left").astype(np.int64)


def severity_ppf(p, sev_name, sev_params, gpd, tail_q=None):
    """Body + POT-GPD splice quantile F_sev^{-1}(p) (unchanged definition).

    The splice quantile is aligned with the data-driven GPD threshold quantile
    gpd['q'] chosen in 10_marginals (fallback SEV_TAIL_Q), so the body->tail
    switch and the GPD threshold are consistent.  Lognormal body and GPD tail
    are evaluated in closed form (bit-for-bit equivalent to scipy's ppf up to
    rounding; checked in main())."""
    p = np.asarray(p, dtype=float)
    if tail_q is None:
        tail_q = gpd.get("q", SEV_TAIL_Q) if gpd else SEV_TAIL_Q
    sp = tuple(float(x) for x in sev_params)
    if len(sp) == 2:
        sp = (sp[0], 0.0, sp[1])
    pc = np.clip(p, 1e-9, 1 - 1e-9)
    if sev_name == "LogNormal":
        y = sp[1] + sp[2] * np.exp(sp[0] * ndtri(pc))
    else:
        dist = {"Gamma": stats.gamma, "Weibull": stats.weibull_min}[sev_name]
        y = dist.ppf(pc, *sp)
    if gpd and not np.isnan(gpd.get("xi", np.nan)):
        u_th = gpd["threshold"]; xi = gpd["xi"]; beta = gpd["beta"]
        tail_mask = p > tail_q
        if tail_mask.any():
            pe = np.clip((p[tail_mask] - tail_q) / (1 - tail_q), 1e-9, 1 - 1e-9)
            if abs(xi) > 1e-12:
                q_gpd = beta * np.expm1(-xi * np.log1p(-pe)) / xi
            else:
                q_gpd = -beta * np.log1p(-pe)
            y[tail_mask] = u_th + q_gpd
    return y


def severity_ppf_scipy_reference(p, sev_name, sev_params, gpd, tail_q=None):
    """Pre-R1 implementation (scipy ppf); used only to verify severity_ppf."""
    if tail_q is None:
        tail_q = gpd.get("q", SEV_TAIL_Q) if gpd else SEV_TAIL_Q
    dist = {"LogNormal": stats.lognorm, "Gamma": stats.gamma,
            "Weibull": stats.weibull_min}[sev_name]
    sp = tuple(sev_params)
    if len(sp) == 2:
        sp = (sp[0], 0.0, sp[1])
    y = dist.ppf(np.clip(p, 1e-9, 1 - 1e-9), *sp)
    if gpd and not np.isnan(gpd.get("xi", np.nan)):
        u_th = gpd["threshold"]; xi = gpd["xi"]; beta = gpd["beta"]
        tail_mask = p > tail_q
        if tail_mask.any():
            pe = (p[tail_mask] - tail_q) / (1 - tail_q)
            y[tail_mask] = u_th + stats.genpareto.ppf(
                np.clip(pe, 1e-9, 1 - 1e-9), xi, scale=beta)
    return y


class SplicePartialMoments:
    """Exact partial moments of the splice quantile function:
         lower L_k(u) = int_0^u F^{-1}(w)^k dw   (body, u <= q)
         upper G_k(x) = int_{1-x}^1 F^{-1}(w)^k dw (GPD tail, x <= 1-q)
    (lognormal body and GPD tail in closed form; the 1e-9 clipping used in the
    simulation is ignored, its effect on E[Y^2] is < 0.1%).  Used by the
    analytic cross-checks (D) to integrate F^{-1} over cells of V'."""

    def __init__(self, sev_name, sev_params, gpd):
        if sev_name != "LogNormal":
            raise NotImplementedError("analytic moments implemented for the "
                                      "lognormal body (AIC choice)")
        s, loc, scale = (float(x) for x in sev_params)
        if loc != 0.0:
            raise NotImplementedError("lognormal loc must be 0")
        self.mu_ln, self.s = np.log(scale), s
        self.q = float(gpd.get("q", SEV_TAIL_Q))
        self.xi, self.beta, self.u = gpd["xi"], gpd["beta"], gpd["threshold"]
        self.A = self.u - self.beta / self.xi
        self.B = self.beta / self.xi
        if not self.xi < 0.5:
            raise ValueError("second moment infinite (xi >= 0.5)")
        self.Lq1 = self.L(np.array([self.q]), 1)[0]
        self.Lq2 = self.L(np.array([self.q]), 2)[0]
        self.G1tot = self.G(np.array([1 - self.q]), 1)[0]
        self.G2tot = self.G(np.array([1 - self.q]), 2)[0]
        self.m1 = self.Lq1 + self.G1tot
        self.m2 = self.Lq2 + self.G2tot

    def L(self, u, k):
        u = np.minimum(np.asarray(u, float), self.q)
        return (np.exp(k * self.mu_ln + 0.5 * (k * self.s) ** 2)
                * ndtr(ndtri(u) - k * self.s))

    def G(self, x, k):
        x = np.minimum(np.asarray(x, float), 1 - self.q)
        xx = x / (1 - self.q)
        xi, A, B = self.xi, self.A, self.B
        if k == 1:
            v = A * xx + B * xx ** (1 - xi) / (1 - xi)
        else:
            v = (A * A * xx + 2 * A * B * xx ** (1 - xi) / (1 - xi)
                 + B * B * xx ** (1 - 2 * xi) / (1 - 2 * xi))
        return (1 - self.q) * v

    def cell_moments(self, u_lo, u_hi, x_lo, x_hi):
        """Mean of F^{-1}(W) and F^{-1}(W)^2 for W ~ U(u_lo, u_hi), where
        x = 1 - u is supplied separately for tail precision."""
        u_lo = np.asarray(u_lo, float); u_hi = np.asarray(u_hi, float)
        x_lo = np.asarray(x_lo, float); x_hi = np.asarray(x_hi, float)
        width = np.where(u_hi <= 0.5, u_hi - u_lo, x_lo - x_hi)
        out = []
        for k, Lq, Gt in ((1, self.Lq1, self.G1tot), (2, self.Lq2, self.G2tot)):
            body = np.where(u_lo < self.q,
                            self.L(np.minimum(u_hi, self.q), k)
                            - self.L(u_lo, k), 0.0)
            tail = np.where(x_hi < 1 - self.q,
                            self.G(np.minimum(x_lo, 1 - self.q), k)
                            - self.G(x_hi, k), 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                out.append(np.where(width > 0, (body + tail) / width, 0.0))
        return out[0], out[1], width


# =============================================================================
# (A) Gaussian count-coupled construction: margin correction H_rho
# =============================================================================
def _count_nodes(nbt: dict, nodes: int = R1_GL_NODES):
    """Gauss-Legendre nodes/weights (weights include phi(t)) on each normal-score
    interval (a_n, b_n], n >= 1, of the capped count."""
    xg, wg = np.polynomial.legendre.leggauss(nodes)
    out = []
    for n in range(1, nbt["nmax"] + 1):
        if nbt["pi"][n] < R1_PI_MIN:
            continue
        a, b = nbt["a"][n], min(nbt["b"][n], R1_T_UPPER)
        if not a < b:
            continue
        t = 0.5 * (b - a) * xg + 0.5 * (a + b)
        w = 0.5 * (b - a) * wg * stats.norm.pdf(t)
        out.append((n, t, w))
    return out


def gauss_H(rho: float, nbt: dict, zgrid: np.ndarray) -> dict:
    """Claim-weighted CDF of V = Phi(rho T + sqrt(1-rho^2) eps) under the model,
         H_rho(v) = (1/E[N]) sum_n n int_{F(n-1)}^{F(n)} Phi((Phi^{-1}(v)-rho
                    Phi^{-1}(w))/sqrt(1-rho^2)) dw,
    evaluated on a normal-score grid z = Phi^{-1}(v) (quadrature in t =
    Phi^{-1}(w)).  Returns g(z) = Phi^{-1}(H_rho(Phi(z))), which is increasing,
    so V' = Phi(interp(z, zgrid, g)) is a monotone map of V."""
    s = np.sqrt(max(1.0 - rho * rho, 1e-300))
    H = np.zeros_like(zgrid); Hc = np.zeros_like(zgrid); tot = 0.0
    for n, t, w in _count_nodes(nbt):
        arg = (zgrid[:, None] - rho * t[None, :]) / s
        H += n * (ndtr(arg) @ w)
        Hc += n * (ndtr(-arg) @ w)
        tot += n * w.sum()
    H /= tot; Hc /= tot
    g = np.where(H < 0.5, ndtri(np.clip(H, 1e-300, 1)),
                 -ndtri(np.clip(Hc, 1e-300, 1)))
    if not np.all(np.diff(g) > 0):
        raise RuntimeError("H_rho grid not strictly increasing")
    return {"rho": float(rho), "g": g, "H": H, "Hc": Hc,
            "norm_quad": float(tot), "norm_EN": nbt["EN"]}


def claim_tau_b_gauss(rho: float, nbt: dict) -> float:
    """Population claim-level Kendall tau-b between N and Y_dep under (A):
    tau_a = 2 sum_{n>m} q_n q_m (2 P(Z_n > Z_m) - 1), q_n = n pi_n / E[N],
    P(Z_n > Z_m) = E Phi(rho (T_n - T_m) / (sqrt(1-rho^2) sqrt 2)), T_n the
    truncated normal score of the count interval; tau_b = tau_a /
    sqrt(1 - sum q_n^2).  Pairs of claims from the same policy have tied N and
    vanish in the limit of a large portfolio."""
    s = np.sqrt(1.0 - rho * rho)
    nodes = _count_nodes(nbt)
    q = {n: n * w.sum() for n, t, w in nodes}
    tot = sum(q.values()); q = {n: v / tot for n, v in q.items()}
    ta = 0.0
    for i, (n, tn, wn) in enumerate(nodes):
        for (m, tm, wm) in nodes[:i]:
            P = (wn / wn.sum()) @ ndtr(rho * (tn[:, None] - tm[None, :])
                                       / (s * np.sqrt(2.0))) @ (wm / wm.sum())
            ta += 2 * q[n] * q[m] * (2 * P - 1)
    ties = sum(v * v for v in q.values())
    return float(ta / np.sqrt(1.0 - ties))


def gauss_policy_moments(rho: float, nbt: dict, zgrid: np.ndarray, Hd: dict,
                         spm: SplicePartialMoments) -> dict:
    """Model-implied per-policy moments under (A) (analytic cross-check D3):
    m(t) = E[Y | T=t], m2(t) = E[Y^2 | T=t] by cell integration of F^{-1}(V')
    over the z-grid; then
      Var(S_i) = sum_n pi_n [n E(Y^2|n) + n(n-1) E(m(T)^2|n)] - (sum_n pi_n n mu_n)^2
    (the n(n-1) term carries the within-policy covariance given N)."""
    s = np.sqrt(1.0 - rho * rho)
    g = Hd["g"]
    # cell boundaries in V' (with the two outer cells)
    u_b = np.concatenate([[0.0], ndtr(g), [1.0]])
    x_b = np.concatenate([[1.0], ndtr(-g), [0.0]])
    cm1, cm2, _ = spm.cell_moments(u_b[:-1], u_b[1:], x_b[:-1], x_b[1:])
    zb = np.concatenate([[-np.inf], zgrid, [np.inf]])
    ES = 0.0; ES2 = 0.0; mu_n = {}
    for n, t, w in _count_nodes(nbt):
        wn = w / w.sum()
        Fz = ndtr((zb[None, :] - rho * t[:, None]) / s)
        Pc = np.diff(Fz, axis=1)                      # P(cell | t)
        m1 = Pc @ cm1; m2 = Pc @ cm2
        mu = float(wn @ m1); ey2 = float(wn @ m2); emm = float(wn @ (m1 * m1))
        pin = nbt["pi"][n]
        ES += pin * n * mu
        ES2 += pin * (n * ey2 + n * (n - 1) * emm)
        mu_n[n] = mu
    var_dep = ES2 - ES ** 2
    var_ind = nbt["EN"] * spm.m2 + nbt["ENN1"] * spm.m1 ** 2 \
        - (nbt["EN"] * spm.m1) ** 2
    return {"ES_policy_dep": ES, "ES_policy_ind": nbt["EN"] * spm.m1,
            "mean_ratio": ES / (nbt["EN"] * spm.m1),
            "var_policy_dep": var_dep, "var_policy_ind": var_ind,
            "var_ratio": var_dep / var_ind,
            "sd_ratio": float(np.sqrt(var_dep / var_ind)),
            "K_change_pred_pct": float((np.sqrt(var_ind / var_dep) - 1) * 100),
            "mu_by_count": {int(k): float(v) for k, v in mu_n.items()
                            if k <= 6}}


# =============================================================================
# (B) Empirical conditional-rank construction
# =============================================================================
def build_empirical(sev: np.ndarray, ncl: np.ndarray, nbt: dict,
                    kmax: int = R1_EMP_KMAX) -> dict:
    """Pooled mid-rank PIT cells of the observed severities and, for each count
    class k (1,...,kmax-1, kmax+), the empirical conditional rank distribution
    R_k.  Cell r = ((r-1)/n, r/n] holds the claim of pooled rank r; a claim's
    mid-rank PIT is its cell centre (tied severities share the union of their
    cells, i.e. the mid-rank interval).  R_k is the continuous (checkerboard)
    version: each claim of class k spreads mass 1/m_k uniformly over its
    (tie-group) cell.  H_emp = sum_k w_k R_k with w_k the NB-implied claim
    shares.  All CDFs are piecewise linear on the grid v = i/n."""
    n = len(sev)
    cls = np.minimum(ncl, kmax).astype(int) - 1          # 0..kmax-1
    order = np.argsort(sev, kind="mergesort")
    s_sorted = sev[order]; c_sorted = cls[order]
    grp_start = np.r_[0, np.flatnonzero(np.diff(s_sorted) != 0) + 1]
    grp_size = np.diff(np.r_[grp_start, n])
    grp_id = np.repeat(np.arange(len(grp_start)), grp_size)
    cnt = np.zeros((len(grp_start), kmax))
    np.add.at(cnt, (grp_id, c_sorted), 1.0)
    mass = np.repeat(cnt / grp_size[:, None], grp_size, axis=0)   # (n, kmax)
    m_k = mass.sum(axis=0)
    cum = np.vstack([np.zeros(kmax), np.cumsum(mass, axis=0)])    # (n+1, kmax)
    rev = np.vstack([np.cumsum(mass[::-1], axis=0)[::-1], np.zeros(kmax)])
    R = cum / m_k; Rc = rev / m_k
    R[-1] = 1.0; Rc[-1] = 0.0
    R = np.maximum.accumulate(R, axis=0)
    vgrid = np.arange(n + 1) / n
    # NB-implied claim shares by class
    share = nbt["claim_share"]
    w = np.array([share[k + 1] for k in range(kmax - 1)]
                 + [share[kmax:].sum()])
    H = R @ w; Hc = Rc @ w
    H[0] = 0.0; H[-1] = 1.0
    data_share = m_k / n
    # mid-rank PIT tau-b(N, PIT) in the data (= tau-b(N, severity))
    return {"n": n, "kmax": kmax, "vgrid": vgrid,
            "R": [np.ascontiguousarray(R[:, k]) for k in range(kmax)],
            "Rc": Rc, "H": H, "Hc": Hc, "w_sim": w, "w_data": data_share,
            "mass": mass, "m_k": m_k, "n_tie_groups": int(len(grp_start)),
            "n_tied_claims": int(grp_size[grp_size > 1].sum()),
            "max_tie_size": int(grp_size.max())}


def emp_policy_moments(emp: dict, nbt: dict, spm: SplicePartialMoments) -> dict:
    """Model-implied class moments under (B) (exact: R_k and H_emp are linear on
    each cell, so V' is uniform on the H-image of each cell) and the implied
    per-policy variance (claims i.i.d. given N)."""
    H, Hc = emp["H"], emp["Hc"]
    cm1, cm2, width = spm.cell_moments(H[:-1], H[1:], Hc[:-1], Hc[1:])
    dmass = emp["mass"] / emp["m_k"]                  # (n, kmax) class weights
    mu_k = dmass.T @ cm1; m2_k = dmass.T @ cm2
    kmax = emp["kmax"]
    ES = 0.0; ES2 = 0.0
    for nn in range(1, nbt["nmax"] + 1):
        k = min(nn, kmax) - 1
        pin = nbt["pi"][nn]
        ES += pin * nn * mu_k[k]
        ES2 += pin * (nn * m2_k[k] + nn * (nn - 1) * mu_k[k] ** 2)
    var_dep = ES2 - ES ** 2
    var_ind = nbt["EN"] * spm.m2 + nbt["ENN1"] * spm.m1 ** 2 \
        - (nbt["EN"] * spm.m1) ** 2
    return {"mu_by_class": [float(x) for x in mu_k],
            "sd_by_class": [float(np.sqrt(b - a * a)) for a, b in zip(mu_k, m2_k)],
            "ES_policy_dep": ES, "ES_policy_ind": nbt["EN"] * spm.m1,
            "mean_ratio": ES / (nbt["EN"] * spm.m1),
            "var_policy_dep": var_dep, "var_policy_ind": var_ind,
            "var_ratio": var_dep / var_ind,
            "sd_ratio": float(np.sqrt(var_dep / var_ind)),
            "K_change_pred_pct": float((np.sqrt(var_ind / var_dep) - 1) * 100)}


def emp_claim_tau_b(emp: dict, nbt: dict) -> float:
    """Population claim-level Kendall tau-b between N and Y_dep under (B):
    P(V_k > V_l) = sum_r dR_k(r) [R_l(v_{r-1}) + dR_l(r)/2] (cell-uniform
    masses), tau_a = 2 sum_{n>m} q_n q_m (2 P(V_k(n) > V_k(m)) - 1)."""
    K = emp["kmax"]
    dm = emp["mass"] / emp["m_k"]
    R = np.column_stack(emp["R"])
    P = np.array([[np.sum(dm[:, k] * (R[:-1, l] + 0.5 * dm[:, l]))
                   for l in range(K)] for k in range(K)])
    q = nbt["claim_share"]
    ns = [n for n in range(1, nbt["nmax"] + 1) if q[n] > 0]
    ta = 0.0
    for i, n in enumerate(ns):
        for m in ns[:i]:
            ta += 2 * q[n] * q[m] * (2 * P[min(n, K) - 1, min(m, K) - 1] - 1)
    return float(ta / np.sqrt(1.0 - np.sum(q[1:] ** 2)))


def empirical_moment_check(sev, ncl, pi, label, kmax=4, exact_upto=None):
    """(D1) Var(S_i) = E[N s2_N] + Var(N mu_N) with unadjusted empirical severity
    moments by count class vs the independent Var = E[N] s2 + Var(N) mu^2
    (pooled claim moments); pi = policy count distribution (index = count)."""
    if exact_upto is None:
        key = np.minimum(ncl, kmax)
        top = kmax
    else:
        key = np.minimum(ncl, exact_upto)
        top = exact_upto
    mu = {}; s2 = {}; nk = {}
    for k in range(1, top + 1):
        y = sev[key == k]
        mu[k] = float(y.mean()); s2[k] = float(y.var()); nk[k] = int(len(y))
    mu0 = float(sev.mean()); s20 = float(sev.var())
    nn = np.arange(len(pi)); EN = float(np.sum(nn * pi))
    ES = 0.0; ES2 = 0.0; EN2 = float(np.sum(nn * nn * pi))
    for n_ in range(1, len(pi)):
        k = min(n_, top)
        ES += pi[n_] * n_ * mu[k]
        ES2 += pi[n_] * (n_ * s2[k] + (n_ * mu[k]) ** 2)
    var_dep = ES2 - ES ** 2
    var_ind = EN * s20 + (EN2 - EN ** 2) * mu0 ** 2
    return {"label": label, "classes": "exact n<=%d, %d+" % (top - 1, top)
            if exact_upto else "1,2,3,4+",
            "class_mean": {str(k): mu[k] for k in mu},
            "class_sd": {str(k): float(np.sqrt(s2[k])) for k in s2},
            "class_n_claims": {str(k): nk[k] for k in nk},
            "pooled_mean": mu0, "pooled_sd": float(np.sqrt(s20)),
            "ES_policy_dep": ES, "ES_policy_ind": EN * mu0,
            "mean_ratio": ES / (EN * mu0),
            "var_ratio": var_dep / var_ind,
            "sd_ratio": float(np.sqrt(var_dep / var_ind)),
            "sd_change_pct": float((np.sqrt(var_dep / var_ind) - 1) * 100)}


# =============================================================================
# Simulation workers (multiprocessing; per-replication seeding -> results do
# not depend on the number of workers or the chunking)
# =============================================================================
_CTX: dict = {}


def _init_worker(ctx):
    global _CTX
    _CTX = ctx


def _rep_rng(block: int, rep: int):
    """Replication `rep` of seed block `block` (block seed = GLOBAL_SEED+1000 b)."""
    return np.random.default_rng([GLOBAL_SEED + 1000 * block, rep])


def _draw_portfolio(rng, n_pol, nbt):
    U = rng.random(n_pol)
    N = sample_counts(U, nbt)
    tot = int(N.sum())
    p = rng.random(tot)
    pol = np.flatnonzero(N)
    Nc = N[pol]
    return U, N, pol, Nc, p


def _dep_uniform(scen, U, pol, Nc, p, zp=None, tclaim=None, cache=None):
    """V' for one dependent scenario (claim order = np.repeat(pol, Nc))."""
    ctx = _CTX
    if scen[0] == "gauss":
        _, name, rho, g = scen
        if cache is not None and "zp" not in cache:
            cache["zp"] = ndtri(p)
            cache["tc"] = np.repeat(ndtri(U[pol]), Nc)
        zp = cache["zp"]; tc = cache["tc"]
        z = rho * tc + np.sqrt(1.0 - rho * rho) * zp
        return ndtr(np.interp(z, ctx["zgrid"], g)), z
    # empirical conditional-rank scenario
    emp = ctx["emp"]
    k = np.minimum(np.repeat(Nc, Nc), emp["kmax"]) - 1
    V = np.empty_like(p)
    for kk in range(emp["kmax"]):
        m = k == kk
        if m.any():
            V[m] = np.interp(p[m], emp["R"][kk], emp["vgrid"])
    return np.interp(V, emp["vgrid"], emp["H"]), V


def _task_paired(task):
    """Paired replications [r0, r1) of seed block b for a list of scenarios."""
    b, r0, r1, scens = task
    ctx = _CTX
    nbt, sv = ctx["nbt"], ctx["sev"]
    n_pol = ctx["n_pol"]
    S_ind = np.empty(r1 - r0)
    S_dep = {sc[1]: np.empty(r1 - r0) for sc in scens}
    for i, rep in enumerate(range(r0, r1)):
        rng = _rep_rng(b, rep)
        U, N, pol, Nc, p = _draw_portfolio(rng, n_pol, nbt)
        if p.size == 0:
            S_ind[i] = 0.0
            for sc in scens:
                S_dep[sc[1]][i] = 0.0
            continue
        S_ind[i] = severity_ppf(p, *sv).sum()
        cache = {}
        for sc in scens:
            Vp, _ = _dep_uniform(sc, U, pol, Nc, p, cache=cache)
            S_dep[sc[1]][i] = severity_ppf(Vp, *sv).sum()
    return b, r0, S_ind, S_dep


def _task_large(task):
    """Large-portfolio claim-level checks for one scenario: tau-b(N, Y_dep)
    (+ batch s.e.), KS of V' against U(0,1) (margin correction) and of the
    uncorrected V, and claim-level severity means of both arms."""
    name, scen, n_pol, seed = task
    ctx = _CTX
    nbt, sv = ctx["nbt"], ctx["sev"]
    rng = np.random.default_rng(seed)
    U, N, pol, Nc, p = _draw_portfolio(rng, n_pol, nbt)
    Vp, raw = _dep_uniform(scen, U, pol, Nc, p, cache={})
    Ncl = np.repeat(Nc, Nc)
    Yd = severity_ppf(Vp, *sv); Yi = severity_ppf(p, *sv)
    tau = stats.kendalltau(Ncl, Yd).statistic
    # MC s.e.: R1_TAU_BATCHES disjoint policy batches (claims of a policy stay
    # together), se = sd(batch tau) / sqrt(#batches)
    pol_of_claim = np.repeat(np.arange(len(pol)), Nc)
    bid = pol_of_claim % R1_TAU_BATCHES
    bt = np.array([stats.kendalltau(Ncl[bid == j], Yd[bid == j]).statistic
                   for j in range(R1_TAU_BATCHES)])
    se = float(bt.std(ddof=1) / np.sqrt(R1_TAU_BATCHES))
    raw_u = ndtr(raw) if scen[0] == "gauss" else raw
    ks_corr = stats.kstest(Vp, "uniform")
    ks_raw = stats.kstest(raw_u, "uniform")
    return {"name": name, "n_pol": int(n_pol), "n_claims": int(p.size),
            "seed": int(seed), "tau_b_claim": float(tau), "tau_b_se": se,
            "ks_Vcorr_D": float(ks_corr.statistic),
            "ks_Vcorr_p": float(ks_corr.pvalue),
            "ks_Vraw_D": float(ks_raw.statistic),
            "ks_Vraw_p": float(ks_raw.pvalue),
            "mean_Y_dep": float(Yd.mean()), "mean_Y_ind": float(Yi.mean()),
            "mean_Y_ratio": float(Yd.mean() / Yi.mean()),
            "mean_Vcorr": float(Vp.mean()), "mean_Vraw": float(raw_u.mean())}


def _task_curve(task):
    """Analytic claim-level tau-b and first-order K change for one tau_lat."""
    tau_lat, = task
    ctx = _CTX
    rho = float(np.sin(np.pi * tau_lat / 2))
    Hd = gauss_H(rho, ctx["nbt"], ctx["zgrid"])
    mom = gauss_policy_moments(rho, ctx["nbt"], ctx["zgrid"], Hd, ctx["spm"])
    return {"tau_lat": float(tau_lat), "rho": rho,
            "tau_b_claim_analytic": claim_tau_b_gauss(rho, ctx["nbt"]),
            "K_change_pred_pct": mom["K_change_pred_pct"],
            "sd_ratio_pred": mom["sd_ratio"], "mean_ratio": mom["mean_ratio"]}


def run_tasks(func, tasks, ctx, pool=None):
    if pool is not None:
        return list(pool.imap(func, tasks))
    _init_worker(ctx)
    return [func(t) for t in tasks]


# =============================================================================
# Risk metrics
# =============================================================================
def risk_metrics(agg, levels=VAR_LEVELS):
    out = {}
    for a in levels:
        var = np.quantile(agg, a)
        tvar = agg[agg >= var].mean() if (agg >= var).any() else var
        out[f"VaR_{a}"] = float(var)
        out[f"TVaR_{a}"] = float(tvar)
    return out


def paired_metrics(agg_dep, agg_ind, levels=VAR_LEVELS):
    """Changes of the independence model relative to the dependent arm
    (positive = independence gives the higher figure), as in the pre-R1 run:
      VaR/TVaR change = X_ind / X_dep - 1,
      capital distance K_a = VaR_a - E[S];  K change = K_ind / K_dep - 1."""
    rd = risk_metrics(agg_dep, levels); ri = risk_metrics(agg_ind, levels)
    md, mi = float(agg_dep.mean()), float(agg_ind.mean())
    out = {"ES_dep": md, "ES_ind": mi, "ES_change_pct": (mi / md - 1) * 100,
           "SD_dep": float(agg_dep.std(ddof=1)),
           "SD_ind": float(agg_ind.std(ddof=1))}
    out["SD_change_pct"] = (out["SD_ind"] / out["SD_dep"] - 1) * 100
    for a in levels:
        vd, vi = rd[f"VaR_{a}"], ri[f"VaR_{a}"]
        td, ti = rd[f"TVaR_{a}"], ri[f"TVaR_{a}"]
        out[f"VaR_dep_{a}"] = vd; out[f"VaR_ind_{a}"] = vi
        out[f"TVaR_dep_{a}"] = td; out[f"TVaR_ind_{a}"] = ti
        out[f"VaR_change_pct_{a}"] = (vi / vd - 1) * 100
        out[f"TVaR_change_pct_{a}"] = (ti / td - 1) * 100
        out[f"K_dep_{a}"] = vd - md; out[f"K_ind_{a}"] = vi - mi
        out[f"K_change_pct_{a}"] = ((vi - mi) / (vd - md) - 1) * 100
    return out


def paired_bootstrap(agg_dep, agg_ind, n_boot, seed, levels=VAR_LEVELS):
    """Paired-replication bootstrap: the same resampled replication indices are
    applied to both arms; percentile 95% CIs of every change metric."""
    rng = np.random.default_rng(seed)
    n = len(agg_dep)
    keys = (["ES_change_pct", "SD_change_pct"]
            + [f"{m}_change_pct_{a}" for a in levels
               for m in ("VaR", "TVaR", "K")])
    res = {k: np.empty(n_boot) for k in keys}
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        pm = paired_metrics(agg_dep[idx], agg_ind[idx], levels)
        for k in keys:
            res[k][i] = pm[k]
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in res.items()}


def kupiec_pof(agg, var, alpha):
    """Kupiec POF test: is the violation rate compatible with 1 - alpha?"""
    n = len(agg); x = int(np.sum(agg > var)); pi = x / n
    p_exp = 1 - alpha
    if x == 0 or x == n:
        return {"violations": x, "expected": float(n * p_exp), "n": n,
                "rate": pi, "LR": np.nan, "p": np.nan}
    LR = -2 * (np.log((1 - p_exp) ** (n - x) * p_exp ** x)
               - np.log((1 - pi) ** (n - x) * pi ** x))
    pval = 1 - stats.chi2.cdf(LR, 1)
    return {"violations": x, "expected": float(n * p_exp), "n": n,
            "rate": float(pi), "LR": float(LR), "p": float(pval)}


def split_sample_exchangeable_p(m: int, n: int, alpha: float, x: int) -> dict:
    """Exact two-sided p-value of the split-sample exceedance count when the
    threshold is ESTIMATED from the first half (the Kupiec LR treats it as
    known and is anti-conservative).  For exchangeable continuous replications,
    the number X of second-half values above the r-th largest of the first
    half satisfies P(X = x) = C(r-1+x, x) C(m+n-r-x, n-x) / C(m+n, n)
    (Gumbel & von Schelling 1950).  np.quantile(., alpha) with m values lies
    at or above the r-th largest, r = m - floor((m-1) alpha); exceedances of
    the r-th largest are used (slightly conservative)."""
    from scipy.special import gammaln
    r = m - int(np.floor((m - 1) * alpha))
    xs = np.arange(n + 1)
    lp = (gammaln(r + xs) - gammaln(xs + 1) - gammaln(r)
          + gammaln(m + n - r - xs + 1) - gammaln(n - xs + 1)
          - gammaln(m - r + 1)
          - (gammaln(m + n + 1) - gammaln(n + 1) - gammaln(m + 1)))
    pmf = np.exp(lp); pmf /= pmf.sum()
    lo = pmf[:x + 1].sum(); hi = pmf[x:].sum()
    return {"p_exchangeable": float(min(1.0, 2 * min(lo, hi))),
            "expected_exchangeable": float(np.sum(xs * pmf)),
            "sd_exchangeable": float(np.sqrt(np.sum(xs ** 2 * pmf)
                                             - np.sum(xs * pmf) ** 2)),
            "sd_binomial": float(np.sqrt(n * (1 - alpha) * alpha)),
            "r_order_stat_from_top": int(r)}


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    global log
    log = setup_logger()
    apply_pub_style()
    t_start = time.time()
    log.info("=" * 78)
    log.info(f"SCRIPT {SCRIPT_NAME} — Capital proxy, count-coupled paired MC (R1)")
    log.info("=" * 78)

    outfiles: list = []
    marg = load_pkl("marginal_fits.pkl")
    nb = tuple(float(x) for x in marg["frequency"]["params"])
    sev_name = marg["severity"]["name"]
    sev_params = [float(x) for x in marg["severity"]["params"]]
    gpd = {k: v for k, v in marg.get("gpd", {}).items()
           if k in ("threshold", "xi", "beta", "q", "n_exceed", "ks")}
    sv = (sev_name, sev_params, gpd)
    try:
        cop = load_pkl("copula_fits.pkl")
        tau_hat = float(cop.get("axis_FS", {}).get("tau_obs", -TAU_FS_FALLBACK))
    except Exception:
        tau_hat = -TAU_FS_FALLBACK
    tau_ci = list(R1_TAU_FS_CI_FALLBACK)
    try:
        m06 = json.load(open(OUTPUTS_DIR / "06_manifest.json", encoding="utf-8"))
        tau_ci = [float(x) for x in m06["axis_FS_bca_ci"]]
    except Exception:
        pass
    nbt = nb_table(*nb, CAPITAL_NMAX)
    log.info(f"\n[1/8] Margins: NB(r={nb[0]:.4f}, p={nb[1]:.4f}) capped at "
             f"{CAPITAL_NMAX}, E[N]={nbt['EN']:.5f}, E[N(N-1)]={nbt['ENN1']:.5f};"
             f" severity {sev_name} body + GPD (q={gpd.get('q')}, "
             f"xi={gpd.get('xi', np.nan):.3f})")
    log.info(f"  Observed claim-level tau_hat_FS = {tau_hat:+.6f} "
             f"(policy-cluster 95% CI [{tau_ci[0]:+.4f}, {tau_ci[1]:+.4f}])")

    # --- verification of the fast inversions against the pre-R1 scipy code ---
    rng_v = np.random.default_rng(GLOBAL_SEED + 5)
    uu = rng_v.random(2_000_000)
    uu[:5] = [1e-12, 0.5, gpd.get("q", 0.9), 1 - 1e-12, 1 - 1e-6]
    y_fast = severity_ppf(uu, *sv)
    y_ref = severity_ppf_scipy_reference(uu, *sv)
    sev_ppf_maxrel = float(np.max(np.abs(y_fast / y_ref - 1)))
    n_ref = np.minimum(stats.nbinom.ppf(uu, *nb).astype(int), CAPITAL_NMAX)
    count_mismatch = int(np.sum(n_ref != sample_counts(uu, nbt)))
    log.info(f"  Check: fast severity quantile vs scipy max rel. diff = "
             f"{sev_ppf_maxrel:.2e}; count inversion mismatches = "
             f"{count_mismatch} / {len(uu):,}")
    spm = SplicePartialMoments(sev_name, sev_params, gpd) \
        if sev_name == "LogNormal" else None
    if spm is not None:
        log.info(f"  Splice moments (exact): E[Y]={spm.m1:,.1f}, "
                 f"SD[Y]={np.sqrt(spm.m2 - spm.m1 ** 2):,.1f}; MC check E[Y]="
                 f"{y_fast.mean():,.1f}")

    # --- claim data for (B) and (D): policy claim count as in 06_dependence ---
    log.info("\n[2/8] Claim data: pooled mid-rank PIT cells by count class (B)")
    paid = load_proc("paid_long_real")[["policy_ref", "claim_file"]]
    freq = (paid.groupby("policy_ref")["claim_file"].nunique()
            .rename("n_claims").reset_index())
    del paid
    ca = load_proc("claim_analytical")[["claim_file", "policy_ref",
                                        "severity_real"]]
    ca = ca.merge(freq, on="policy_ref", how="left")
    ca = ca[ca["severity_real"] > 0]
    sev = ca["severity_real"].to_numpy(float)
    ncl = ca["n_claims"].to_numpy(int)
    tau_data = float(stats.kendalltau(ncl, sev).statistic)
    del ca, freq
    emp = build_empirical(sev, ncl, nbt)
    emp["tau_b_analytic"] = emp_claim_tau_b(emp, nbt)
    log.info(f"  n_claims={len(sev):,}; tau-b(N, severity) recomputed = "
             f"{tau_data:+.6f} (06/11: {tau_hat:+.6f}); tie groups: "
             f"{emp['n_tied_claims']:,} claims in ties, max tie size "
             f"{emp['max_tie_size']}")
    log.info(f"  (B) population claim-level tau-b under the fitted NB = "
             f"{emp['tau_b_analytic']:+.6f}")
    log.info("  claim shares by class 1/2/3/4+: data "
             + "/".join(f"{x:.4f}" for x in emp["w_data"]) + "; fitted NB "
             + "/".join(f"{x:.4f}" for x in emp["w_sim"]))

    zgrid = np.linspace(-R1_ZGRID_MAX, R1_ZGRID_MAX, R1_ZGRID_N)
    emp_ctx = {"kmax": emp["kmax"], "vgrid": emp["vgrid"], "R": emp["R"],
               "H": emp["H"]}
    ctx = {"nbt": nbt, "sev": sv, "n_pol": N_POLICIES_SIM, "zgrid": zgrid,
           "emp": emp_ctx, "spm": spm}

    pool = None
    if R1_N_WORKERS > 1:
        try:
            import multiprocessing as mp
            pool = mp.get_context("spawn").Pool(R1_N_WORKERS,
                                                initializer=_init_worker,
                                                initargs=(ctx,))
        except Exception as e:                       # pragma: no cover
            log.info(f"  multiprocessing unavailable ({e}); sequential run")
            pool = None
    _init_worker(ctx)
    try:
        results = _run_all(ctx, pool, nbt, sv, spm, emp, zgrid, tau_hat,
                           tau_ci, tau_data, sev, ncl, outfiles,
                           sev_ppf_maxrel, count_mismatch, gpd)
    finally:
        if pool is not None:
            pool.close(); pool.join()
    log.info(f"\nTotal run time {time.time() - t_start:,.0f} s")
    log.info("\n" + "=" * 78)
    log.info("SCRIPT 13 COMPLETED")
    log.info("=" * 78)
    return results


def _gauss_scen(name, tau_lat, nbt, zgrid):
    rho = float(np.sin(np.pi * tau_lat / 2))
    Hd = gauss_H(rho, nbt, zgrid)
    return ("gauss", name, rho, Hd["g"]), Hd


def _run_all(ctx, pool, nbt, sv, spm, emp, zgrid, tau_hat, tau_ci, tau_data,
             sev, ncl, outfiles, sev_ppf_maxrel, count_mismatch, gpd):
    # ------------------------------------------------------------------ (A) --
    log.info("\n[3/8] (A) Calibration of the latent tau to the claim-level tau-b")
    cal_tasks = []
    for tl in R1_CALIB_TAU_LAT:
        sc, _ = _gauss_scen(f"calib_{tl:+.2f}", tl, nbt, zgrid)
        cal_tasks.append((sc[1], sc, R1_CALIB_N_POL, R1_SEED_CALIB))
    cal = run_tasks(_task_large, cal_tasks, ctx, pool)
    x = np.array(R1_CALIB_TAU_LAT); y = np.array([c["tau_b_claim"] for c in cal])
    c_lin = float(np.sum(x * y) / np.sum(x * x))
    Xc = np.column_stack([x, x ** 3])
    c1, c3 = np.linalg.lstsq(Xc, y, rcond=None)[0]
    for tl, c in zip(R1_CALIB_TAU_LAT, cal):
        rho_ = np.sin(np.pi * tl / 2)
        c["tau_lat"] = tl; c["rho"] = float(rho_)
        c["tau_b_claim_analytic"] = claim_tau_b_gauss(rho_, nbt)
        c["ratio_sim"] = c["tau_b_claim"] / tl
        log.info(f"  tau_lat={tl:+.2f} (rho={rho_:+.4f}): simulated claim "
                 f"tau-b={c['tau_b_claim']:+.5f} (MC s.e. {c['tau_b_se']:.5f}),"
                 f" analytic={c['tau_b_claim_analytic']:+.5f}, ratio="
                 f"{c['ratio_sim']:.4f}; {c['n_claims']:,} claims")
    # odd in tau_lat (sign symmetry); c = slope at the origin, where the target
    # lies; the cubic term absorbs the mild curvature over [-0.40, -0.05]
    c_slope = float(c1)
    tau_lat_star = tau_hat / c_slope
    rho_star = float(np.sin(np.pi * tau_lat_star / 2))
    # exact analytic root (cross-check)
    from scipy.optimize import brentq
    f_an = lambda tl: claim_tau_b_gauss(np.sin(np.pi * tl / 2), nbt) - tau_hat
    tau_lat_star_an = float(brentq(f_an, -0.2, -1e-6)) if tau_hat < 0 else \
        float(brentq(f_an, 1e-6, 0.2))
    c_an = float(claim_tau_b_gauss(np.sin(np.pi * 1e-3 / 2), nbt) / 1e-3)
    log.info(f"  Regression tau_claim = c1 tau_lat + c3 tau_lat^3: c1={c1:.5f},"
             f" c3={c3:+.4f}; through-origin linear slope={c_lin:.5f}; "
             f"analytic slope at 0 = {c_an:.5f}")
    log.info(f"  >>> tau_lat* = tau_hat / c = {tau_hat:+.6f} / {c_slope:.5f} = "
             f"{tau_lat_star:+.6f};  rho* = sin(pi tau_lat*/2) = {rho_star:+.6f}"
             f"  (exact analytic root {tau_lat_star_an:+.6f}; pre-R1 "
             f"de-attenuated {R1_OLD_TAU_DEATT:+.4f})")

    # scenarios
    scen_main, H_main = _gauss_scen("A_gauss_calibrated", tau_lat_star, nbt,
                                    zgrid)
    scen_emp = ("emp", "B_empirical_rank")
    sweep_scens = {}
    for tl in R1_SWEEP_TAU_LAT:
        sc, _ = _gauss_scen(f"sweep_{tl:+.2f}", tl, nbt, zgrid)
        sweep_scens[tl] = sc
    log.info(f"  H_rho* normalisation: quadrature sum n*P(N=n)="
             f"{H_main['norm_quad']:.8f} vs E[N]={H_main['norm_EN']:.8f}")

    # large-sample checks (claim-level tau, margin) for main, (B), sweep (+)
    log.info("\n[4/8] Large-portfolio checks: claim-level tau-b and "
             "Uniform(0,1) of the margin-corrected V'")
    chk_tasks = [("A_gauss_calibrated", scen_main, R1_CHECK_N_POL,
                  R1_SEED_CHECK),
                 ("B_empirical_rank", scen_emp, R1_CHECK_N_POL,
                  R1_SEED_CHECK + 1)]
    for tl in R1_SWEEP_TAU_LAT:
        if tl in R1_CALIB_TAU_LAT:
            continue
        chk_tasks.append((sweep_scens[tl][1], sweep_scens[tl], R1_VALID_N_POL,
                          R1_SEED_CALIB))
    chk = {c["name"]: c for c in run_tasks(_task_large, chk_tasks, ctx, pool)}
    for c in cal:                                   # reuse calibration runs
        chk[f"sweep_{c['tau_lat']:+.2f}"] = c
    for nm in ["A_gauss_calibrated", "B_empirical_rank"] + \
            [f"sweep_{tl:+.2f}" for tl in R1_SWEEP_TAU_LAT]:
        c = chk[nm]
        log.info(f"  {nm:>20s}: tau-b(N,Y_dep)={c['tau_b_claim']:+.5f} "
                 f"(s.e. {c['tau_b_se']:.5f}); KS(V' vs U) D={c['ks_Vcorr_D']:.5f}"
                 f" p={c['ks_Vcorr_p']:.3f}; uncorrected V: D={c['ks_Vraw_D']:.5f}"
                 f" p={c['ks_Vraw_p']:.2g}; mean Y dep/ind="
                 f"{c['mean_Y_ratio']:.5f}; {c['n_claims']:,} claims")

    # ------------------------------------------------------------ paired MC --
    log.info(f"\n[5/8] Paired compound MC: main (A) {N_MC_PAIRED:,} reps in "
             f"{CAPITAL_SEED_BLOCKS} seed blocks; (B) {R1_EMP_N_REP:,} reps in "
             f"{R1_EMP_BLOCKS} blocks; sweep {len(R1_SWEEP_TAU_LAT)} x "
             f"{R1_SWEEP_N_REP:,} reps in {R1_SWEEP_BLOCKS} blocks; "
             f"{N_POLICIES_SIM:,} policies each; "
             f"{R1_N_WORKERS} workers")
    block = N_MC_PAIRED // CAPITAL_SEED_BLOCKS
    emp_block = R1_EMP_N_REP // R1_EMP_BLOCKS
    sw_block = R1_SWEEP_N_REP // R1_SWEEP_BLOCKS
    tasks = []
    for b in range(CAPITAL_SEED_BLOCKS):
        for r0 in range(0, block, R1_CHUNK_REPS):
            r1 = min(r0 + R1_CHUNK_REPS, block)
            sc = [scen_main]
            if b < R1_EMP_BLOCKS and r0 < emp_block:
                sc.append(scen_emp)
            if b < R1_SWEEP_BLOCKS and r0 < sw_block:
                sc += [sweep_scens[tl] for tl in R1_SWEEP_TAU_LAT]
            tasks.append((b, r0, r1, sc))
    tasks.sort(key=lambda t: -len(t[3]))            # heavy tasks first
    t0 = time.time()
    res = run_tasks(_task_paired, tasks, ctx, pool)
    log.info(f"  {len(tasks)} tasks done in {time.time() - t0:,.0f} s")
    res.sort(key=lambda r: (r[0], r[1]))
    names = [scen_main[1], scen_emp[1]] + [sweep_scens[tl][1]
                                           for tl in R1_SWEEP_TAU_LAT]
    ind_by_block = {b: [] for b in range(CAPITAL_SEED_BLOCKS)}
    dep_by_block = {nm: {b: [] for b in range(CAPITAL_SEED_BLOCKS)}
                    for nm in names}
    for b, r0, S_ind, S_dep in res:
        ind_by_block[b].append(S_ind)
        for nm, arr in S_dep.items():
            dep_by_block[nm][b].append(arr)
    ind_blocks = {b: np.concatenate(v) for b, v in ind_by_block.items()}

    def arms(nm, blocks, n_per_block):
        d = [np.concatenate(dep_by_block[nm][b])[:n_per_block] for b in blocks]
        i = [ind_blocks[b][:n_per_block] for b in blocks]
        return d, i

    # ------------------------------------------------------------- metrics --
    log.info("\n[6/8] Risk metrics, capital distance K = VaR99.5 - E[S], "
             "paired-bootstrap CIs, seed blocks, Kupiec")
    a_s = SOLVENCY_ALPHA
    scen_out = {}
    agg_store = {}
    specs = [("A_gauss_calibrated", list(range(CAPITAL_SEED_BLOCKS)), block,
              GLOBAL_SEED + 99),
             ("B_empirical_rank", list(range(R1_EMP_BLOCKS)), emp_block,
              GLOBAL_SEED + 199)]
    for nm, blocks, npb, bseed in specs:
        d_bl, i_bl = arms(nm, blocks, npb)
        agg_dep = np.concatenate(d_bl); agg_ind = np.concatenate(i_bl)
        agg_store[nm] = (agg_dep, agg_ind)
        pm = paired_metrics(agg_dep, agg_ind)
        ci = paired_bootstrap(agg_dep, agg_ind, CAPITAL_SCR_BOOT, bseed)
        blk = [paired_metrics(d, i) for d, i in zip(d_bl, i_bl)]
        half = len(agg_dep) // 2
        var_split = float(np.quantile(agg_dep[:half], a_s))
        kup = kupiec_pof(agg_dep[half:], var_split, a_s)
        kup.update(split_sample_exchangeable_p(half, len(agg_dep) - half, a_s,
                                               kup["violations"]))
        scen_out[nm] = {"metrics": pm, "ci95": ci,
                        "seed_blocks_VaR995_change_pct":
                            [b_[f"VaR_change_pct_{a_s}"] for b_ in blk],
                        "seed_blocks_K995_change_pct":
                            [b_[f"K_change_pct_{a_s}"] for b_ in blk],
                        "block_seeds": [GLOBAL_SEED + 1000 * b for b in blocks],
                        "n_rep": int(len(agg_dep)), "kupiec": kup,
                        "claim_check": chk[nm]}
        log.info(f"  {nm}: E[S] dep={pm['ES_dep']:,.0f} ind={pm['ES_ind']:,.0f}"
                 f" ({pm['ES_change_pct']:+.4f}%, CI [{ci['ES_change_pct'][0]:+.4f},"
                 f" {ci['ES_change_pct'][1]:+.4f}])")
        for a in VAR_LEVELS:
            log.info(f"    alpha={a}: VaR dep={pm[f'VaR_dep_{a}']:,.0f} ind="
                     f"{pm[f'VaR_ind_{a}']:,.0f} change={pm[f'VaR_change_pct_{a}']:+.4f}%"
                     f" [{ci[f'VaR_change_pct_{a}'][0]:+.4f}, "
                     f"{ci[f'VaR_change_pct_{a}'][1]:+.4f}]; TVaR change="
                     f"{pm[f'TVaR_change_pct_{a}']:+.4f}%; K change="
                     f"{pm[f'K_change_pct_{a}']:+.3f}% [{ci[f'K_change_pct_{a}'][0]:+.3f},"
                     f" {ci[f'K_change_pct_{a}'][1]:+.3f}]")
        log.info("    seed blocks VaR99.5 change: "
                 + ", ".join(f"{v:+.4f}" for v in
                             scen_out[nm]["seed_blocks_VaR995_change_pct"])
                 + "; K change: " + ", ".join(
                     f"{v:+.3f}" for v in scen_out[nm]["seed_blocks_K995_change_pct"]))
        log.info(f"    Kupiec split-sample: threshold from first {half:,} reps, "
                 f"{kup['violations']} exceedances in the second {half:,} "
                 f"(expected {kup['expected']:.0f}), Kupiec p={kup['p']:.3f}; "
                 f"exact p with estimated threshold={kup['p_exchangeable']:.3f} "
                 f"(sd {kup['sd_exchangeable']:.1f} vs binomial "
                 f"{kup['sd_binomial']:.1f})")

    # sweep
    sweep_rows = []
    for j, tl in enumerate(R1_SWEEP_TAU_LAT):
        nm = sweep_scens[tl][1]
        d_bl, i_bl = arms(nm, list(range(R1_SWEEP_BLOCKS)), sw_block)
        d = np.concatenate(d_bl); i = np.concatenate(i_bl)
        blk = [paired_metrics(dd, ii) for dd, ii in zip(d_bl, i_bl)]
        pm = paired_metrics(d, i)
        ci = paired_bootstrap(d, i, CAPITAL_SCR_BOOT, GLOBAL_SEED + 300 + j)
        c = chk[nm]
        sweep_rows.append({
            "tau_lat": tl, "rho": sweep_scens[tl][2],
            "tau_b_claim_sim": c["tau_b_claim"], "tau_b_claim_se": c["tau_b_se"],
            "tau_b_claim_analytic": claim_tau_b_gauss(sweep_scens[tl][2], nbt),
            "n_rep": int(len(d)), "n_pol": N_POLICIES_SIM,
            "ES_change_pct": pm["ES_change_pct"],
            "ES_change_lo": ci["ES_change_pct"][0],
            "ES_change_hi": ci["ES_change_pct"][1],
            "VaR995_change_pct": pm[f"VaR_change_pct_{a_s}"],
            "VaR995_change_lo": ci[f"VaR_change_pct_{a_s}"][0],
            "VaR995_change_hi": ci[f"VaR_change_pct_{a_s}"][1],
            "TVaR995_change_pct": pm[f"TVaR_change_pct_{a_s}"],
            "K995_change_pct": pm[f"K_change_pct_{a_s}"],
            "K995_change_lo": ci[f"K_change_pct_{a_s}"][0],
            "K995_change_hi": ci[f"K_change_pct_{a_s}"][1],
            "K995_dep": pm[f"K_dep_{a_s}"], "K995_ind": pm[f"K_ind_{a_s}"],
            "blocks_K995_change_pct": ";".join(
                f"{b_[f'K_change_pct_{a_s}']:+.3f}" for b_ in blk),
            "ks_Vcorr_p": c["ks_Vcorr_p"], "mean_Y_ratio_claim": c["mean_Y_ratio"]})
    # --------------------------------------------------------- analytic (D) --
    log.info("\n[7/8] (D) Analytic first-order cross-checks of Var(S_i)")
    ana_pol = load_proc("analytical_dataset")["n_claims"].to_numpy(int)
    pi_emp = np.bincount(ana_pol) / len(ana_pol)
    del ana_pol
    pi_nb = nbt["pi"]
    D = {}
    D["D1_class_empirical_policy_counts"] = empirical_moment_check(
        sev, ncl, pi_emp, "empirical class moments; empirical policy count "
        "distribution (auditor's calculation)")
    D["D1_class_NB"] = empirical_moment_check(
        sev, ncl, pi_nb, "empirical class moments; fitted NB (capped)")
    D["D1_exact_count_NB"] = empirical_moment_check(
        sev, ncl, pi_nb, "empirical moments by exact count (5+ pooled); "
        "fitted NB", exact_upto=5)
    if spm is not None:
        D["D2_model_B_empirical_rank"] = emp_policy_moments(emp, nbt, spm)
        D["D3_model_A_gauss_calibrated"] = gauss_policy_moments(
            rho_star, nbt, zgrid, H_main, spm)
        D["D3_model_A_rho0_check"] = gauss_policy_moments(
            0.0, nbt, zgrid, gauss_H(0.0, nbt, zgrid), spm)
    for k, v in D.items():
        log.info(f"  {k}: Var ratio dep/ind={v['var_ratio']:.5f}, SD ratio="
                 f"{v['sd_ratio']:.5f} (SD change {(v['sd_ratio'] - 1) * 100:+.3f}%),"
                 f" E[S_i] ratio={v['mean_ratio']:.6f}")
    for k in ("D1_class_NB",):
        log.info("    class means 1/2/3/4+: " + ", ".join(
            f"{x:,.0f}" for x in D[k]["class_mean"].values()))
    if "D2_model_B_empirical_rank" in D:
        log.info("    (B) model class means 1/2/3/4+: " + ", ".join(
            f"{x:,.0f}" for x in D["D2_model_B_empirical_rank"]["mu_by_class"]))
    # analytic curve for the sweep figure
    curve = run_tasks(_task_curve, [(tl,) for tl in R1_CURVE_TAU_LAT], ctx, pool)
    curve = sorted(curve, key=lambda r: r["tau_lat"])
    for row in sweep_rows:
        cr = _task_curve((row["tau_lat"],))
        row["K995_change_pred_firstorder_pct"] = cr["K_change_pred_pct"]

    # ------------------------------------------------------------- outputs --
    log.info("\n[8/8] Outputs")
    main_nm = "A_gauss_calibrated"
    pm = scen_out[main_nm]["metrics"]; ci = scen_out[main_nm]["ci95"]
    rows = []
    for a in VAR_LEVELS:
        rows.append({
            "alpha": a, "VaR_dep": pm[f"VaR_dep_{a}"], "VaR_ind": pm[f"VaR_ind_{a}"],
            "VaR_bias_pct": pm[f"VaR_change_pct_{a}"],
            "VaR_bias_lo": ci[f"VaR_change_pct_{a}"][0],
            "VaR_bias_hi": ci[f"VaR_change_pct_{a}"][1],
            "TVaR_dep": pm[f"TVaR_dep_{a}"], "TVaR_ind": pm[f"TVaR_ind_{a}"],
            "TVaR_bias_pct": pm[f"TVaR_change_pct_{a}"],
            "TVaR_bias_lo": ci[f"TVaR_change_pct_{a}"][0],
            "TVaR_bias_hi": ci[f"TVaR_change_pct_{a}"][1],
            "ES_dep": pm["ES_dep"], "ES_ind": pm["ES_ind"],
            "K_dep": pm[f"K_dep_{a}"], "K_ind": pm[f"K_ind_{a}"],
            "K_bias_pct": pm[f"K_change_pct_{a}"],
            "K_bias_lo": ci[f"K_change_pct_{a}"][0],
            "K_bias_hi": ci[f"K_change_pct_{a}"][1]})
    pd.DataFrame(rows).to_csv(OUTPUTS_DIR / "13_capital.csv", index=False)
    outfiles.append(str((OUTPUTS_DIR / "13_capital.csv").resolve()))

    summ = []
    for nm, meta in (("A_gauss_calibrated",
                      {"construction": "Gaussian count-coupled, margin-corrected"
                                       " (H_rho), rho* calibrated to tau_hat",
                       "tau_lat": tau_lat_star, "rho": rho_star}),
                     ("B_empirical_rank",
                      {"construction": "copula-free empirical conditional rank "
                                       "distribution by count class 1/2/3/4+, "
                                       "margin-corrected (H_emp)",
                       "tau_lat": np.nan, "rho": np.nan})):
        so = scen_out[nm]; m_ = so["metrics"]; c_ = so["ci95"]
        row = {"scenario": nm, **meta, "tau_target": tau_hat,
               "tau_b_claim_sim": so["claim_check"]["tau_b_claim"],
               "tau_b_claim_se": so["claim_check"]["tau_b_se"],
               "tau_b_claim_analytic": (claim_tau_b_gauss(rho_star, nbt)
                                        if nm.startswith("A")
                                        else emp["tau_b_analytic"]),
               "ks_Vcorr_p": so["claim_check"]["ks_Vcorr_p"],
               "n_rep": so["n_rep"], "n_blocks": len(so["block_seeds"]),
               "n_pol": N_POLICIES_SIM}
        for k in ("ES_dep", "ES_ind", "ES_change_pct"):
            row[k] = m_[k]
        row["ES_change_lo"], row["ES_change_hi"] = c_["ES_change_pct"]
        for a in VAR_LEVELS:
            for mm in ("VaR", "TVaR"):
                row[f"{mm}_dep_{a}"] = m_[f"{mm}_dep_{a}"]
                row[f"{mm}_ind_{a}"] = m_[f"{mm}_ind_{a}"]
                row[f"{mm}_change_pct_{a}"] = m_[f"{mm}_change_pct_{a}"]
                row[f"{mm}_change_lo_{a}"], row[f"{mm}_change_hi_{a}"] = \
                    c_[f"{mm}_change_pct_{a}"]
        row["K995_dep"] = m_[f"K_dep_{a_s}"]; row["K995_ind"] = m_[f"K_ind_{a_s}"]
        row["K995_change_pct"] = m_[f"K_change_pct_{a_s}"]
        row["K995_change_lo"], row["K995_change_hi"] = c_[f"K_change_pct_{a_s}"]
        for j, (v1, v2) in enumerate(zip(so["seed_blocks_VaR995_change_pct"],
                                         so["seed_blocks_K995_change_pct"])):
            row[f"block{j + 1}_VaR995_change_pct"] = v1
            row[f"block{j + 1}_K995_change_pct"] = v2
        row.update({"kupiec_violations": so["kupiec"]["violations"],
                    "kupiec_expected": so["kupiec"]["expected"],
                    "kupiec_p": so["kupiec"]["p"],
                    "split_p_exchangeable": so["kupiec"]["p_exchangeable"]})
        dk = ("D3_model_A_gauss_calibrated" if nm.startswith("A")
              else "D2_model_B_empirical_rank")
        if dk in D:
            row["K995_change_pred_firstorder_pct"] = D[dk]["K_change_pred_pct"]
        summ.append(row)
    pd.DataFrame(summ).to_csv(OUTPUTS_DIR / "13_capital_summary.csv", index=False)
    outfiles.append(str((OUTPUTS_DIR / "13_capital_summary.csv").resolve()))
    pd.DataFrame(sweep_rows).to_csv(OUTPUTS_DIR / "13_tau_sweep.csv", index=False)
    outfiles.append(str((OUTPUTS_DIR / "13_tau_sweep.csv").resolve()))

    # --------------------------------------------------------------- figure --
    make_figure(agg_store[main_nm], scen_out, sweep_rows, curve, tau_hat,
                tau_ci, outfiles)

    # ------------------------------------------------------------- manifest --
    # old de-attenuation value via the 09 recovery curve (comparison only)
    tau_deatt_09 = None
    rec_path = OUTPUTS_DIR / "09_recovery_table.csv"
    if rec_path.exists():
        try:
            rec = pd.read_csv(rec_path).sort_values("tau_inject")
            xs = np.maximum.accumulate(
                rec["tau_recovered_perclaim"].values.astype(float))
            ys = rec["tau_inject"].values.astype(float)
            ta = float(np.interp(abs(tau_hat), xs, ys))
            tau_deatt_09 = float(np.sign(tau_hat) * max(ta, abs(tau_hat)))
        except Exception:
            tau_deatt_09 = None
    so = scen_out[main_nm]
    so_b = scen_out["B_empirical_rank"]
    manifest = {
        "script": SCRIPT_NAME,
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        # --- keys read by 20_synthesis (kept) ---
        "tau_dependence": float(tau_hat),
        "scr_dependent": pm[f"VaR_dep_{a_s}"],
        "scr_independent": pm[f"VaR_ind_{a_s}"],
        "scr_bias_pct": pm[f"VaR_change_pct_{a_s}"],
        "scr_bias_mc_ci95": ci[f"VaR_change_pct_{a_s}"],
        "scr_bias_seed_blocks": so["seed_blocks_VaR995_change_pct"],
        "n_paired_total": so["n_rep"],
        "n_tail_obs": int(so["n_rep"] * (1 - a_s)),
        "deattenuated_scenario": {
            "status": "superseded (R1): the de-attenuation via the inverse of "
                      "the 09 positive-control recovery curve is no longer used;"
                      " the latent tau is calibrated directly so that the "
                      "simulated claim-level tau-b equals tau_hat",
            "used_09_recovery_curve": False,
            "tau_deattenuated": None, "scr_bias_pct": None,
            "tau_deattenuated_09curve_for_comparison": tau_deatt_09,
            "tau_deattenuated_pre_R1_run": R1_OLD_TAU_DEATT,
            "tau_lat_star_direct_calibration": tau_lat_star,
            "rho_star_direct_calibration": rho_star},
        "risk_metrics_dependent": {f"{m}_{a}": pm[f"{m}_dep_{a}"]
                                   for a in VAR_LEVELS for m in ("VaR", "TVaR")},
        "risk_metrics_independent": {f"{m}_{a}": pm[f"{m}_ind_{a}"]
                                     for a in VAR_LEVELS for m in ("VaR", "TVaR")},
        "coverage_check": {**so["kupiec"],
                           "design": "split-sample (threshold: seed blocks 1-2;"
                                     " test: blocks 3-4), dependent arm, main "
                                     "scenario",
                           "note": "simulation-internal quantile coverage check;"
                                   " NOT an out-of-sample backtest"},
        "copula_family_sensitivity": {
            "main_family": "gauss_count_coupled_margin_corrected",
            "main_pct": pm[f"VaR_change_pct_{a_s}"],
            "main_K_pct": pm[f"K_change_pct_{a_s}"],
            "alt_family": "empirical_conditional_rank (copula-free, count "
                          "classes 1/2/3/4+)",
            "alt_pct": so_b["metrics"][f"VaR_change_pct_{a_s}"],
            "alt_K_pct": so_b["metrics"][f"K_change_pct_{a_s}"],
            "alt_n_rep": so_b["n_rep"],
            "note": "pre-R1 Frank-copula run removed (it used the "
                    "count-independent latent design)"},
        "gpd_tail_q_used": float(gpd.get("q", SEV_TAIL_Q)) if gpd else None,
        "design_choices": {
            "frequency_severity_leakage": False,
            "paired": True,
            "leakage_note": "no shared-uniform leakage (Proposition 2 guard): "
                            "the count uniform is never reused as a severity "
                            "uniform; it enters the dependent arm only through "
                            "the calibrated Gaussian link rho*Phi^-1(U_i) (A) or"
                            " the count class (B), and the claim-level severity "
                            "margin is re-imposed exactly by H",
            "count_coupled_dependent_arm": True,
            "pre_R1_design_removed": "policy latent independent of the count "
                                     "uniform (N independent of every severity;"
                                     " O(rho^2) within-policy coupling only)",
            "margin_correction": "V' = H(V), H = claim-weighted CDF of V "
                                 "(A: Gauss-Legendre quadrature in normal-score "
                                 "space, %d nodes per count interval, grid z in "
                                 "[-%.0f, %.0f] with %d points, linear "
                                 "interpolation of Phi^-1(H) - monotone; B: "
                                 "piecewise-linear exact)" % (
                                     R1_GL_NODES, R1_ZGRID_MAX, R1_ZGRID_MAX,
                                     R1_ZGRID_N),
            "count_cap": CAPITAL_NMAX,
            "n_policies": N_POLICIES_SIM,
            "capital_distance": "K = VaR_0.995 - E[S] (arm-specific sample "
                                "mean); change = K_ind/K_dep - 1",
            "change_convention": "X_ind / X_dep - 1 (positive = independence "
                                 "gives the higher figure), as pre-R1",
            "bootstrap": f"paired-replication percentile bootstrap, "
                         f"B={CAPITAL_SCR_BOOT}",
            "seeding": "replication r of seed block b uses "
                       "np.random.default_rng([GLOBAL_SEED + 1000 b, r]); the "
                       "same streams are shared by all scenarios (common "
                       "random numbers): the main run, B and every sweep point"
                       " use the same 4 blocks x 5,000 replications, so their "
                       "independent arms are identical",
            "n_workers": R1_N_WORKERS, "chunk_reps": R1_CHUNK_REPS,
            "severity_quantile_fast_vs_scipy_max_rel_diff": sev_ppf_maxrel,
            "count_inversion_mismatches_vs_scipy_ppf": count_mismatch},
        "calibration": {
            "method": "large simulated portfolios with common random numbers "
                      "at tau_lat in %s; regression tau_claim = c1 tau_lat + "
                      "c3 tau_lat^3 (odd by sign symmetry); c = c1 = slope at "
                      "the origin; tau_lat* = tau_hat / c" % (
                          list(R1_CALIB_TAU_LAT),),
            "n_pol_per_point": R1_CALIB_N_POL, "seed": R1_SEED_CALIB,
            "points": [{k: c[k] for k in ("tau_lat", "rho", "tau_b_claim",
                                          "tau_b_se", "tau_b_claim_analytic",
                                          "ratio_sim", "n_claims")}
                       for c in cal],
            "c": c_slope, "c3": float(c3), "c_linear_through_origin": c_lin,
            "c_analytic_slope_at_0": c_an,
            "tau_hat": tau_hat, "tau_hat_recomputed": tau_data,
            "tau_lat_star": tau_lat_star, "rho_star": rho_star,
            "tau_lat_star_analytic_root": tau_lat_star_an,
            "check_tau_b_claim_at_rho_star": chk[main_nm]["tau_b_claim"],
            "check_tau_b_se": chk[main_nm]["tau_b_se"],
            "check_n_pol": R1_CHECK_N_POL, "check_seed": R1_SEED_CHECK,
            "check_tau_b_claim_analytic_at_rho_star":
                claim_tau_b_gauss(rho_star, nbt),
            "old_tau_deattenuated_for_comparison": R1_OLD_TAU_DEATT},
        "margin_correction_validation": {nm: chk[nm] for nm in chk},
        "scenarios": {nm: {k: v for k, v in so_.items() if k != "claim_check"}
                      for nm, so_ in scen_out.items()},
        "empirical_scenario": {
            "count_classes": "1, 2, 3, 4+",
            "pit": "pooled mid-rank PIT cells ((r-1)/n, r/n]; R_k = "
                   "checkerboard (cell-uniform) conditional rank distribution;"
                   " tied severities spread over their joint mid-rank cell",
            "n_claims": emp["n"], "tau_b_data": tau_data,
            "claim_share_data": [float(x) for x in emp["w_data"]],
            "claim_share_NB": [float(x) for x in emp["w_sim"]],
            "n_tied_claims": emp["n_tied_claims"],
            "max_tie_size": emp["max_tie_size"],
            "tau_b_claim_analytic_population": emp["tau_b_analytic"],
            "n_rep": R1_EMP_N_REP, "n_blocks": R1_EMP_BLOCKS},
        "tau_sweep": {"tau_lat_grid": list(R1_SWEEP_TAU_LAT),
                      "n_rep_per_point": R1_SWEEP_N_REP,
                      "n_blocks": R1_SWEEP_BLOCKS,
                      "note": "brief specified 4,000 replications per point; 20,000 used (runtime < 2 min) because 4,000 gave capital-distance CIs of +-2 to +-5 percentage points",
                      "rows": sweep_rows,
                      "analytic_first_order_curve": curve},
        "analytic_check": {
            "formula": "Var(S_i) = E[N s2_N] + Var(N mu_N) (+ E[N(N-1) Cov(Y1,Y2"
                       "|N)] for A) vs independence E[N] s2 + Var(N) mu^2; "
                       "portfolio of independent policies; K change "
                       "predicted to first order by the SD ratio",
            **D},
        "old_pre_R1_results": R1_OLD_SCR_BIAS,
        "observed_tau_cluster_ci95": tau_ci,
        "output_files": outfiles,
        "environment": env_info(),
    }
    with open(OUTPUTS_DIR / "13_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=_json_default)
    log.info(f"  written: {', '.join(Path(p).name for p in outfiles)}, "
             f"13_manifest.json")
    return manifest


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# =============================================================================
# Figure 6
# =============================================================================
def make_figure(aggs, scen_out, sweep_rows, curve, tau_hat, tau_ci, outfiles):
    """Figure 6: (a) paired aggregate-loss exceedance curves (main scenario vs
    independence) with the 99.5% quantiles and the capital distance K;
    (b) K change vs the implied claim-level tau-b (Gaussian sweep with MC 95%
    CIs, first-order analytic curve), with a zoom on the observed tau_hat."""
    agg_dep, agg_ind = aggs
    a_s = SOLVENCY_ALPHA
    C_DEP, C_IND, C_EMP, C_GREY = "#1f4e79", "#c0504d", "#2e7d32", "0.45"
    C_BAND = "#f2c14e"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W_FULL, 2.95),
                                   layout="constrained")
    fig.get_layout_engine().set(w_pad=0.04, wspace=0.06)
    # (a) exceedance probabilities (upper half) of the paired aggregates
    sc = 1e-6
    ex = np.geomspace(0.5, 1.0 / len(agg_dep), 500)
    for arr, col, ls, lab in ((agg_ind, C_IND, "--", "Independence"),
                              (agg_dep, C_DEP, "-", "Dependent (main)")):
        ax1.plot(np.quantile(arr, 1.0 - ex) * sc, ex, color=col, ls=ls,
                 lw=1.0, label=lab)
    # Data S1 (R1 final): data of the curves drawn in panel (a) — a NEW file
    # only; the simulation, seeds and existing outputs are NOT AFFECTED (synthetic)
    fp6a = OUTPUTS_DIR / "13_fig6a_exceedance_curves.csv"
    pd.DataFrame({"exceedance_prob": ex,
                  "aggregate_loss_dependent_TRY": np.quantile(agg_dep, 1.0 - ex),
                  "aggregate_loss_independent_TRY": np.quantile(agg_ind, 1.0 - ex)}
                 ).to_csv(fp6a, index=False)
    outfiles.append(str(fp6a.resolve()))
    pm = scen_out["A_gauss_calibrated"]["metrics"]
    ci = scen_out["A_gauss_calibrated"]["ci95"]
    v_d, v_i = pm[f"VaR_dep_{a_s}"] * sc, pm[f"VaR_ind_{a_s}"] * sc
    es = pm["ES_dep"] * sc
    ax1.set_yscale("log")
    ax1.set_ylim(0.5 / len(agg_dep), 0.8)
    ax1.axhline(1 - a_s, color=C_GREY, lw=0.6, ls=":")
    ax1.axvline(es, color=C_GREY, lw=0.6, ls=":")
    ax1.plot([v_d], [1 - a_s], "o", ms=3.5, color=C_DEP, zorder=5)
    ax1.plot([v_i], [1 - a_s], "s", ms=4.5, mfc="none", color=C_IND, zorder=5)
    ax1.annotate("", xy=(v_d, 2.0e-3), xytext=(es, 2.0e-3),
                 arrowprops=dict(arrowstyle="<->", lw=0.7, color="0.2"))
    ax1.text(0.5 * (es + v_d), 2.4e-3, r"K = VaR$_{99.5\%}$ − E[S]",
             ha="center", va="bottom", fontsize=6.5)
    ax1.text(es, 0.75, " E[S]", ha="left", va="top", fontsize=6.5, color="0.3")
    ax1.text(es + 0.1, (1 - a_s) * 1.15, "99.5% level",
             fontsize=6.5, color="0.3", va="bottom", ha="left")
    kc, klo, khi = (pm[f"K_change_pct_{a_s}"], ci[f"K_change_pct_{a_s}"][0],
                    ci[f"K_change_pct_{a_s}"][1])
    ax1.text(0.97, 0.97,
             f"99.5% VaR (TRY million)\ndependent {v_d:,.2f}\n"
             f"independence {v_i:,.2f}\n"
             f"K change {kc:+.2f}%\n[{klo:+.2f}%, {khi:+.2f}%]",
             transform=ax1.transAxes, ha="right", va="top", fontsize=6.5,
             linespacing=1.25)
    ax1.set_xlabel("Aggregate loss S (TRY million)")
    ax1.set_ylabel("Exceedance probability P(S > s)")
    ax1.set_title("(a) Paired aggregate losses, upper half")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2,
               fontsize=6.5, handlelength=2.2, columnspacing=1.2)

    # (b) sweep: K change vs implied claim-level tau
    xs = np.array([r["tau_b_claim_sim"] for r in sweep_rows])
    ys = np.array([r["K995_change_pct"] for r in sweep_rows])
    lo = np.array([r["K995_change_lo"] for r in sweep_rows])
    hi = np.array([r["K995_change_hi"] for r in sweep_rows])
    cx = np.array([c["tau_b_claim_analytic"] for c in curve])
    cy = np.array([c["K_change_pred_pct"] for c in curve])
    pts = []
    for nm in ("A_gauss_calibrated", "B_empirical_rank"):
        so = scen_out[nm]
        yv = so["metrics"][f"K_change_pct_{a_s}"]
        l_, h_ = so["ci95"][f"K_change_pct_{a_s}"]
        pts.append((so["claim_check"]["tau_b_claim"], yv, l_, h_))
    sty = {"A_gauss_calibrated": dict(fmt="D", ms=3.6, color=C_DEP,
                                      mfc="white", label="Main, calibrated"),
           "B_empirical_rank": dict(fmt="^", ms=3.8, color=C_EMP,
                                    label="Empirical ranks")}

    def draw(ax, sweep):
        ax.axvspan(tau_ci[0], tau_ci[1], color=C_BAND, alpha=0.5, lw=0,
                   label=r"Observed $\hat\tau$, 95% CI")
        ax.axhline(0, color=C_GREY, lw=0.6)
        ax.plot(cx, cy, color="0.5", lw=0.8, ls="--",
                label="First-order analytic")
        if sweep:
            ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], fmt="o", ms=3,
                        color=C_DEP, ecolor=C_DEP, elinewidth=0.8, capsize=1.5,
                        label="Gaussian sweep (MC 95% CI)")
            return
        for (xv, yv, l_, h_), nm in zip(pts, sty):
            ax.errorbar([xv], [yv], yerr=[[yv - l_], [h_ - yv]],
                        ecolor=sty[nm]["color"], elinewidth=0.8, capsize=1.5,
                        zorder=6, **sty[nm])

    draw(ax2, True)
    for nm in sty:                                   # legend proxies
        ax2.errorbar([np.nan], [np.nan], yerr=[[np.nan], [np.nan]],
                     ecolor=sty[nm]["color"], elinewidth=0.8, capsize=1.5,
                     **sty[nm])
    ax2.set_xlim(-0.33, 0.33)
    ax2.set_ylim(min(lo.min(), cy.min()) - 1.0, max(hi.max(), cy.max()) + 1.5)
    ax2.set_xlabel(r"Claim-level Kendall $\tau_b(N, Y)$")
    ax2.set_ylabel("K change from ignoring\ndependence (%)")
    ax2.set_title("(b) Capital distance vs dependence strength")
    h, l = ax2.get_legend_handles_labels()
    order = [l.index(k) for k in (r"Observed $\hat\tau$, 95% CI",
                                  "First-order analytic",
                                  "Gaussian sweep (MC 95% CI)",
                                  "Main, calibrated", "Empirical ranks")]
    ax2.legend([h[i] for i in order], [l[i] for i in order],
               loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3,
               fontsize=6.2, handlelength=1.6, columnspacing=0.9)
    # zoom on the observed tau_hat
    axin = ax2.inset_axes([0.07, 0.07, 0.40, 0.42])
    draw(axin, False)
    xlo = min(tau_ci[0], min(p_[0] for p_ in pts)) - 0.0025
    xhi = max(tau_ci[1], max(p_[0] for p_ in pts)) + 0.0025
    ylo = min(p_[2] for p_ in pts); yhi = max(p_[3] for p_ in pts)
    pad = 0.15 * (yhi - ylo)
    axin.set_xlim(xlo, xhi); axin.set_ylim(ylo - pad, yhi + pad)
    axin.tick_params(labelsize=5.5, length=2, pad=1)
    axin.set_title(r"zoom at $\hat\tau$", fontsize=6, pad=2)
    axin.grid(alpha=0.2)
    for sp_ in axin.spines.values():
        sp_.set_visible(True); sp_.set_linewidth(0.5)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"13_aggregate.{ext}")
    plt.close(fig)
    outfiles.append(str((FIGURES_DIR / "13_aggregate.pdf").resolve()))
    outfiles.append(str((FIGURES_DIR / "13_aggregate.png").resolve()))


if __name__ == "__main__":
    main()
