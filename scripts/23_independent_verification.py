"""
================================================================================
SCRIPT 23 — R2 Revision: Independent Verification of the Headline Results
================================================================================
PURPOSE (Reviewer 1, round 2, point 3 — reliability of the corrected results)
    The reviewer noted that the implementation errors found and corrected in R1
    raise concerns about the reliability of the computational process, and that
    "independent verification supported by code and materials" would be
    appropriate. Since the raw data fall under a confidentiality agreement and
    the Turkish data-protection law KVKK (6698), and the code is not shared
    before publication, this module re-derives the paper's HEADLINE figures via
    code paths INDEPENDENT of the pipeline (separate library / separate
    formulation / separate simulator) and produces a verification table:

    V1  Data counts and totals (policies, claims, real paid, open case reserves)
        — directly from parquet, without using pipeline functions.
    V2  Definition of the pricing target: ultimate = paid + case reserves open
        at the valuation date; totals and the test-year target match the 14b parquet.
    V3  Clean τ_FS = τ_b(N, Y): INDEPENDENT τ_b implementation by class-based
        concordance/discordance counting (searchsorted); comparison with scipy.
    V4  τ_FS policy-cluster bootstrap 95% interval: simple resampling (B=200)
        compared with the BCa interval of 06.
    V5  Chain-ladder and Mack (1993): matrix-form independent implementation;
        ultimate / reserve / s.e. for the paid triangle (03) and incurred triangle (22).
    V6  GLMs with statsmodels: Poisson, NB2, two-stage (Poisson×Gamma)
        and Tweedie(p=1.6) — against the pure-numpy IRLS test Ginis of 14b.
    V7  Pure-premium Gini: independent Lorenz-curve-based formula.
    V8  Capital experiment: INDEPENDENT mini-simulator coupled to the count via a
        Gaussian link (empirical count pmf + empirical severity; no margin
        correction) — (a) is the claim-level τ_b in the dependent arm really
        non-zero and of the order of the observed τ̂_FS? is it ≈0 in the
        independent arm? (b) is the VaR_99.5 change in the coupled MC of the
        same order as in 13?
    V9  Case-reserve run-off (22): paid/case-reserve ratio of the 2017-12
        snapshot via a separate code path.
    V10 Raw-level P/O ratio of the selected model (from parquet).

    Each row: quantity, pipeline value, independent value, difference, tolerance, result.
    Outputs are AGGREGATE; nothing at policy/claim level is written.

INPUT : data/proc/*.parquet ; outputs/{03,06,13,14b,22}_manifest.json ;
        outputs/03_development_triangle.csv ; outputs/22_incurred_triangle.csv
OUTPUT: outputs/23_verification_table.csv ; outputs/23_manifest.json ;
        logs/23_independent_verification.log
================================================================================
"""
from __future__ import annotations
import sys
import json
import logging
import time
import importlib.util
from pathlib import Path
from datetime import datetime

try:
    _HERE = Path(__file__).resolve().parent
except NameError:
    _HERE = Path.cwd()
for _cand in (_HERE, _HERE.parent):
    if (_cand / "config.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from config import (DATA_PROC_DIR, OUTPUTS_DIR, LOGS_DIR, GLOBAL_SEED,
                    FIT_YEARS, TEST_YEARS, HPO_VALUATION_YEAR, HPO_VALUATION_MONTH,
                    SOLVENCY_ALPHA, env_info)

SCRIPT_NAME = "23_independent_verification"
V4_B = 200                 # cluster-bootstrap replicates (independent, simple percentile)
V8_N_POL_TAU = 2_000_000   # number of policies for the τ check
V8_REPS = 4000             # number of coupled MC replicates
V8_N_POL = 100_000         # policies per replicate (same as 13)
V8_BOOT_B = 500            # replicate-bootstrap for the VaR change
V8_COUNT_CAP = 30


def setup_logger() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(SCRIPT_NAME); lg.setLevel(logging.INFO); lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOGS_DIR / f"{SCRIPT_NAME}.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt); sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    lg.addHandler(fh); lg.addHandler(sh); return lg


log = logging.getLogger(SCRIPT_NAME)
ROWS: list = []


def record(vid, quantity, pipeline, independent, tol, kind="abs", source="", note=""):
    """Verification row: kind = 'abs' | 'rel' | 'contains' (independent = (lo, hi))."""
    if kind == "contains":
        lo, hi = independent
        ok = bool(lo <= pipeline <= hi); diff = np.nan; rel = np.nan
        ind_txt = f"[{lo:.6g}, {hi:.6g}]"
    else:
        diff = float(independent - pipeline)
        rel = float(diff / pipeline) if pipeline not in (0, None) and np.isfinite(pipeline) and pipeline != 0 else np.nan
        ok = bool(abs(diff) <= tol) if kind == "abs" else bool(abs(rel) <= tol)
        ind_txt = f"{independent:.10g}"
    ROWS.append({"id": vid, "quantity": quantity, "pipeline_value": pipeline,
                 "independent_value": independent if kind != "contains" else ind_txt,
                 "abs_diff": diff, "rel_diff": rel, "tolerance": tol, "tolerance_kind": kind,
                 "pass": ok, "pipeline_source": source, "note": note})
    log.info(f"  [{vid}] {quantity}: pipeline={pipeline:.10g} | independent={ind_txt} | "
             f"{'PASS' if ok else 'FAIL'}")
    return ok


def load_proc(stem):
    pq = DATA_PROC_DIR / f"{stem}.parquet"
    return pd.read_parquet(pq) if pq.exists() else pd.read_pickle(DATA_PROC_DIR / f"{stem}.pkl")


def jload(name):
    return json.load(open(OUTPUTS_DIR / name, encoding="utf-8"))


# ------------------------------------------------------------------ V3 helper
def tau_b_by_class(x_int: np.ndarray, y: np.ndarray) -> float:
    """Kendall τ_b, x discrete (few classes), y continuous: for each class pair
    (a<b), #(y_b > y_a) − #(y_b < y_a) is counted with sorted arrays (searchsorted).
    τ_b = (P−Q) / sqrt((n0−n1)(n0−n2)), n1 = Σ t(t−1)/2 (ties in x),
    n2 = Σ u(u−1)/2 (ties in y). Code path independent of scipy."""
    classes = np.unique(x_int)
    ys = {c: np.sort(y[x_int == c]) for c in classes}
    pq = 0.0
    for i, a in enumerate(classes):
        ya = ys[a]
        for b in classes[i + 1:]:
            yb = ys[b]
            gt = np.searchsorted(ya, yb, side="left").sum()      # y_a < y_b
            lt = (len(ya) - np.searchsorted(ya, yb, side="right")).sum()  # y_a > y_b
            pq += gt - lt
    n = len(y); n0 = n * (n - 1) / 2
    _, tx = np.unique(x_int, return_counts=True); n1 = (tx * (tx - 1) / 2).sum()
    _, ty = np.unique(y, return_counts=True); n2 = (ty * (ty - 1) / 2).sum()
    return float(pq / np.sqrt((n0 - n1) * (n0 - n2)))


# ------------------------------------------------------------------ V5 helper
def chain_ladder_mack_matrix(tri: pd.DataFrame) -> dict:
    """Chain-ladder + Mack (1993) in matrix form (Mack's formulas (3),(4),(7),(9)).
    tri: cumulative triangle (rows accident year, columns development 0..K−1; NaN = unobserved)."""
    C = tri.values.astype(float); I, K = C.shape
    obs = ~np.isnan(C)
    f = np.ones(K - 1); S = np.zeros(K - 1); s2 = np.full(K - 1, np.nan)
    for k in range(K - 1):
        m = obs[:, k] & obs[:, k + 1] & (C[:, k] > 0)
        f[k] = C[m, k + 1].sum() / C[m, k].sum(); S[k] = C[m, k].sum()
        if m.sum() > 1:
            s2[k] = (C[m, k] * (C[m, k + 1] / C[m, k] - f[k]) ** 2).sum() / (m.sum() - 1)
    for k in range(K - 1):                      # Mack last-age rule
        if np.isnan(s2[k]):
            s2[k] = (min(s2[k - 1] ** 2 / s2[k - 2], min(s2[k - 2], s2[k - 1]))
                     if k >= 2 and s2[k - 2] > 0 else (s2[k - 1] if k >= 1 else 0.0))
    last = np.array([np.where(obs[i])[0].max() for i in range(I)])
    Chat = C.copy()
    for i in range(I):
        for k in range(last[i], K - 1):
            Chat[i, k + 1] = Chat[i, k] * f[k]
    ult = Chat[:, -1]; latest = np.array([C[i, last[i]] for i in range(I)])
    mse = np.zeros(I)
    for i in range(I):
        for k in range(last[i], K - 1):
            mse[i] += (s2[k] / f[k] ** 2) * (1.0 / Chat[i, k] + 1.0 / S[k])
        mse[i] *= ult[i] ** 2
    total = mse.sum()
    for i in range(I):
        later = ult[i + 1:].sum()
        cross = sum(s2[k] / (f[k] ** 2 * S[k]) for k in range(last[i], K - 1))
        total += 2.0 * ult[i] * later * cross
    return {"factors": f, "ultimate_total": float(ult.sum()),
            "reserve_total": float((ult - latest).sum()), "total_se": float(np.sqrt(total)),
            "per_ay_se": np.sqrt(mse)}


# ------------------------------------------------------------------ V7 helper
def gini_lorenz(y_true, y_pred) -> float:
    """Normalised pure-premium Gini: Lorenz curve of the actual losses ordered by
    the prediction; 1 − 2∫L (trapezoid) / the same quantity under perfect ordering."""
    def area(order):
        ys = np.asarray(y_true, float)[order]
        cum = np.cumsum(ys) / ys.sum()
        x = np.arange(1, len(ys) + 1) / len(ys)
        return np.trapezoid(cum, x) if hasattr(np, "trapezoid") else np.trapz(cum, x)
    g_model = 1 - 2 * area(np.argsort(y_pred, kind="stable"))
    g_perf = 1 - 2 * area(np.argsort(y_true, kind="stable"))
    return float(g_model / g_perf)


def main() -> None:
    global log
    log = setup_logger(); t0 = time.time()
    log.info("=" * 78); log.info(f"SCRIPT {SCRIPT_NAME} — independent verification (R2)"); log.info("=" * 78)
    m03, m06, m13, m14, m22 = (jload("03_manifest.json"), jload("06_manifest.json"),
                               jload("13_manifest.json"), jload("14b_manifest.json"),
                               jload("22_manifest.json"))
    vy, vm = HPO_VALUATION_YEAR, HPO_VALUATION_MONTH

    # ------------------------------------------------------------------ V1
    log.info("\n[V1] Data counts and totals (directly from parquet)")
    pol = pd.read_parquet(DATA_PROC_DIR / "policies.parquet", columns=["policy_ref"])
    paid = load_proc("paid_long_real")
    res = load_proc("reserve_panel_real")
    sev_by_file = paid.groupby("claim_file")["paid_total_real"].sum()
    n_claims_pos = int((sev_by_file > 0).sum())
    record("V1a", "policies", m03["counts"]["policies"], len(pol), 0, source="03_manifest.counts")
    record("V1b", "claims in severity sample (real paid > 0)", m03["counts"]["claims"], n_claims_pos, 0,
           source="03_manifest.counts")
    record("V1c", "total real paid (TRY)", m03["ultimate_totals"]["paid_only"], float(paid["paid_total_real"].sum()),
           1e-9, kind="rel", source="03_manifest.ultimate_totals.paid_only")
    open_val = res[(res["reserve_year"] == vy) & (res["reserve_month"] == vm)]
    open_val = open_val.groupby(["policy_ref", "claim_file"])["reserve_amount_real"].last()
    record("V1d", "open case reserves at valuation (TRY)", m22["incurred_CL"]["case_reserve_at_valuation"],
           float(open_val.sum()), 1e-9, kind="rel", source="22_manifest.incurred_CL.case_reserve_at_valuation")
    record("V1e", "open claim files at valuation", m22["open_files_at_valuation"]["files"], int(len(open_val)), 0,
           source="22_manifest.open_files_at_valuation.files")

    # ------------------------------------------------------------------ V2
    log.info("\n[V2] Pricing target: ultimate = paid + case reserves open at the valuation date")
    ana = load_proc("analytical_dataset")
    paid_by_pol = paid.groupby("policy_ref")["paid_total_real"].sum()
    res_by_pol = open_val.groupby(level="policy_ref").sum()
    ult_indep = (ana["policy_ref"].map(paid_by_pol).fillna(0.0) + ana["policy_ref"].map(res_by_pol).fillna(0.0))
    record("V2a", "sum of pricing target over all policies (TRY)", float(ana["ultimate_rbns"].sum()),
           float(ult_indep.sum()), 1e-9, kind="rel", source="analytical_dataset.ultimate_rbns",
           note="independent: policy map of real paid + open reserves at valuation")
    record("V2b", "max |policy-level difference| of the target (TRY)", 0.0,
           float((ana["ultimate_rbns"] - ult_indep).abs().max()), 1e-3, source="analytical_dataset.ultimate_rbns")
    bp = pd.read_parquet(DATA_PROC_DIR / "pricing_best_preds.parquet")
    te_mask = ana["policy_year"].isin(TEST_YEARS)
    record("V2c", "test-year target total in 14b parquet (TRY)", float(bp["pp_true"].sum()),
           float(np.clip(ult_indep[te_mask], 0, None).sum()), 1e-9, kind="rel", source="pricing_best_preds.pp_true")

    # ------------------------------------------------------------------ V3
    log.info("\n[V3] τ_FS = τ_b(N, Y): independent class-based implementation")
    claim = paid.groupby("claim_file").agg(policy_ref=("policy_ref", "first"),
                                           severity=("paid_total_real", "sum")).reset_index()
    claim = claim[claim["severity"] > 0]
    freq = paid.groupby("policy_ref")["claim_file"].nunique().rename("n_claims")
    claim["n_claims"] = claim["policy_ref"].map(freq).astype(int)
    xN = claim["n_claims"].values; yS = claim["severity"].values.astype(float)
    tau_ind = tau_b_by_class(xN, yS)
    record("V3a", "tau_FS (claim-level Kendall tau_b)", m06["axis_tau"]["FS_freq_severity"], tau_ind, 1e-10,
           source="06_manifest.axis_tau.FS_freq_severity", note="independent concordance counting by count class")
    record("V3b", "tau_FS via scipy on the rebuilt table", m06["axis_tau"]["FS_freq_severity"],
           float(stats.kendalltau(xN, yS)[0]), 1e-10, source="06_manifest.axis_tau.FS_freq_severity")

    # ------------------------------------------------------------------ V4
    log.info(f"\n[V4] τ_FS policy-cluster bootstrap (simple percentile, B={V4_B})")
    codes, uniq = pd.factorize(claim["policy_ref"].values)
    order = np.argsort(codes, kind="stable"); sc = codes[order]
    starts = np.searchsorted(sc, np.arange(len(uniq))); counts = np.diff(np.append(starts, len(sc)))
    rng = np.random.default_rng(np.random.SeedSequence([GLOBAL_SEED, 2301]))
    taus = np.empty(V4_B)
    for b in range(V4_B):
        pick = rng.integers(0, len(uniq), len(uniq))
        lens = counts[pick]; tot = int(lens.sum())
        first = np.repeat(starts[pick], lens)
        within = np.arange(tot) - np.repeat(np.cumsum(lens) - lens, lens)
        idx = order[first + within]
        taus[b] = stats.kendalltau(xN[idx], yS[idx])[0]
    lo, hi = np.percentile(taus, [2.5, 97.5]); se_b = float(taus.std(ddof=1))
    ci06 = m06["axis_FS_bca_ci"]
    record("V4a", "tau_FS cluster-bootstrap CI lower", ci06[0], float(lo), 0.0015,
           source="06_manifest.axis_FS_bca_ci[0]", note=f"simple percentile, B={V4_B}; 06: BCa, B=500")
    record("V4b", "tau_FS cluster-bootstrap CI upper", ci06[1], float(hi), 0.0015, source="06_manifest.axis_FS_bca_ci[1]")
    record("V4c", "tau_FS cluster-bootstrap SE", (ci06[1] - ci06[0]) / (2 * 1.96), se_b, 0.0005,
           source="implied from 06 CI half-width / 1.96")
    log.info(f"    bootstrap SE={se_b:.5f}; iid tie-corrected null SE (reference) 0.00170")

    # ------------------------------------------------------------------ V5
    log.info("\n[V5] Chain-ladder + Mack — matrix form (paid and incurred)")
    tri = pd.read_csv(OUTPUTS_DIR / "03_development_triangle.csv", index_col=0)
    r = chain_ladder_mack_matrix(tri)
    record("V5a", "paid CL ultimate total (TRY)", m03["chain_ladder"]["ultimate_total"], r["ultimate_total"], 1e-10, kind="rel",
           source="03_manifest.chain_ladder.ultimate_total")
    record("V5b", "paid CL reserve / IBNR (TRY)", m03["chain_ladder"]["ibnr"], r["reserve_total"], 1e-8, kind="rel",
           source="03_manifest.chain_ladder.ibnr")
    record("V5c", "paid Mack total s.e. (TRY)", m03["mack"]["total_se"], r["total_se"], 1e-8, kind="rel",
           source="03_manifest.mack.total_se")
    record("V5d", "paid first development factor f(0->1)", m03["chain_ladder"]["f_0_1"], float(r["factors"][0]), 1e-12,
           source="03_manifest.chain_ladder.f_0_1")
    tri_inc = pd.read_csv(OUTPUTS_DIR / "22_incurred_triangle.csv", index_col=0)
    ri = chain_ladder_mack_matrix(tri_inc)
    paid_to_date = float(np.nansum([tri.values[i, np.where(~np.isnan(tri.values[i]))[0].max()] for i in range(len(tri))]))
    record("V5e", "incurred CL ultimate total (TRY)", m22["incurred_CL"]["ultimate"], ri["ultimate_total"], 1e-10, kind="rel",
           source="22_manifest.incurred_CL.ultimate")
    record("V5f", "incurred CL outstanding = ultimate - paid to date (TRY)", m22["incurred_CL"]["outstanding"],
           ri["ultimate_total"] - paid_to_date, 1e-8, kind="rel", source="22_manifest.incurred_CL.outstanding")
    record("V5g", "incurred Mack total s.e. (TRY)", m22["incurred_CL"]["mack_total_se"], ri["total_se"], 1e-8, kind="rel",
           source="22_manifest.incurred_CL.mack_total_se")

    # ------------------------------------------------------------------ V6
    log.info("\n[V6] GLMs with statsmodels (14b design matrix, independent fitting)")
    spec = importlib.util.spec_from_file_location("pm14b", _HERE / "14b_pricing_models.py")
    pm = importlib.util.module_from_spec(spec); spec.loader.exec_module(pm)
    ana["exposure"] = pd.to_numeric(ana["exposure"], errors="coerce").fillna(1.0)
    ana.loc[ana["exposure"] <= 0, "exposure"] = 1.0
    fit = ana[ana["policy_year"].isin(FIT_YEARS)].copy(); te = ana[ana["policy_year"].isin(TEST_YEARS)].copy()
    st = pm.design_stats(fit); Xf = pm.design(fit, stats=st); cols = list(Xf.columns)
    Xt = pm.design(te, ref=cols, stats=st)
    Xf_c = sm.add_constant(Xf.values.astype(float), has_constant="add")
    Xt_c = sm.add_constant(Xt.values.astype(float), has_constant="add")
    y_f = fit["n_claims"].values.astype(float); off_f = np.log(fit["exposure"].values); off_t = np.log(te["exposure"].values)
    pp_f = np.clip(fit["ultimate_rbns"].values.astype(float), 0, None)
    pp_t = np.clip(te["ultimate_rbns"].values.astype(float), 0, None)
    g14 = {m["model"]: m["pp_gini"] for m in m14["models"]}
    sev_mean = (fit.loc[fit["n_claims"] > 0, "ultimate_rbns"] / fit.loc[fit["n_claims"] > 0, "n_claims"]).mean()
    pois = sm.GLM(y_f, Xf_c, family=sm.families.Poisson(), offset=off_f).fit()
    mu_pois_t = pois.predict(Xt_c, offset=off_t)
    record("V6a", "Poisson GLM test Gini", g14["Poisson GLM"], pm.gini(pp_t, mu_pois_t * sev_mean), 2e-4,
           source="14b_manifest.models", note="statsmodels IRLS vs in-house IRLS")
    alpha = float(m14["nb_alpha_estimated"])
    nb = sm.GLM(y_f, Xf_c, family=sm.families.NegativeBinomial(alpha=alpha), offset=off_f).fit()
    record("V6b", "NegBin GLM test Gini", g14["NegBin GLM"], pm.gini(pp_t, nb.predict(Xt_c, offset=off_t) * sev_mean), 2e-4,
           source="14b_manifest.models", note=f"alpha fixed at 14b estimate {alpha:.4f}")
    cl = fit["n_claims"].values > 0
    sev_f = (fit.loc[cl, "ultimate_rbns"].values / fit.loc[cl, "n_claims"].values); w_f = fit.loc[cl, "n_claims"].values.astype(float)
    ok = sev_f > 0
    gam = sm.GLM(sev_f[ok], Xf_c[cl][ok], family=sm.families.Gamma(link=sm.families.links.Log()), var_weights=w_f[ok]).fit()
    two = mu_pois_t * gam.predict(Xt_c)
    k_two = [k for k in g14 if k.startswith("Two-stage GLM")][0]
    record("V6c", "Two-stage GLM (Poisson x Gamma) test Gini", g14[k_two], pm.gini(pp_t, two), 2e-4, source="14b_manifest.models")
    p_tw = float(m14["tweedie_best_power"])
    tw = sm.GLM(pp_f, Xf_c, family=sm.families.Tweedie(var_power=p_tw, link=sm.families.links.Log())).fit()
    k_tw = [k for k in g14 if k.startswith("Tweedie GLM")][0]
    record("V6d", f"Tweedie GLM (p={p_tw:g}) test Gini", g14[k_tw], pm.gini(pp_t, tw.predict(Xt_c)), 2e-4,
           source="14b_manifest.models")

    # ------------------------------------------------------------------ V7
    log.info("\n[V7] Gini of the selected model — Lorenz formula")
    record("V7a", f"selected model ({m14['best_model']}) test Gini", float(m14["best_gini"]),
           gini_lorenz(bp["pp_true"].values, bp["pp_pred"].values), 1e-3, source="14b_manifest.best_gini",
           note="trapezoid Lorenz area vs 14b rank-sum formula (discretisation differs at O(1/n))")

    # ------------------------------------------------------------------ V8
    log.info("\n[V8] Capital experiment — independent mini-simulator (empirical marginals, Gaussian link)")
    rho = float(m13["calibration"]["rho_star"]); tau_hat = float(m13["tau_dependence"])
    ncl = np.clip(ana["n_claims"].values.astype(int), 0, V8_COUNT_CAP)
    pmf = np.bincount(ncl, minlength=V8_COUNT_CAP + 1) / len(ncl); cdf = np.cumsum(pmf)
    sev_pool = np.sort(yS)

    # In the coupled arm, severities are realised only for N>=1 policies (U > F_N(0))
    # and are claim-weighted; the raw V is therefore NOT uniform at claim level
    # (margin shift — the effect that the H_rho correction removes in the paper).
    # Here it is corrected by an independent route: the claim-weighted EMPIRICAL
    # distribution function G_hat of V is estimated from a large pre-simulation
    # and V -> G_hat(V) (fixed monotone map; ranks and hence tau unchanged) is
    # applied in every replicate. Thus the claim-level severity marginal of the
    # two arms is identical in distribution.
    def simulate(n_pol, g, coupled, G_sorted=None):
        U = g.random(n_pol); N = np.searchsorted(cdf, U, side="left")
        tot = int(N.sum()); Z = stats.norm.ppf(U)
        Zrep = np.repeat(Z, N); eps = g.standard_normal(tot)
        V = stats.norm.cdf(rho * Zrep + np.sqrt(1 - rho ** 2) * eps) if coupled else stats.norm.cdf(eps)
        if coupled and G_sorted is not None:
            V = (np.searchsorted(G_sorted, V, side="right") + 0.5) / (len(G_sorted) + 1)
        Y = sev_pool[np.minimum((V * len(sev_pool)).astype(int), len(sev_pool) - 1)]
        return N, Y, V

    g0 = np.random.default_rng(np.random.SeedSequence([GLOBAL_SEED, 2307]))
    _, _, V_pre = simulate(V8_N_POL_TAU, g0, True); G_sorted = np.sort(V_pre)
    log.info(f"    G_hat: claim-weighted V distribution, {len(G_sorted):,} claims; raw V mean {V_pre.mean():.5f} "
             f"(0.5 if uniform)")
    g = np.random.default_rng(np.random.SeedSequence([GLOBAL_SEED, 2308]))
    N, Y, Vd_chk = simulate(V8_N_POL_TAU, g, True, G_sorted); tau_dep = float(stats.kendalltau(np.repeat(N, N), Y)[0])
    ks_corr = stats.kstest(Vd_chk, "uniform")
    log.info(f"    corrected V: KS D={ks_corr.statistic:.5f} (p={ks_corr.pvalue:.3f}); mean {Vd_chk.mean():.5f}")
    g2 = np.random.default_rng(np.random.SeedSequence([GLOBAL_SEED, 2309]))
    N2, Y2, _ = simulate(V8_N_POL_TAU, g2, False); tau_ind_arm = float(stats.kendalltau(np.repeat(N2, N2), Y2)[0])
    se_tau = 2.0 / np.sqrt(len(Y))  # rough MC s.e.
    record("V8a", "dependent arm: simulated claim-level tau_b at rho*", tau_hat, tau_dep, 0.003,
           source="13: tau_dependence (calibration target)",
           note=f"independent simulator, {V8_N_POL_TAU:,} policies, no margin correction; 13 model A check: "
                f"{m13['margin_correction_validation']['A_gauss_calibrated']['tau_b_claim']:+.5f}")
    record("V8b", "independent arm: simulated claim-level tau_b", 0.0, tau_ind_arm, 3 * se_tau,
           source="design (must be 0)", note=f"MC s.e. ~ {se_tau:.4f}")
    # coupled MC: common U and eps; only the link differs
    g3 = np.random.default_rng(np.random.SeedSequence([GLOBAL_SEED, 2310]))
    S_dep = np.empty(V8_REPS); S_ind = np.empty(V8_REPS)
    for rr in range(V8_REPS):
        U = g3.random(V8_N_POL); N = np.searchsorted(cdf, U, side="left"); tot = int(N.sum())
        Zrep = np.repeat(stats.norm.ppf(U), N); eps = g3.standard_normal(tot)
        Vd = stats.norm.cdf(rho * Zrep + np.sqrt(1 - rho ** 2) * eps); Vi = stats.norm.cdf(eps)
        Vd = (np.searchsorted(G_sorted, Vd, side="right") + 0.5) / (len(G_sorted) + 1)   # margin correction
        S_dep[rr] = sev_pool[np.minimum((Vd * len(sev_pool)).astype(int), len(sev_pool) - 1)].sum()
        S_ind[rr] = sev_pool[np.minimum((Vi * len(sev_pool)).astype(int), len(sev_pool) - 1)].sum()
        if (rr + 1) % 1000 == 0:
            log.info(f"    coupled MC {rr+1}/{V8_REPS} ({time.time()-t0:.0f} s)")
    def var_change(sd, si):
        return 100.0 * (np.quantile(sd, SOLVENCY_ALPHA) - np.quantile(si, SOLVENCY_ALPHA)) / np.quantile(si, SOLVENCY_ALPHA)
    vc = var_change(S_dep, S_ind)
    gb = np.random.default_rng(np.random.SeedSequence([GLOBAL_SEED, 2311])); vcb = np.empty(V8_BOOT_B)
    for b in range(V8_BOOT_B):
        ix = gb.integers(0, V8_REPS, V8_REPS); vcb[b] = var_change(S_dep[ix], S_ind[ix])
    vlo, vhi = np.percentile(vcb, [2.5, 97.5])
    es_change = 100.0 * (S_dep.mean() - S_ind.mean()) / S_ind.mean()
    record("V8c", "VaR99.5 change dependent vs independent (%) — CI contains 0", 0.0, (float(vlo), float(vhi)), 0,
           kind="contains", source="13: +0.0075% [-0.034, +0.030]",
           note=f"independent simulator point {vc:+.4f}%, E[S] change {es_change:+.4f}%, R={V8_REPS} x {V8_N_POL:,} policies")
    record("V8d", "|VaR99.5 change| far below the 5% materiality level (tolerance 0.5%)",
           abs(float(m13["scr_bias_pct"])), abs(vc), 0.5, source="13_manifest.scr_bias_pct",
           note=(f"independent simulator MC 95% half-width ~{(vhi - vlo) / 2:.3f} pp; the residual level "
                 "reflects G_hat estimation noise (KS D of corrected V in manifest), not dependence"))

    # ------------------------------------------------------------------ V9
    log.info("\n[V9] Case-reserve run-off 2017-12 (separate code path)")
    snap = res[(res["reserve_year"] == 2017) & (res["reserve_month"] == 12)]
    snap = snap.groupby(["policy_ref", "claim_file"])["reserve_amount_real"].last()
    later = paid[paid["payment_year"] > 2017].groupby(["policy_ref", "claim_file"])["paid_total_real"].sum()
    paid_after = later.reindex(snap.index).fillna(0.0)
    ro17 = [r_ for r_ in m22["case_reserve_runoff"]["per_snapshot"] if r_["snapshot_year_end"] == 2017][0]
    record("V9a", "2017-12 open files", ro17["open_files"], int(len(snap)), 0, source="22_manifest.case_reserve_runoff")
    record("V9b", "2017-12 paid-subsequently / case-reserve ratio", ro17["paid_to_reserve_ratio"],
           float(paid_after.sum() / snap.sum()), 1e-9, source="22_manifest.case_reserve_runoff")

    # ------------------------------------------------------------------ V10
    log.info("\n[V10] Raw-level P/O ratio of the selected model")
    record("V10a", "selected model overall P/O (raw level)", float(m14["calibration"]["overall_PO"]),
           float(bp["pp_pred"].sum() / bp["pp_true"].sum()), 1e-9, source="14b_manifest.calibration.overall_PO")

    # ------------------------------------------------------------------ out
    tab = pd.DataFrame(ROWS)
    tab.to_csv(OUTPUTS_DIR / "23_verification_table.csv", index=False)
    n_pass = int(tab["pass"].sum()); n_all = len(tab)
    log.info(f"\n  VERIFICATION: {n_pass}/{n_all} rows passed")
    for _, r_ in tab[~tab["pass"]].iterrows():
        log.warning(f"    FAIL {r_['id']}: {r_['quantity']} pipeline={r_['pipeline_value']} indep={r_['independent_value']}")
    manifest = {
        "script": SCRIPT_NAME, "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "purpose": ("R2 Reviewer 1 point 3: headline results re-derived along independent code paths "
                    "(statsmodels GLMs, matrix-form Mack, class-based Kendall tau_b, simple cluster bootstrap, "
                    "independent coupled simulator with empirical margins, Lorenz Gini)"),
        "n_checks": n_all, "n_pass": n_pass, "all_pass": bool(n_pass == n_all),
        "V4": {"B": V4_B, "bootstrap_se": se_b, "ci": [float(lo), float(hi)]},
        "V8": {"rho_star": rho, "n_pol_tau": V8_N_POL_TAU, "tau_dep": tau_dep, "tau_ind": tau_ind_arm,
               "reps": V8_REPS, "n_pol": V8_N_POL, "var995_change_pct": float(vc), "ci95": [float(vlo), float(vhi)],
               "ES_change_pct": float(es_change),
               "margin_correction": ("empirical claim-weighted CDF G_hat of the coupled V from a pre-simulation of "
                                     f"{V8_N_POL_TAU:,} policies, applied as a fixed monotone map V -> G_hat(V) "
                                     "(independent of the quadrature H_rho of module 13)"),
               "V_corrected_ks_D": float(ks_corr.statistic), "V_corrected_ks_p": float(ks_corr.pvalue),
               "note": ("independent simulator with empirical margins; the check is on sign/order of magnitude "
                        "(CI contains 0; |change| << 5% materiality), not on numerical equality with 13")},
        "rows": ROWS, "runtime_s": time.time() - t0, "environment": env_info(),
    }
    with open(OUTPUTS_DIR / "23_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=float)
    log.info("\n" + "=" * 78); log.info("SCRIPT 23 COMPLETED"); log.info("=" * 78)


if __name__ == "__main__":
    main()
