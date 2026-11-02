"""
================================================================================
SCRIPT 24 — R2 Revision: Artifact/Residual Decomposition of the Naive Measures (B1, B2)
================================================================================
PURPOSE (Reviewer 1, round 2, point 1 — justification of B1/B2/B3 as a contribution;
      relation to Oh, Ahn & Lee (2021))
    The reviewer noted that B1 (count × total) is a basic property of the collective
    risk model, and that B2 (count × average) is a "structural dependence" already
    examined by Oh et al. (2021), and asked for B2 "where exactly previous approaches
    are problematic, and what this analysis adds beyond the literature".

    This module answers the question WITH DATA — under the casco portfolio's OWN
    margins and claim-count composition, it computes the value taken by the naive
    measures in a null-hypothesis world where frequency and severity are FULLY
    INDEPENDENT, and decomposes the observed value into "artifact + residual":

    [1] Observed: τ_b(N, ΣY) [A, B1], τ_b(N, ΣY/N) [B, B2] (claimant policies)
        and the clean claim-level τ_b(N, Y) [FS] — exactly the same definitions as 06.
    [2] Permutation null (R replicates): severities are randomly shuffled across
        claims (the policies' claim counts and the severity margin are preserved;
        frequency–severity independence holds EXACTLY); A, B and FS are recomputed.
        Mean = the artificial association produced by the portfolio's own
        structure; observed − mean = the residual beyond independence in the data.
    [3] Symmetric-severity null: the same permutation, but severities are drawn from
        a SYMMETRIC (normal) distribution with matched mean and variance. The 1/N
        variance scaling (heteroskedasticity) emphasised by Oh et al. (2021) is
        preserved, skewness is removed → Remark 1's prediction: the rank association
        (τ) vanishes; i.e. the sign and magnitude of B2 on the RANK scale come from
        skewness, not from the variance scaling.
    [4] Remark 1 analytic terms: τ_a(N, S/N | N≥1) = 2 Σ_{n1<n2} π_n1 π_n2
        E[sgn(Ȳ_n2 − Ȳ_n1)] — Monte Carlo for the fitted lognormal and the empirical
        severity; compared with the permutation null (carried to the τ_b scale
        with a tie correction).

    Only data/proc is read; outputs are AGGREGATE tables; the seed is fixed.

INPUT : data/proc/paid_long_real.parquet ; outputs/06_manifest.json (cross-check)
OUTPUT: outputs/24_artifact_decomposition.csv ; outputs/24_permutation_null.csv ;
        outputs/24_symmetric_null.csv ; outputs/24_remark1_terms.csv ;
        figures/24_artifact_decomposition.{pdf,png} ; outputs/24_manifest.json ;
        logs/24_b2_decomposition.log
================================================================================
"""
from __future__ import annotations
import sys
import json
import logging
import time
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (DATA_PROC_DIR, OUTPUTS_DIR, FIGURES_DIR, LOGS_DIR, GLOBAL_SEED,
                    FIG_DPI, FIG_W_TEXT, apply_pub_style, env_info)

SCRIPT_NAME = "24_b2_decomposition"
R_PERM = 200            # permutation replicates (null distribution)
R_SYM = 200             # symmetric-severity replicates
N_MC_REMARK = 400_000   # number of MC draws for the Remark 1 terms (per pair)
MAX_N_REMARK = 8        # Remark 1: n1 < n2 ≤ MAX_N_REMARK (8+ pooled)
PERM_STREAM, SYM_STREAM, REMARK_STREAM = 2401, 2402, 2403


def setup_logger() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(SCRIPT_NAME); lg.setLevel(logging.INFO); lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOGS_DIR / f"{SCRIPT_NAME}.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt); sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    lg.addHandler(fh); lg.addHandler(sh); return lg


log = logging.getLogger(SCRIPT_NAME)


def load_proc(stem: str) -> pd.DataFrame:
    pq = DATA_PROC_DIR / f"{stem}.parquet"
    return pd.read_parquet(pq) if pq.exists() else pd.read_pickle(DATA_PROC_DIR / f"{stem}.pkl")


def ktau(x, y) -> float:
    """Tie-corrected Kendall τ_b (scipy; same as 06)."""
    return float(stats.kendalltau(x, y)[0])


def build_claim_table() -> pd.DataFrame:
    """EXACTLY as in 06_dependence: claim file = total real paid (>0);
    policy claim count = number of claim files on the policy (all files)."""
    paid = load_proc("paid_long_real")
    claim = paid.groupby("claim_file").agg(
        policy_ref=("policy_ref", "first"),
        severity=("paid_total_real", "sum")).reset_index()
    claim = claim[claim["severity"] > 0]
    freq = (paid.groupby("policy_ref")["claim_file"].nunique()
            .rename("n_claims").reset_index())
    claim = claim.merge(freq, on="policy_ref", how="left")
    return claim.dropna(subset=["n_claims"]).reset_index(drop=True)


def naive_measures(sev: np.ndarray, pol_codes: np.ndarray, n_pol: int,
                   n_claims_pol: np.ndarray, n_claims_claim: np.ndarray) -> dict:
    """A (N×ΣY), B (N×ΣY/N) and FS (N×Y) for the given severity vector."""
    total = np.bincount(pol_codes, weights=sev, minlength=n_pol)
    m = total > 0            # claimant policies (06: n_claims>0 and severity>0)
    avg = total[m] / n_claims_pol[m]
    return {"tau_A": ktau(n_claims_pol[m], total[m]),
            "tau_B": ktau(n_claims_pol[m], avg),
            "tau_FS": ktau(n_claims_claim, sev)}


def remark1_terms(draw_mean, pi: dict, rng, label: str) -> tuple[float, float, list]:
    """τ_a(N, S/N | N≥1) = 2 Σ_{n1<n2} π_n1 π_n2 E[sgn(Ȳ_n2 − Ȳ_n1)];
    draw_mean(n, size, rng) → mean of n claims (size draws)."""
    ns = sorted(pi)
    rows = []; tau_a = 0.0
    for i, n1 in enumerate(ns):
        for n2 in ns[i + 1:]:
            s = np.sign(draw_mean(n2, N_MC_REMARK, rng) - draw_mean(n1, N_MC_REMARK, rng))
            e = float(s.mean()); w = 2.0 * pi[n1] * pi[n2]
            tau_a += w * e
            rows.append({"distribution": label, "n1": n1, "n2": n2, "pi_n1": pi[n1], "pi_n2": pi[n2],
                         "E_sgn_mean_n2_minus_mean_n1": e, "mc_se": float(s.std(ddof=1) / np.sqrt(len(s))),
                         "weight_2_pi_pi": w, "contribution": w * e})
    tie_corr = float(np.sqrt(1.0 - sum(p * p for p in pi.values())))  # severity ties ignored
    return tau_a, tau_a / tie_corr, rows


def main() -> None:
    global log
    log = setup_logger(); apply_pub_style()
    log.info("=" * 78)
    log.info(f"SCRIPT {SCRIPT_NAME} — Artifact/residual decomposition of the naive measures (R2)")
    log.info("=" * 78)
    outfiles: list = []
    t0 = time.time()

    # ------------------------------------------------------------------ [1]
    log.info("\n[1/5] Claim table and observed measures (06 definitions)")
    claim = build_claim_table()
    sev = claim["severity"].values.astype(float)
    n_claim = claim["n_claims"].values.astype(float)
    pol_codes, pol_uniq = pd.factorize(claim["policy_ref"].values)
    n_pol = len(pol_uniq)
    n_claims_pol = np.zeros(n_pol); n_claims_pol[pol_codes] = n_claim   # N per policy
    obs = naive_measures(sev, pol_codes, n_pol, n_claims_pol, n_claim)
    log.info(f"  claims n={len(sev):,}; claimant policies G={n_pol:,}")
    log.info(f"  observed: A τ_b(N,ΣY)={obs['tau_A']:+.4f}; B τ_b(N,ΣY/N)={obs['tau_B']:+.4f}; "
             f"FS τ_b(N,Y)={obs['tau_FS']:+.4f}")
    chk = {}
    try:
        m06 = json.load(open(OUTPUTS_DIR / "06_manifest.json", encoding="utf-8"))["axis_tau"]
        chk = {"A_06": m06["A_mechanical"], "B_06": m06["B_ratio"],
               "A_diff": obs["tau_A"] - m06["A_mechanical"], "B_diff": obs["tau_B"] - m06["B_ratio"]}
        log.info(f"  cross-check against 06: A diff {chk['A_diff']:+.2e}, B diff {chk['B_diff']:+.2e}")
    except Exception as e:  # pragma: no cover
        log.warning(f"  06 manifest could not be read: {e}")
    # claim-count composition (claimant policies) → Remark 1 weights
    counts = pd.Series(n_claims_pol[n_claims_pol > 0].astype(int)).value_counts().sort_index()
    pi_full = (counts / counts.sum()).to_dict()
    pi = {int(k): float(v) for k, v in pi_full.items() if k < MAX_N_REMARK}
    pi[MAX_N_REMARK] = float(sum(v for k, v in pi_full.items() if k >= MAX_N_REMARK))
    log.info("  π_n (claimant policies): " + ", ".join(f"{k}:{v:.4f}" for k, v in pi.items()))

    # ------------------------------------------------------------------ [2]
    log.info(f"\n[2/5] Permutation null (exact independence; R={R_PERM})")
    rng = np.random.default_rng(np.random.SeedSequence([GLOBAL_SEED, PERM_STREAM]))
    perm_rows = []
    for r in range(R_PERM):
        s_perm = sev[rng.permutation(len(sev))]
        d = naive_measures(s_perm, pol_codes, n_pol, n_claims_pol, n_claim); d["rep"] = r
        perm_rows.append(d)
        if (r + 1) % 50 == 0:
            log.info(f"    permutation {r+1}/{R_PERM} ({time.time()-t0:.0f} s)")
    perm = pd.DataFrame(perm_rows)
    perm.to_csv(OUTPUTS_DIR / "24_permutation_null.csv", index=False)
    outfiles.append(str((OUTPUTS_DIR / "24_permutation_null.csv").resolve()))
    null = {k: {"mean": float(perm[k].mean()), "sd": float(perm[k].std(ddof=1)),
                "q025": float(perm[k].quantile(0.025)), "q975": float(perm[k].quantile(0.975))}
            for k in ("tau_A", "tau_B", "tau_FS")}
    for k in null:
        log.info(f"  {k}: null mean={null[k]['mean']:+.4f} (sd {null[k]['sd']:.4f}) "
                 f"[{null[k]['q025']:+.4f}, {null[k]['q975']:+.4f}]; observed {obs[k]:+.4f}; "
                 f"residual {obs[k]-null[k]['mean']:+.4f}")

    # ------------------------------------------------------------------ [3]
    log.info(f"\n[3/5] Symmetric-severity null (normal, mean/SD matched; R={R_SYM})")
    rng_s = np.random.default_rng(np.random.SeedSequence([GLOBAL_SEED, SYM_STREAM]))
    mu_y, sd_y = float(sev.mean()), float(sev.std(ddof=1))
    sym_rows = []
    for r in range(R_SYM):
        s_sym = rng_s.normal(mu_y, sd_y, len(sev))
        # no sign/zero issue for rank measures; for the total>0 mask the
        # mean is positive (mean=4,243, SD=9,895): negative totals are rare but
        # possible — the mask is dropped, all claimant policies are taken.
        total = np.bincount(pol_codes, weights=s_sym, minlength=n_pol)
        avg = total / n_claims_pol
        sym_rows.append({"rep": r, "tau_A": ktau(n_claims_pol, total),
                         "tau_B": ktau(n_claims_pol, avg), "tau_FS": ktau(n_claim, s_sym)})
    sym = pd.DataFrame(sym_rows)
    sym.to_csv(OUTPUTS_DIR / "24_symmetric_null.csv", index=False)
    outfiles.append(str((OUTPUTS_DIR / "24_symmetric_null.csv").resolve()))
    sym_null = {k: {"mean": float(sym[k].mean()), "sd": float(sym[k].std(ddof=1)),
                    "q025": float(sym[k].quantile(0.025)), "q975": float(sym[k].quantile(0.975))}
                for k in ("tau_A", "tau_B", "tau_FS")}
    for k in sym_null:
        log.info(f"  {k}: symmetric null mean={sym_null[k]['mean']:+.4f} "
                 f"[{sym_null[k]['q025']:+.4f}, {sym_null[k]['q975']:+.4f}]")

    # ------------------------------------------------------------------ [4]
    log.info("\n[4/5] Remark 1 analytic terms (lognormal MLE and empirical severity)")
    rng_r = np.random.default_rng(np.random.SeedSequence([GLOBAL_SEED, REMARK_STREAM]))
    ls = np.log(sev); mu_ln, sg_ln = float(ls.mean()), float(ls.std(ddof=0))
    skew_emp = float(stats.skew(sev)); skew_ln = float((np.exp(sg_ln**2) + 2) * np.sqrt(np.exp(sg_ln**2) - 1))

    def mean_ln(n, size, g):
        return np.exp(g.normal(mu_ln, sg_ln, (size, n))).mean(axis=1)

    def mean_emp(n, size, g):
        return sev[g.integers(0, len(sev), (size, n))].mean(axis=1)

    ta_ln, tb_ln, rows_ln = remark1_terms(mean_ln, pi, rng_r, "lognormal_MLE")
    ta_emp, tb_emp, rows_emp = remark1_terms(mean_emp, pi, rng_r, "empirical_bootstrap")
    rem = pd.DataFrame(rows_ln + rows_emp)
    rem.to_csv(OUTPUTS_DIR / "24_remark1_terms.csv", index=False)
    outfiles.append(str((OUTPUTS_DIR / "24_remark1_terms.csv").resolve()))
    log.info(f"  lognormal(μ={mu_ln:.3f}, σ={sg_ln:.3f}; skewness {skew_ln:.1f}): "
             f"τ_a={ta_ln:+.4f} → τ_b-scale {tb_ln:+.4f}")
    log.info(f"  empirical severity (skewness {skew_emp:.1f}): τ_a={ta_emp:+.4f} → τ_b-scale {tb_emp:+.4f}; "
             f"permutation null B={null['tau_B']['mean']:+.4f}")
    e12 = [r for r in rows_ln if r["n1"] == 1 and r["n2"] == 2][0]
    log.info(f"  (1,2) term: E[sgn]={e12['E_sgn_mean_n2_minus_mean_n1']:+.3f}, "
             f"weighted contribution {e12['contribution']:+.4f}")

    # ------------------------------------------------------------------ [5]
    log.info("\n[5/5] Decomposition table + figure + manifest")
    rows = []
    for k, lab, mech in (("tau_A", "A: count x aggregate loss (B1)", "count-sum tautology"),
                         ("tau_B", "B: count x average severity (B2)", "aggregation-induced rank association"),
                         ("tau_FS", "FS: count x per-claim severity (clean)", "none (clean axis)")):
        resid = obs[k] - null[k]["mean"]
        rows.append({"measure": lab, "mechanism": mech, "observed_tau_b": obs[k],
                     "independence_null_mean": null[k]["mean"], "independence_null_sd": null[k]["sd"],
                     "independence_null_q025": null[k]["q025"], "independence_null_q975": null[k]["q975"],
                     "artifact_share_of_observed": (null[k]["mean"] / obs[k]) if abs(obs[k]) > 1e-12 else np.nan,
                     "residual_observed_minus_null": resid,
                     "residual_z_vs_null_sd": resid / null[k]["sd"] if null[k]["sd"] > 0 else np.nan,
                     "symmetric_severity_null_mean": sym_null[k]["mean"],
                     "symmetric_severity_null_q025": sym_null[k]["q025"],
                     "symmetric_severity_null_q975": sym_null[k]["q975"]})
    dec = pd.DataFrame(rows)
    dec.to_csv(OUTPUTS_DIR / "24_artifact_decomposition.csv", index=False)
    outfiles.append(str((OUTPUTS_DIR / "24_artifact_decomposition.csv").resolve()))
    for _, r in dec.iterrows():
        log.info(f"  {r['measure']:40s} observed {r['observed_tau_b']:+.4f} = artifact {r['independence_null_mean']:+.4f} "
                 f"+ residual {r['residual_observed_minus_null']:+.4f}  (symmetric null {r['symmetric_severity_null_mean']:+.4f})")

    # figure: observed vs independence null (three measures)
    fig, ax = plt.subplots(figsize=(FIG_W_TEXT, 2.6))
    labs = ["A: count $\\times$ aggregate\nloss (B1)", "B: count $\\times$ average\nseverity (B2)",
            "FS: count $\\times$ per-claim\nseverity (clean)"]
    x = np.arange(3); w = 0.27
    ax.bar(x - w, [obs[k] for k in ("tau_A", "tau_B", "tau_FS")], w, color="#eb6834", label="observed")
    nm = [null[k]["mean"] for k in ("tau_A", "tau_B", "tau_FS")]
    err = np.array([[nm[i] - null[k]["q025"], null[k]["q975"] - nm[i]]
                    for i, k in enumerate(("tau_A", "tau_B", "tau_FS"))]).T
    ax.bar(x, nm, w, color="#2a78d6", yerr=err, capsize=2, label="exact independence (permutation null)")
    ax.bar(x + w, [sym_null[k]["mean"] for k in ("tau_A", "tau_B", "tau_FS")], w, color="#7a7974",
           label="independence, symmetric severities")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylabel("Kendall's $\\tau_b$")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    for ext in ("pdf", "png"):
        fp = FIGURES_DIR / f"24_artifact_decomposition.{ext}"
        fig.savefig(fp, dpi=FIG_DPI); outfiles.append(str(fp.resolve()))
    plt.close(fig)

    manifest = {
        "script": SCRIPT_NAME, "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "purpose": ("R2 Reviewer 1 point 1: decompose the naive count-sum (B1) and count-average (B2) "
                    "associations of the casco portfolio into the part reproduced under exact "
                    "frequency-severity independence (permutation null with the portfolio's own "
                    "claim-count composition and severity margin) and the residual"),
        "n_claims": int(len(sev)), "n_claimant_policies": int(n_pol),
        "observed": obs, "crosscheck_06": chk,
        "claim_count_shares_claimant_policies": {str(k): v for k, v in pi.items()},
        "permutation_null": {"R": R_PERM, "seed": [GLOBAL_SEED, PERM_STREAM],
                             "definition": ("severities permuted across claims; claim files, policies and "
                                            "claim counts fixed; A, B, FS recomputed per replicate"),
                             "summary": null},
        "symmetric_severity_null": {"R": R_SYM, "seed": [GLOBAL_SEED, SYM_STREAM],
                                    "definition": ("severities drawn i.i.d. normal with the observed mean and SD "
                                                   "(variance scaling 1/N of the average preserved, skewness removed)"),
                                    "mean_sd_used": [mu_y, sd_y], "summary": sym_null},
        "remark1": {"n_mc_per_pair": N_MC_REMARK, "max_n_pooled": MAX_N_REMARK,
                    "lognormal_mle": {"mu": mu_ln, "sigma": sg_ln, "skewness": skew_ln,
                                      "tau_a": ta_ln, "tau_b_scale": tb_ln},
                    "empirical": {"skewness": skew_emp, "tau_a": ta_emp, "tau_b_scale": tb_emp},
                    "tie_correction_note": ("tau_b scale = tau_a / sqrt(1 - sum pi_n^2); severity ties ignored")},
        "decomposition": dec.to_dict(orient="records"),
        "interpretation": ("If the independence null reproduces the observed B (and A) coefficient, the "
                           "naive measures carry no information about frequency-severity dependence in "
                           "this portfolio; the symmetric-severity null isolates skewness (Remark 1) from "
                           "the 1/N variance scaling discussed by Oh, Ahn and Lee (2021)."),
        "runtime_s": time.time() - t0, "output_files": outfiles, "environment": env_info(),
    }
    with open(OUTPUTS_DIR / "24_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=float)
    log.info("\n" + "=" * 78); log.info("SCRIPT 24 COMPLETED"); log.info("=" * 78)


if __name__ == "__main__":
    main()
