"""
================================================================================
SCRIPT 21 — R1 Revision Analyses (Revision-Stage Quantities, Seeded)
================================================================================
PURPOSE
    Produces, inside the pipeline, seeded and manifest-audited, all numbers
    that were added to the paper in the R1 revision and had previously been
    computed with an unrecorded ad hoc script and HARD-CODED into the text.
    Only data/proc (+ the 06/08 manifests, the 00/01/02/03 logs) is read; the
    outputs are AGGREGATE tables only (no policy/claim rows are written).

    [1] Table 1: policy-level rating covariates (818,789 policies) — mean,
        SD, P5, median, P95; vehicle value is NOMINAL (not deflated; recorded
        in the manifest), gender and body-type shares.
    [2] Portfolio composition: underwriting year 2014-2018 → policies, earned
        exposure, claims (BASIS: distinct claim files linked to the policy in
        the paid ledger; the policy file has no count field, only a binary
        claim-status flag), frequency per exposure; accident year 2014-2019 →
        linked claim counts.
    [3] Claim-count distribution (0/1/2/3+, mean, VMR); q_k (share of claims
        by policy claim count); tie bounds √(1−Σq_k²) and 1−Σq_k²; share of
        claims on multi-claim policies; β_N composition effect (severity- and
        count-weighted; multiplicative and linear definitions).
    [4] Severity statistics, real (Dec-2018 prices) and nominal, on the SAME
        claim set; reconciliation of 3,309/1,142 with 3,297/1,134.
    [5] Fully covariate-adjusted log-severity regression (Appendix A, Table A1)
        — identical to 06's design; policy-cluster bootstrap B=500 (same seed
        as 06 → same draws); VIF in the full design and in the reduced numeric
        design; value/horsepower/engine-displacement correlations; degenerate-
        term flags.
    [6] TOST equivalence test for τ_FS over a grid of margins.
    [7] Power: i.i.d. tie-free formula SE, tie-corrected EXACT null SE
        (Kendall 1970 / SciPy kendalltau p-value variance), MDEs, cluster MDEs
        (08), design effects, power at the observed |τ| with the CLUSTER SE.
    [8] Data-cleaning accounting (number of records removed per rule).

INPUT : data/proc/{analytical_dataset,paid_long,paid_long_real,
        reserve_panel_real}.parquet ; outputs/{06,08}_manifest.json ;
        logs/{00_data_ingest,01_clean,02_inflation,03_analytical}.log
OUTPUT: outputs/21_*.csv ; outputs/21_manifest.json ; logs/21_revision_R1.log
================================================================================
"""
from __future__ import annotations
import re
import sys
import json
import logging
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
from scipy import stats

from config import (
    DATA_PROC_DIR, OUTPUTS_DIR, LOGS_DIR, GLOBAL_SEED, NUMERIC_COVARS,
    CATEG_COVARS, AXIS_D_N_BOOTSTRAP, TAU_CLUSTER_N_BOOTSTRAP, POWER_ALPHA,
    POWER_TARGET, STUDY_YEAR_MIN, STUDY_YEAR_MAX, CLAIM_STATUS_CLAIMED,
    INFLATION_BASE_YEAR_MONTH, CLIP_NEGATIVE_PAID, env_info,
)

SCRIPT_NAME = "21_revision_R1"

# --- R1 revision constants ---------------------------------------------------
# (config.py is not edited; constants specific to this module live only here)
R1_REG_BOOT_B = AXIS_D_N_BOOTSTRAP        # 500 — same as 06 axis_D_direct
R1_REG_BOOT_SEED = GLOBAL_SEED            # same seed as 06 + same cluster
                                          # order → identical draws
R1_LOG_EXPOSURE_FLOOR = 1e-6              # 06: log(clip(exposure, 1e-6))
R1_STD_EPS = 1e-9                         # 06: (x−mean)/(std+1e-9), ddof=0
R1_DESC_Q = (0.05, 0.50, 0.95)            # Table 1 percentiles (linear)
R1_SEV_Q = (0.90, 0.95, 0.99)             # severity percentiles (linear)
R1_TOST_MARGINS = (0.005, 0.0075, 0.01, 0.015, 0.02, 0.05, 0.10, 0.20)
R1_TOST_REF_MARGIN = 0.02                 # the paper's reference margin
R1_TOST_ALPHA = 0.05                      # level of each one-sided test
R1_QK_TOPCODE = 6                         # q_k output is pooled at k≥6
                                          # (disclosure); Σq_k² from the full distribution
R1_DEGENERATE_MIN_CLAIMS = 30             # "degenerate" flag if a dummy is
                                          # identified by ≤ this many claims
R1_VALUE_DEFLATE_MONTH = 6                # sensitivity: vehicle value with the
                                          # policy-year mid-month CPI (same rule
                                          # as the premium deflation in 18)
R1_GENDER_LABELS = {"ERKEK": "male", "KADIN": "female", "TUZEL": "corporate"}
R1_COVAR_LABELS = {
    "Arac_Yasi": "Vehicle age (years)",
    "Sigortali_Yasi": "Insured age (years)",
    "Horse_Power": "Horsepower",
    "CCM": "Engine displacement (cc)",
    "Arac_Bedeli": "Insured vehicle value (TRY, nominal)",
    "exposure": "Earned exposure (years)",
}
R1_TERM_LABELS = {
    "const": "Intercept", "n_claims": "Claim count N",
    "log_exposure": "log(earned exposure)",
    "Arac_Yasi": "Vehicle age (std.)", "Sigortali_Yasi": "Insured age (std.)",
    "Horse_Power": "Horsepower (std.)", "CCM": "Displacement (std.)",
    "Arac_Bedeli": "Vehicle value (std., nominal)",
    "Cinsiyet_KADIN": "Gender: female", "Cinsiyet_TUZEL": "Gender: corporate",
}
# Values HARD-CODED in the paper (R1 draft) — for comparison only.
# (value, number of quoted decimals)
R1_MS_TABLE1 = {  # variable: (mean, sd, p5, median, p95) + decimals
    "Arac_Yasi": ((3.32, 2.95, 0, 3, 9), (2, 2, 0, 0, 0)),
    "Sigortali_Yasi": ((36.8, 20.7, 0, 39, 67), (1, 1, 0, 0, 0)),
    "Horse_Power": ((115.0, 34.3, 75, 110, 177), (1, 1, 0, 0, 0)),
    "CCM": ((1545, 297, 1198, 1560, 1995), (0, 0, 0, 0, 0)),
    "Arac_Bedeli": ((76656, 64593, 24240, 60550, 183439), (0, 0, 0, 0, 0)),
    "exposure": ((0.99, 0.05, 0.99, 1.00, 1.00), (2, 2, 2, 2, 2)),
}
R1_MS_TABLE_A1 = {  # term: (beta, se, lo, hi, decimals)
    "const": (7.304, 0.012, 7.282, 7.328, 3),
    "n_claims": (-0.0436, 0.0040, -0.0515, -0.0355, 4),
    "AY2015": (-0.002, 0.012, -0.027, 0.020, 3),
    "AY2016": (0.061, 0.012, 0.036, 0.082, 3),
    "AY2017": (0.145, 0.012, 0.119, 0.168, 3),
    "AY2018": (0.092, 0.013, 0.066, 0.117, 3),
    "AY2019": (0.017, 0.021, -0.020, 0.061, 3),
    "log_exposure": (0.042, 0.097, -0.073, 0.297, 3),
    "Arac_Yasi": (-0.012, 0.004, -0.020, -0.005, 3),
    "Sigortali_Yasi": (-0.006, 0.006, -0.017, 0.005, 3),
    "Horse_Power": (0.117, 0.007, 0.104, 0.131, 3),
    "CCM": (-0.011, 0.006, -0.023, 0.002, 3),
    "Arac_Bedeli": (0.140, 0.006, 0.128, 0.151, 3),
    "Cinsiyet_KADIN": (-0.005, 0.008, -0.022, 0.010, 3),
    "Cinsiyet_TUZEL": (-0.027, 0.015, -0.054, 0.002, 3),
    "Body_Code_STW": (-1.64, 0.76, -1.67, 0.00, 2),
}
R1_MS_VIF = {"n_claims": 1.01, "log_exposure": 1.02, "Arac_Yasi": 1.33,
             "Sigortali_Yasi": 1.06, "Horse_Power": 3.71, "CCM": 2.93,
             "Arac_Bedeli": 2.47}
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


def load_proc(stem: str) -> pd.DataFrame:
    pq = DATA_PROC_DIR / f"{stem}.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    return pd.read_pickle(DATA_PROC_DIR / f"{stem}.pkl")


def read_json(name: str) -> dict:
    p = OUTPUTS_DIR / name
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # pragma: no cover
        log.warning(f"  {name} could not be read: {e}")
        return {}


def read_log(name: str) -> str | None:
    p = LOGS_DIR / name
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return None


def _f(v):
    """JSON-safe float."""
    if v is None:
        return None
    try:
        v = float(v)
    except Exception:
        return v
    return None if not np.isfinite(v) else v


def _jsonable(o):
    """Recursive clean-up for the manifest: numpy → Python, NaN/inf → None."""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if o is pd.NA or o is pd.NaT:
        return None
    return o


def write_csv(df: pd.DataFrame, stem: str, outfiles: list) -> None:
    p = OUTPUTS_DIR / f"{stem}.csv"
    df.to_csv(p, index=False)
    outfiles.append(str(p.resolve()))
    log.info(f"  → outputs/{stem}.csv ({len(df)} rows)")


# =============================================================================
# Data: claim axis (IDENTICAL construction and order to 06_dependence)
# =============================================================================
def build_claim_axis(paid: pd.DataFrame, ana: pd.DataFrame):
    """Same claim-level table as 06_dependence.main() [1/4].

    claim_all : ALL distinct claim files seen in the paid ledger
                (real + nominal cumulative payment, mean delay)
    claim     : files with cumulative REAL payment > 0 + policy N +
                policy covariates — same row order as in 06 (needed for an
                exact replay of the cluster-bootstrap draws)."""
    claim_all = paid.groupby("claim_file").agg(
        policy_ref=("policy_ref", "first"),
        accident_year=("accident_year", "first"),
        severity=("paid_total_real", "sum"),
        delay=("dev_year", "mean"),
        severity_nom=("paid_total", "sum"),
        n_policy_refs=("policy_ref", "nunique"),
        n_accident_years=("accident_year", "nunique")).reset_index()
    claim = claim_all[claim_all["severity"] > 0]
    freq = (paid.groupby("policy_ref")["claim_file"].nunique()
            .rename("n_claims").reset_index())
    claim = claim.merge(freq, on="policy_ref", how="left")
    keep = ["policy_ref", "exposure"] + [c for c in NUMERIC_COVARS
                                         + CATEG_COVARS if c in ana.columns]
    claim = claim.merge(ana[keep].drop_duplicates("policy_ref"),
                        on="policy_ref", how="left")
    claim = claim.dropna(subset=["n_claims"]).reset_index(drop=True)
    return claim_all, claim


# =============================================================================
# [1] Table 1
# =============================================================================
def table1(ana: pd.DataFrame, cpi_factor_by_year: dict) -> pd.DataFrame:
    rows = []
    for c in ["Arac_Yasi", "Sigortali_Yasi", "Horse_Power", "CCM",
              "Arac_Bedeli", "exposure"]:
        x = pd.to_numeric(ana[c], errors="coerce")
        q = x.quantile(list(R1_DESC_Q))
        rows.append({
            "block": "numeric", "variable": c, "level": "",
            "label": R1_COVAR_LABELS[c],
            "price_basis": ("nominal TRY (policy-year prices), NOT deflated"
                            if c == "Arac_Bedeli" else ""),
            "n": int(x.notna().sum()), "n_missing": int(x.isna().sum()),
            "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "p5": float(q.iloc[0]), "median": float(q.iloc[1]),
            "p95": float(q.iloc[2]), "count": np.nan, "share": np.nan})
    # sensitivity: vehicle value at Dec-2018 prices (policy-year mid-month CPI)
    if cpi_factor_by_year:
        fac = ana["policy_year"].map(cpi_factor_by_year)
        if fac.notna().all():
            x = pd.to_numeric(ana["Arac_Bedeli"], errors="coerce") * fac
            q = x.quantile(list(R1_DESC_Q))
            rows.append({
                "block": "sensitivity (not Table 1)", "variable": "Arac_Bedeli",
                "level": "real_Dec2018",
                "label": "Insured vehicle value (TRY, Dec-2018 prices)",
                "price_basis": (f"deflated with policy-year CPI of month "
                                f"{R1_VALUE_DEFLATE_MONTH:02d} to "
                                f"{INFLATION_BASE_YEAR_MONTH}"),
                "n": int(x.notna().sum()), "n_missing": int(x.isna().sum()),
                "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
                "p5": float(q.iloc[0]), "median": float(q.iloc[1]),
                "p95": float(q.iloc[2]), "count": np.nan, "share": np.nan})
    n = len(ana)
    g = ana["Cinsiyet"].astype(str).value_counts()
    for lev in ["ERKEK", "KADIN", "TUZEL"] + [l for l in g.index
                                             if l not in R1_GENDER_LABELS]:
        cnt = int(g.get(lev, 0))
        rows.append({"block": "categorical", "variable": "Cinsiyet",
                     "level": lev,
                     "label": f"Gender: {R1_GENDER_LABELS.get(lev, lev)}",
                     "price_basis": "", "n": n,
                     "n_missing": int(ana["Cinsiyet"].isna().sum()),
                     "count": cnt, "share": cnt / n})
    age0 = pd.to_numeric(ana["Sigortali_Yasi"], errors="coerce") == 0
    corp = ana["Cinsiyet"].astype(str) == "TUZEL"
    for lev, lab, m in [
            ("insured_age==0", "Insured age recorded as zero", age0),
            ("corporate & age==0", "Corporate policyholders with age zero",
             age0 & corp),
            ("corporate & age>0", "Corporate policyholders with age > 0",
             corp & ~age0),
            ("non-corporate & age==0",
             "Non-corporate policyholders with age zero", age0 & ~corp)]:
        rows.append({"block": "categorical", "variable": "Sigortali_Yasi",
                     "level": lev, "label": lab, "price_basis": "", "n": n,
                     "n_missing": 0, "count": int(m.sum()),
                     "share": float(m.mean())})
    b = ana["Body_Code"].astype(str).value_counts()
    modal = b.index[0]
    rows.append({"block": "categorical", "variable": "Body_Code",
                 "level": f"modal ({modal})", "label": "Body type: modal share",
                 "price_basis": "", "n": n,
                 "n_missing": int(ana["Body_Code"].isna().sum()),
                 "count": int(b.iloc[0]), "share": float(b.iloc[0] / n)})
    rows.append({"block": "categorical", "variable": "Body_Code",
                 "level": "non-modal (all)",
                 "label": ("Body type: non-modal policies ("
                           + ", ".join(f"{k} {int(v)}" for k, v in
                                       b.iloc[1:].items()) + ")"),
                 "price_basis": "", "n": n, "n_missing": 0,
                 "count": int(b.iloc[1:].sum()),
                 "share": float(b.iloc[1:].sum() / n)})
    return pd.DataFrame(rows)


# =============================================================================
# [2] Portfolio composition
# =============================================================================
def portfolio_by_uw_year(ana: pd.DataFrame, claim_all: pd.DataFrame,
                         cpi_factor_by_year: dict) -> pd.DataFrame:
    cl = claim_all.merge(ana[["policy_ref", "policy_year"]],
                         on="policy_ref", how="left")
    rows = []
    for yr in range(STUDY_YEAR_MIN, STUDY_YEAR_MAX + 1):
        a = ana[ana["policy_year"] == yr]
        c = cl[cl["policy_year"] == yr]
        expo = float(a["exposure"].sum())
        n_files = int(len(c))
        n_pos = int((c["severity"] > 0).sum())
        val = pd.to_numeric(a["Arac_Bedeli"], errors="coerce")
        rows.append({
            "underwriting_year": yr, "policies": int(len(a)),
            "earned_exposure": expo,
            "claims_linked_files": n_files,
            "claims_positive_real_payment": n_pos,
            "policies_with_linked_claim": int((a["n_claims"] > 0).sum()),
            "policies_claim_flag_policyfile": int(
                (a["claim_status"].astype(str) == CLAIM_STATUS_CLAIMED).sum()),
            "freq_per_exposure_linked_files": n_files / expo if expo else np.nan,
            "freq_per_exposure_positive_claims": n_pos / expo if expo else np.nan,
            "freq_per_policy_linked_files": n_files / len(a) if len(a) else np.nan,
            "vehicle_value_median_nominal": float(val.median()),
            "vehicle_value_median_real_Dec2018": (
                float(val.median() * cpi_factor_by_year[yr])
                if yr in cpi_factor_by_year else np.nan),
            "claims_basis": ("linked claim files = distinct claim-file numbers "
                             "in the paid-claims ledger linked through the "
                             "policy reference (the policy file has no claim-"
                             "count field, only a binary claim-status flag)"),
        })
    tot = {"underwriting_year": "total", "policies": int(len(ana)),
           "earned_exposure": float(ana["exposure"].sum()),
           "claims_linked_files": int(len(cl)),
           "claims_positive_real_payment": int((cl["severity"] > 0).sum()),
           "policies_with_linked_claim": int((ana["n_claims"] > 0).sum()),
           "policies_claim_flag_policyfile": int(
               (ana["claim_status"].astype(str) == CLAIM_STATUS_CLAIMED).sum())}
    tot["freq_per_exposure_linked_files"] = (tot["claims_linked_files"]
                                             / tot["earned_exposure"])
    tot["freq_per_exposure_positive_claims"] = (
        tot["claims_positive_real_payment"] / tot["earned_exposure"])
    tot["freq_per_policy_linked_files"] = (tot["claims_linked_files"]
                                           / tot["policies"])
    rows.append(tot)
    return pd.DataFrame(rows)


def claims_by_accident_year(claim_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ay, g in claim_all.groupby("accident_year"):
        rows.append({"accident_year": int(ay),
                     "claims_positive_real_payment": int((g["severity"] > 0).sum()),
                     "claims_linked_files_all": int(len(g)),
                     "claims_nonpositive_real_payment": int((g["severity"] <= 0).sum())})
    df = pd.DataFrame(rows)
    tot = {"accident_year": "total"}
    for c in df.columns[1:]:
        tot[c] = int(df[c].sum())
    return pd.concat([df, pd.DataFrame([tot])], ignore_index=True)


# =============================================================================
# [3] Claim-count distribution, q_k, tie bounds, composition effect
# =============================================================================
def claim_count_distribution(ana: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    n = ana["n_claims"].astype(int)
    rows = []
    for lab, m in [("0", n == 0), ("1", n == 1), ("2", n == 2), ("3+", n >= 3)]:
        rows.append({"quantity": f"share_policies_N={lab}",
                     "value": float(m.mean()), "count": int(m.sum())})
    mean, var = float(n.mean()), float(n.var(ddof=1))
    summ = {"n_policies": int(len(n)), "total_linked_claim_files": int(n.sum()),
            "mean": mean, "variance_ddof1": var, "vmr": var / mean}
    for k, v in summ.items():
        rows.append({"quantity": k, "value": float(v), "count": np.nan})
    df = pd.DataFrame(rows)
    df["basis"] = ("policy-level N = distinct linked claim files per policy "
                   "(analytical_dataset.n_claims; includes files with "
                   "nonpositive cumulative payment)")
    return df, summ


def composition(claim: pd.DataFrame, beta_N: float) -> tuple[pd.DataFrame, dict]:
    k = claim["n_claims"].astype(int).values
    s = claim["severity"].values.astype(float)
    n = len(k)
    kv, tk = np.unique(k, return_counts=True)
    q = tk / n
    S2 = float((q ** 2).sum())
    tkf = tk.astype(np.float64)
    exact_a = 1.0 - float((tkf * (tkf - 1)).sum()) / (n * (n - 1.0))
    rows = []
    # q_k (k ≥ R1_QK_TOPCODE pooled — disclosure)
    for kk, tt in zip(kv, tk):
        if kk < R1_QK_TOPCODE:
            rows.append({"section": "q_k", "quantity": f"q_{kk}",
                         "value": tt / n, "count": int(tt),
                         "definition": (f"share of the {n:,} severity claims "
                                        f"on policies with N = {kk}")})
    m_top = kv >= R1_QK_TOPCODE
    rows.append({"section": "q_k", "quantity": f"q_{R1_QK_TOPCODE}+",
                 "value": float(tk[m_top].sum() / n),
                 "count": int(tk[m_top].sum()),
                 "definition": (f"share of claims on policies with N >= "
                                f"{R1_QK_TOPCODE} (top-coded for disclosure; "
                                f"sum of q_k^2 uses the full distribution)")})
    tie = {
        "sum_qk2": S2,
        "tau_a_bound_asymptotic": 1.0 - S2,
        "tau_b_bound_asymptotic": float(np.sqrt(1.0 - S2)),
        "tau_a_bound_exact_finite_n": exact_a,
        "tau_b_bound_exact_finite_n": float(np.sqrt(exact_a)),
    }
    defs = {
        "sum_qk2": "sum_k q_k^2",
        "tau_a_bound_asymptotic": ("max tau-a under perfect monotone "
                                   "dependence given the count ties = "
                                   "1 - sum q_k^2"),
        "tau_b_bound_asymptotic": ("max tau-b (continuous severity) = "
                                   "sqrt(1 - sum q_k^2)"),
        "tau_a_bound_exact_finite_n": ("1 - sum t_k(t_k-1)/(n(n-1)), t_k = "
                                       "claims with N = k"),
        "tau_b_bound_exact_finite_n": "sqrt of the exact tau-a bound",
    }
    for kq, v in tie.items():
        rows.append({"section": "tie_bound", "quantity": kq, "value": v,
                     "count": np.nan, "definition": defs[kq]})
    multi = k >= 2
    mc = {"share_claims_on_multiclaim_policies": float(multi.mean()),
          "share_real_loss_on_multiclaim_policies": float(s[multi].sum() / s.sum()),
          "mean_N_over_claims": float(k.mean())}
    for kq, v in mc.items():
        rows.append({"section": "multi_claim", "quantity": kq, "value": v,
                     "count": int(multi.sum()) if kq.startswith("share_claims")
                     else np.nan,
                     "definition": ("claims (or real losses) on policies with "
                                    "N >= 2 / all severity claims"
                                    if kq.startswith("share") else
                                    "claim-weighted mean policy count")})
    # Composition effect: claim i (on a policy with N=k_i), m_i = exp(β(k_i−1))
    mult = np.exp(beta_N * (k - 1)) - 1.0
    loglin = beta_N * (k - 1)
    pctlin = (np.exp(beta_N) - 1.0) * (k - 1)
    ce = {
        "C1_severity_weighted_multiplicative":
            (float((s * mult).sum() / s.sum()),
             "sum_i s_i[exp(b(k_i-1))-1] / sum_i s_i  (expected-loss shift, "
             "exact multiplicative; weights = observed real severities)"),
        "C2_count_weighted_multiplicative":
            (float(mult.mean()),
             "(1/n) sum_i [exp(b(k_i-1))-1]  (average per-claim severity "
             "shift; each claim weighted equally)"),
        "C3_severity_weighted_loglinear":
            (float((s * loglin).sum() / s.sum()),
             "sum_i s_i b(k_i-1) / sum_i s_i  (first-order log approximation, "
             "severity-weighted)"),
        "C4_count_weighted_loglinear":
            (float(loglin.mean()), "(1/n) sum_i b(k_i-1)"),
        "C5_severity_weighted_pctlinear":
            (float((s * pctlin).sum() / s.sum()),
             "sum_i s_i (exp(b)-1)(k_i-1) / sum_i s_i  (-4.26% per extra "
             "claim applied additively)"),
        "C6_count_weighted_pctlinear":
            (float(pctlin.mean()), "(1/n) sum_i (exp(b)-1)(k_i-1)"),
    }
    for kq, (v, d) in ce.items():
        rows.append({"section": "composition_effect", "quantity": kq,
                     "value": v, "count": np.nan,
                     "definition": d + f"; b = beta_N = {beta_N:.7f}"})
    out = {"n_claims_axis": int(n), **tie, **mc,
           **{kq: v for kq, (v, _) in ce.items()},
           "beta_N_used": float(beta_N)}
    return pd.DataFrame(rows), out


# =============================================================================
# [4] Severity statistics (real and nominal, same set)
# =============================================================================
def _sev_stats(v: np.ndarray) -> dict:
    v = np.asarray(v, dtype=float)
    p = np.percentile(v, [100 * q for q in R1_SEV_Q])
    return {"n": int(len(v)), "mean": float(v.mean()),
            "median": float(np.median(v)), "sd": float(v.std(ddof=1)),
            "p90": float(p[0]), "p95": float(p[1]), "p99": float(p[2]),
            "cv": float(v.std(ddof=1) / v.mean())}


def severity_summary(claim_all: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    sets = {
        "S1_real_positive (primary severity sample)":
            claim_all["severity"] > 0,
        "S2_nominal_positive": claim_all["severity_nom"] > 0,
        "S3_all_linked_claim_files": np.ones(len(claim_all), dtype=bool),
    }
    rows, out = [], {}
    for sname, m in sets.items():
        sub = claim_all[np.asarray(m)]
        for basis, col in [("real_Dec2018", "severity"),
                           ("nominal", "severity_nom")]:
            st = _sev_stats(sub[col].values)
            rows.append({"claim_set": sname, "price_basis": basis, **st})
            out[f"{sname.split()[0]}|{basis}"] = st
    df = pd.DataFrame(rows)
    df["set_definition"] = df["claim_set"].map({
        "S1_real_positive (primary severity sample)":
            "claim files with cumulative REAL payment > 0 (n used in all "
            "severity analyses)",
        "S2_nominal_positive":
            "claim files with cumulative NOMINAL payment > 0",
        "S3_all_linked_claim_files":
            "all claim files in the paid ledger (incl. nonpositive sums)"})
    return df, out


# =============================================================================
# [5] Fully covariate-adjusted regression (Table A1)
# =============================================================================
def design_matrix(df: pd.DataFrame):
    """IDENTICAL design and column order to 06_dependence.axis_d_direct_test:
    [constant, N, accident-year dummies, log exposure, standardised
    numerics, gender and body-type dummies (drop_first)]."""
    y = np.log(df["severity"].values.astype(float))
    cols, names = [np.ones(len(df)), df["n_claims"].values.astype(float)], \
        ["const", "n_claims"]
    years = sorted(df["accident_year"].unique())
    for yr in years[1:]:
        cols.append((df["accident_year"] == yr).values.astype(float))
        names.append(f"AY{int(yr)}")
    ex = pd.to_numeric(df["exposure"], errors="coerce").fillna(1.0)
    cols.append(np.log(np.clip(ex.values.astype(float),
                               R1_LOG_EXPOSURE_FLOOR, None)))
    names.append("log_exposure")
    raw_num = {}
    for c in NUMERIC_COVARS:
        x = pd.to_numeric(df[c], errors="coerce").fillna(0).values.astype(float)
        raw_num[c] = x
        cols.append((x - x.mean()) / (x.std() + R1_STD_EPS))
        names.append(c)
    for c in CATEG_COVARS:
        d = pd.get_dummies(df[c].astype(str), prefix=c, drop_first=True)
        for col in d.columns:
            cols.append(d[col].values.astype(float))
            names.append(col)
    X = np.column_stack(cols)
    return X, y, names, raw_num


def ols(X: np.ndarray, y: np.ndarray) -> dict:
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    n, p = X.shape
    sigma2 = rss / (n - p)
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)
    return {"beta": beta, "se_iid": se, "r2": 1 - rss / tss,
            "adj_r2": 1 - (rss / (n - p)) / (tss / (n - 1)), "n": n, "p": p}


def cluster_index_blocks(cluster_ids: np.ndarray):
    """Same as 06: pd.factorize → stable sort → cluster start/end."""
    codes, uniq = pd.factorize(cluster_ids)
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    starts = np.searchsorted(sorted_codes, np.arange(len(uniq)))
    ends = np.append(starts[1:], len(sorted_codes))
    return order, starts, ends, len(uniq)


def cluster_bootstrap_coefs(X, y, cluster_ids, B=R1_REG_BOOT_B,
                            seed=R1_REG_BOOT_SEED):
    """Policy-cluster bootstrap: ALL coefficients. Draws are the same as in 06
    (same rng, same cluster order); the index vector is built vectorised but
    is identical to 06's np.concatenate([order[s:e] for p in pol]) output.
    If a dummy column is entirely zero in a replicate, the lstsq minimum-norm
    solution sets that coefficient to 0 — such replicates are counted as
    'unidentified'."""
    order, starts, ends, n_cl = cluster_index_blocks(cluster_ids)
    lens = ends - starts
    rng = np.random.default_rng(seed)
    p = X.shape[1]
    betas = np.full((B, p), np.nan)
    unident = np.zeros((B, p), dtype=bool)
    for b in range(B):
        pol = rng.integers(0, n_cl, n_cl)
        L = lens[pol]
        tot = int(L.sum())
        rep_start = np.repeat(starts[pol], L)
        within = np.arange(tot) - np.repeat(np.cumsum(L) - L, L)
        idx = order[rep_start + within]
        Xb = X[idx]
        unident[b] = ~(Xb != 0).any(axis=0)
        try:
            betas[b] = np.linalg.lstsq(Xb, y[idx], rcond=None)[0]
        except Exception:
            pass
    return betas, unident, n_cl


def vif_table(D: np.ndarray, names: list) -> dict:
    """VIF_j = 1/(1−R²_j), R²_j: from the OLS regression of column j on the
    constant + all other columns (D does NOT contain the constant column)."""
    out = {}
    n = D.shape[0]
    one = np.ones((n, 1))
    for j, nm in enumerate(names):
        yj = D[:, j]
        Z = np.hstack([one, np.delete(D, j, axis=1)])
        bj = np.linalg.lstsq(Z, yj, rcond=None)[0]
        r = yj - Z @ bj
        tss = float(((yj - yj.mean()) ** 2).sum())
        r2 = 1 - float(r @ r) / tss if tss > 0 else np.nan
        out[nm] = float(1.0 / (1.0 - r2)) if r2 < 1 else np.inf
    return out


# =============================================================================
# [7] Kendall τ null variance (tie-corrected; Kendall 1970, SciPy formula)
# =============================================================================
def tie_sums(v: np.ndarray) -> dict:
    _, t = np.unique(np.asarray(v), return_counts=True)
    t = t[t > 1].astype(np.float64)
    return {"n_tie_groups": int(t.size), "n_tied_obs": int(t.sum()),
            "T": float((t * (t - 1) / 2).sum()),           # Σ t(t−1)/2
            "V0": float((t * (t - 1) * (t - 2)).sum()),     # Σ t(t−1)(t−2)
            "V1": float((t * (t - 1) * (2 * t + 5)).sum())}  # Σ t(t−1)(2t+5)


NO_TIES = {"n_tie_groups": 0, "n_tied_obs": 0, "T": 0.0, "V0": 0.0, "V1": 0.0}


def kendall_null(n: int, tx: dict, ty: dict) -> dict:
    """EXACT null variance of S = C − D under independence (conditional on
    the tie pattern; Kendall 1970, eq. 4.3; same formula as used in the
    asymptotic p-value of scipy.stats.kendalltau):
        Var(S) = [n(n−1)(2n+5) − Σt(t−1)(2t+5) − Σu(u−1)(2u+5)]/18
               + [Σt(t−1)(t−2)][Σu(u−1)(u−2)] / [9n(n−1)(n−2)]
               + [Σt(t−1)][Σu(u−1)] / [2n(n−1)]
    τ-b = S / √((n0−T_x)(n0−T_y)), n0 = n(n−1)/2 → SE(τ-b) = √Var(S)/√(...).
    Without ties SE(τ-b) = √(2(2n+5)/(9n(n−1))) (08's i.i.d. formula)."""
    n = float(n)
    m = n * (n - 1.0)
    var_S = ((m * (2 * n + 5) - tx["V1"] - ty["V1"]) / 18.0
             + 2.0 * tx["T"] * ty["T"] / m
             + tx["V0"] * ty["V0"] / (9.0 * m * (n - 2.0)))
    n0 = m / 2.0
    den_b = np.sqrt((n0 - tx["T"]) * (n0 - ty["T"]))
    return {"var_S": float(var_S), "n0": n0, "den_tau_b": float(den_b),
            "se_tau_b": float(np.sqrt(var_S) / den_b),
            "se_tau_a": float(np.sqrt(var_S) / n0)}


def se_iid_noties(n: int) -> float:
    return float(np.sqrt(2 * (2 * n + 5) / (9 * n * (n - 1))))


# =============================================================================
# [8] Data-cleaning accounting
# =============================================================================
def _num(s: str) -> int:
    return int(s.replace(",", "").replace(".", ""))


def parse_cleaning_logs() -> dict:
    # EN edition: the patterns accept both the original Turkish and the translated English log wording.
    out = {"sources": {}}
    t00 = read_log("00_data_ingest.log")
    if t00 is not None:
        ing = {}
        for key in ("uretim", "odenen", "muallak"):
            m = re.search(rf"\[{key}\] ✓ \S+ \(([\d,]+) (?:satır|rows)", t00)
            if m:
                ing[key] = _num(m.group(1))
            elif re.search(rf"\[{key}\] \S+ (?:zaten var|already exists)", t00):
                ing[key] = "skipped (Parquet cache existed; ingestion rows " \
                           "not logged in this run)"
        out["ingest"] = ing
        out["sources"]["00"] = "logs/00_data_ingest.log"
    t01 = read_log("01_clean.log")
    if t01 is not None:
        pats = {"raw_policies": r"(?:ham üretim|raw production): \((\d+), \d+\)",
                "clean_policies": r"(?:temiz poliçe|clean policies): ([\d,]+)",
                "raw_paid": r"(?:ham ödenen|raw paid): \((\d+), \d+\)",
                "neg_paid_kept": r"(?:negatif ödenen|negative paid) .*?: ([\d,]+) (?:satır|rows)",
                "clean_paid": r"(?:temiz ödenen|clean paid): ([\d,]+)",
                "raw_reserve": r"(?:ham muallak|raw case reserves): \((\d+), \d+\)",
                "clean_reserve": r"(?:temiz muallak|clean case reserves): ([\d,]+)"}
        for k, p in pats.items():
            m = re.search(p, t01)
            out[k] = _num(m.group(1)) if m else None
        out["sources"]["01"] = "logs/01_clean.log"
    t03 = read_log("03_analytical.log")
    if t03 is not None:
        m = re.search(r"(?:Benzersiz hasar|Unique claims): ([\d,]+)", t03)
        out["claims_03"] = _num(m.group(1)) if m else None
        out["sources"]["03"] = "logs/03_analytical.log"
    m01 = read_json("01_manifest.json").get("counts", {})
    out["manifest_01_counts"] = m01
    return out


def cleaning_accounting(paid_nom: pd.DataFrame, paid: pd.DataFrame,
                        res: pd.DataFrame, ana: pd.DataFrame,
                        claim_all: pd.DataFrame, claim: pd.DataFrame
                        ) -> tuple[pd.DataFrame, dict]:
    L = parse_cleaning_logs()
    rows = []

    def add(stage, rule, before, removed, after, unit, source, note=""):
        rows.append({"stage": stage, "rule": rule, "records_before": before,
                     "records_removed": removed, "records_after": after,
                     "unit": unit, "source": source, "note": note})

    def split_rule(stage, rules, raw, clean, unit, src):
        """If several rules are applied in one step the log only gives the
        total; if the total is 0 every rule is 0 (cannot be negative)."""
        if raw is None or clean is None:
            for r in rules:
                add(stage, r, raw, "not recorded", clean, unit, src)
            return
        tot = raw - clean
        for r in rules:
            if tot == 0:
                add(stage, r, raw, 0, clean, unit, src,
                    "derived: raw count = cleaned count in the log, so every "
                    "rule of this step removed 0 records")
            else:
                add(stage, r, raw, "not recorded separately", clean, unit, src,
                    f"combined removal of this step = {tot:,}")

    ing = L.get("ingest", {})
    for key, lab in [("uretim", "policy file"), ("odenen", "paid-claims ledger"),
                     ("muallak", "case-reserve ledger")]:
        v = ing.get(key)
        raw_key = {"uretim": "raw_policies", "odenen": "raw_paid",
                   "muallak": "raw_reserve"}[key]
        add("00_data_ingest", f"Excel -> Parquet streaming ({lab}); every "
            "worksheet row is written (00_data_ingest.py has no filter)",
            v if isinstance(v, int) else "not recorded in current run",
            0, L.get(raw_key), "rows",
            L["sources"].get("00", "not available"),
            "removed = 0 by design (code inspection); "
            + (f"00 log: {v}" if v is not None else "00 log not available")
            + "; records_after = raw Parquet row count from the raw-shape "
            "line of logs/01_clean.log")
    src01 = L["sources"].get("01", "logs/01_clean.log (not available)")
    split_rule("01_clean policies",
               ["missing underwriting (policy start) date",
                f"underwriting year outside {STUDY_YEAR_MIN}-{STUDY_YEAR_MAX}"],
               L.get("raw_policies"), L.get("clean_policies"), "policies",
               src01)
    split_rule("01_clean paid",
               ["missing accident date or payment year-month",
                "negative development year (payment year < accident year)"],
               L.get("raw_paid"), L.get("clean_paid"), "payment rows", src01)
    split_rule("01_clean reserve", ["missing reserve year-month"],
               L.get("raw_reserve"), L.get("clean_reserve"),
               "reserve snapshots", src01)
    n_neg = int((paid_nom["paid_total"] < 0).sum())
    n_zero = int((paid_nom["paid_total"] == 0).sum())
    add("01_clean paid", "negative payments (salvage/subrogation) RETAINED",
        len(paid_nom), 0, len(paid_nom), "payment rows",
        f"{src01}; recomputed from data/proc/paid_long",
        f"{n_neg:,} negative rows kept (log: {L.get('neg_paid_kept')}); "
        f"{n_zero:,} zero-amount rows kept; clip_negative_paid="
        f"{CLIP_NEGATIVE_PAID}")
    n_pm_na = int(paid["payment_month"].isna().sum())
    n_rm_na = int(res["reserve_month"].isna().sum())
    add("02_inflation paid", "missing payment month -> imputed month "
        "(DEFAULT_PAID_MONTH), not removed", len(paid), 0, len(paid),
        "payment rows", "computed from data/proc/paid_long_real",
        f"{n_pm_na:,} rows imputed")
    add("02_inflation reserve", "missing reserve month -> imputed month "
        "(DEFAULT_RESERVE_MONTH), not removed", len(res), 0, len(res),
        "reserve snapshots", "computed from data/proc/reserve_panel_real",
        f"{n_rm_na:,} rows imputed")
    n_all = len(claim_all)
    n_pos = int((claim_all["severity"] > 0).sum())
    n_zero_c = int((claim_all["severity"] == 0).sum())
    n_neg_c = int((claim_all["severity"] < 0).sum())
    add("03/06 severity sample", "claim files with nonpositive cumulative "
        "REAL payment excluded from severity analyses", n_all, n_all - n_pos,
        n_pos, "claim files", "computed from data/proc/paid_long_real "
        f"(03 log: {L.get('claims_03')})",
        f"{n_zero_c:,} with zero and {n_neg_c:,} with negative cumulative "
        "real payment; these files still count in the policy claim count N")
    nom_pos = int((claim_all["severity_nom"] > 0).sum())
    rp_nn = int(((claim_all["severity"] > 0) & (claim_all["severity_nom"] <= 0)).sum())
    np_rn = int(((claim_all["severity_nom"] > 0) & (claim_all["severity"] <= 0)).sum())
    add("reconciliation only", "claim files with nonpositive cumulative "
        "NOMINAL payment (nominal-positive set, used for the old 3,309/1,142)",
        n_all, n_all - nom_pos, nom_pos, "claim files",
        "computed from data/proc/paid_long_real",
        f"{rp_nn:,} files are real-positive but nominal-nonpositive and "
        f"{np_rn:,} are nominal-positive but real-nonpositive")
    n_multi_pol = int((claim_all["n_policy_refs"] > 1).sum())
    n_multi_ay = int((claim_all["n_accident_years"] > 1).sum())
    add("03/06 claim linkage", "claim files linked to >1 policy reference or "
        ">1 accident year (first value used; not removed)", n_all, 0, n_all,
        "claim files", "computed from data/proc/paid_long_real",
        f"{n_multi_pol:,} with >1 policy reference; {n_multi_ay:,} with >1 "
        "accident year")
    unlinked = int((~claim_all["policy_ref"].isin(set(ana["policy_ref"]))).sum())
    add("03/06 claim linkage", "claim files whose policy reference is not in "
        "the cleaned policy file (unlinked)", n_all, unlinked, n_all - unlinked,
        "claim files", "computed from data/proc", "")
    for c in NUMERIC_COVARS + CATEG_COVARS + ["exposure"]:
        how = ("06 regression would fill NA with 0" if c in NUMERIC_COVARS
               else "06 regression would fill NA with 1.0" if c == "exposure"
               else "NA would form its own dummy level")
        add("covariates", f"missing values in {c} (not removed; {how})",
            len(ana), 0, len(ana), "policies",
            "computed from data/proc/analytical_dataset",
            f"{int(ana[c].isna().sum()):,} missing")
    zexp_pol = int((ana["exposure"] <= 0).sum())
    zexp_cl = int((pd.to_numeric(claim["exposure"], errors="coerce") <= 0).sum())
    add("covariates", "policies with nonpositive earned exposure (kept; "
        "log exposure floored at log(1e-6) in the regression)", len(ana), 0,
        len(ana), "policies", "computed from data/proc/analytical_dataset",
        f"{zexp_pol:,} policies; {zexp_cl:,} severity claims on them")
    info = {"parsed_logs": {k: v for k, v in L.items()
                            if k not in ("manifest_01_counts",)},
            "negative_payment_rows_kept": n_neg, "zero_payment_rows": n_zero,
            "payment_month_imputed": n_pm_na, "reserve_month_imputed": n_rm_na,
            "claim_files_all": n_all, "claim_files_real_positive": n_pos,
            "claim_files_nonpositive_real": n_all - n_pos,
            "claim_files_zero_real": n_zero_c,
            "claim_files_negative_real": n_neg_c,
            "claim_files_nominal_positive": nom_pos,
            "real_pos_nominal_nonpos": rp_nn, "nominal_pos_real_nonpos": np_rn,
            "claim_files_multi_policy": n_multi_pol,
            "claim_files_multi_accident_year": n_multi_ay,
            "claim_files_unlinked": unlinked,
            "policies_nonpositive_exposure": zexp_pol,
            "severity_claims_on_nonpositive_exposure": zexp_cl}
    return pd.DataFrame(rows), info


# =============================================================================
# Manuscript comparison
# =============================================================================
def _match(ms_val, new_val, dec) -> bool | None:
    if ms_val is None or new_val is None or not np.isfinite(new_val):
        return None
    return bool(abs(round(float(new_val), dec) - float(ms_val))
                < 0.5 * 10 ** (-dec) + 1e-12)


def main() -> None:
    global log
    log = setup_logger()
    log.info("=" * 78)
    log.info(f"SCRIPT {SCRIPT_NAME} — R1 revision analyses (seeded)")
    log.info("=" * 78)
    outfiles: list = []
    checks: list = []   # (section, quantity, manuscript, new, decimals, note)

    def chk(section, qty, ms, new, dec, note="", match=None):
        checks.append({"section": section, "quantity": qty,
                       "manuscript_value": ms, "pipeline_value": _f(new),
                       "quoted_decimals": dec,
                       "match_at_quoted_precision": (
                           _match(ms, new, dec) if match is None
                           else bool(match)),
                       "note": note})

    ana = load_proc("analytical_dataset")
    paid = load_proc("paid_long_real")
    paid_nom = load_proc("paid_long")
    res = load_proc("reserve_panel_real")
    m06 = read_json("06_manifest.json")
    m08 = read_json("08_manifest.json")

    # CPI factor (policy-year mid-month) — from the deflation-factor map in
    # data/proc (CPI_base/CPI_month computed by 02), no raw data.
    fmap = (paid[["ym_key", "deflation_factor"]].drop_duplicates("ym_key")
            .set_index("ym_key")["deflation_factor"].to_dict())
    cpi_factor_by_year = {yr: float(fmap[yr * 100 + R1_VALUE_DEFLATE_MONTH])
                          for yr in range(STUDY_YEAR_MIN, STUDY_YEAR_MAX + 1)
                          if (yr * 100 + R1_VALUE_DEFLATE_MONTH) in fmap}

    claim_all, claim = build_claim_axis(paid, ana)
    log.info(f"  Policies: {len(ana):,}; linked claim files: {len(claim_all):,}; "
             f"severity sample (real>0): {len(claim):,}")

    # ------------------------------------------------------------------ [1]
    log.info("\n[1/8] Table 1 — policy-level covariates")
    t1 = table1(ana, cpi_factor_by_year)
    write_csv(t1, "21_table1_covariates", outfiles)
    for c, (vals, decs) in R1_MS_TABLE1.items():
        r = t1[(t1["variable"] == c) & (t1["block"] == "numeric")].iloc[0]
        for stat, ms, dec in zip(["mean", "sd", "p5", "median", "p95"], vals,
                                 decs):
            chk("Table 1", f"{c} {stat}", ms, r[stat], dec)
        log.info(f"  {R1_COVAR_LABELS[c]:38s} mean={r['mean']:,.4f} "
                 f"sd={r['sd']:,.4f} P5={r['p5']:,.2f} med={r['median']:,.2f} "
                 f"P95={r['p95']:,.2f}")
    corp_share = float(t1.loc[(t1["variable"] == "Cinsiyet")
                              & (t1["level"] == "TUZEL"), "share"].iloc[0])
    chk("Sec 2.1", "corporate share of policies (%)", 17.1, 100 * corp_share, 1)
    n_nonmodal = int(t1.loc[t1["level"] == "non-modal (all)", "count"].iloc[0])
    chk("App A", "non-modal body-type policies", 4, n_nonmodal, 0)

    # ------------------------------------------------------------------ [2]
    log.info("\n[2/8] Portfolio composition (underwriting year, accident year)")
    uw = portfolio_by_uw_year(ana, claim_all, cpi_factor_by_year)
    write_csv(uw, "21_portfolio_by_uw_year", outfiles)
    uwy = uw[uw["underwriting_year"] != "total"]
    chk("Sec 2.1", "min policies per underwriting year", 134925,
        uwy["policies"].min(), 0)
    chk("Sec 2.1", "max policies per underwriting year", 196301,
        uwy["policies"].max(), 0)
    f14 = float(uwy.loc[uwy["underwriting_year"] == STUDY_YEAR_MIN,
                        "freq_per_exposure_linked_files"].iloc[0])
    f18 = float(uwy.loc[uwy["underwriting_year"] == STUDY_YEAR_MAX,
                        "freq_per_exposure_linked_files"].iloc[0])
    chk("Sec 2.1", "claim frequency per exposure 2014 (linked files)", 0.294,
        f14, 3)
    chk("Sec 2.1", "claim frequency per exposure 2018 (linked files)", 0.240,
        f18, 3)
    for _, r in uwy.iterrows():
        log.info(f"  UY {r['underwriting_year']}: policies={r['policies']:,} "
                 f"exposure={r['earned_exposure']:,.1f} "
                 f"claims(linked)={r['claims_linked_files']:,} "
                 f"(real>0: {r['claims_positive_real_payment']:,}) "
                 f"freq/exposure={r['freq_per_exposure_linked_files']:.4f}")
    ayt = claims_by_accident_year(claim_all)
    write_csv(ayt, "21_claims_by_accident_year", outfiles)
    ms_ay = {2014: 21416, 2015: 54316, 2016: 46298, 2017: 41312, 2018: 36287,
             2019: 14654}
    for ay, v in ms_ay.items():
        new = ayt.loc[ayt["accident_year"] == ay,
                      "claims_positive_real_payment"]
        chk("Sec 2.1", f"linked claims accident year {ay}", v,
            float(new.iloc[0]) if len(new) else None, 0)

    # ------------------------------------------------------------------ [5]
    # (regression before [3]: the composition effect uses β_N)
    log.info("\n[5/8] Fully covariate-adjusted log-severity regression (Table A1)")
    X, y, names, raw_num = design_matrix(claim)
    fit = ols(X, y)
    beta = fit["beta"]
    jN = names.index("n_claims")
    log.info(f"  n={fit['n']:,}, p={fit['p']}, β_N={beta[jN]:+.7f}, "
             f"R²={fit['r2']:.5f} (adj {fit['adj_r2']:.5f})")
    betas_b, unident, n_clusters = cluster_bootstrap_coefs(
        X, y, claim["policy_ref"].values)
    se_b = np.nanstd(betas_b, axis=0, ddof=1)
    lo_b, hi_b = np.nanpercentile(betas_b, [2.5, 97.5], axis=0)
    frac_unid = unident.mean(axis=0)
    log.info(f"  Cluster bootstrap B={R1_REG_BOOT_B} (seed {R1_REG_BOOT_SEED}), "
             f"{n_clusters:,} policy clusters: β_N CI "
             f"[{lo_b[jN]:+.6f}, {hi_b[jN]:+.6f}], SE={se_b[jN]:.6f}")
    # consistency: 06 manifest
    axD = m06.get("axis_D_direct", {})
    cons06 = {}
    if axD:
        cons06 = {
            "beta_N_06": axD.get("beta_N"), "ci_06": axD.get("ci"),
            "abs_diff_beta_N": abs(float(axD["beta_N"]) - float(beta[jN])),
            "abs_diff_ci": [abs(float(axD["ci"][0]) - float(lo_b[jN])),
                            abs(float(axD["ci"][1]) - float(hi_b[jN]))]}
        cons06["identical_to_06"] = bool(
            cons06["abs_diff_beta_N"] < 1e-10
            and max(cons06["abs_diff_ci"]) < 1e-10)
        log.info(f"  Consistency with the 06 manifest: β_N difference "
                 f"{cons06['abs_diff_beta_N']:.2e}, CI difference "
                 f"{max(cons06['abs_diff_ci']):.2e} → "
                 f"{'IDENTICAL' if cons06['identical_to_06'] else 'DIFFERENT'}")
    # degenerate terms + VIF (full design)
    D_full = X[:, 1:]
    vif_full = vif_table(D_full, names[1:])
    red_names = ["n_claims", "log_exposure"] + list(NUMERIC_COVARS)
    D_red = np.column_stack([X[:, names.index(nm)] for nm in red_names])
    vif_red = vif_table(D_red, red_names)
    pol_ids = claim["policy_ref"].values
    rows = []
    for j, nm in enumerate(names):
        col = X[:, j]
        is_dummy = (nm.startswith("AY") or any(nm.startswith(c + "_")
                                               for c in CATEG_COVARS))
        nz = col != 0
        n_nz = int(nz.sum()) if is_dummy else np.nan
        n_pol_nz = int(pd.unique(pol_ids[nz]).size) if is_dummy else np.nan
        degenerate = bool(is_dummy and n_nz <= R1_DEGENERATE_MIN_CLAIMS)
        flag = ""
        if degenerate:
            flag = (f"DEGENERATE: identified by {n_nz} claim(s) on "
                    f"{n_pol_nz} policy(ies); column is all-zero in "
                    f"{frac_unid[j]:.1%} of bootstrap replicates (min-norm "
                    "lstsq sets the coefficient to 0 there), so the "
                    "percentile interval is not interpretable")
        if nm == "log_exposure":
            nzexp = int((pd.to_numeric(claim["exposure"], errors="coerce")
                         <= 0).sum())
            if nzexp:
                flag = (f"LEVERAGE: {nzexp} claim(s) on zero-exposure "
                        f"policies enter at log({R1_LOG_EXPOSURE_FLOOR:g}); "
                        "wide/asymmetric bootstrap interval")
        lab = R1_TERM_LABELS.get(nm)
        if lab is None:
            if nm.startswith("AY"):
                lab = f"Accident year {nm[2:]}"
            elif nm.startswith("Body_Code_"):
                lab = f"Body type: {nm.split('_')[-1]} (non-modal)"
            else:
                lab = nm
        rows.append({"term": nm, "label": lab, "beta": beta[j],
                     "se_cluster_boot": se_b[j], "ci_lo": lo_b[j],
                     "ci_hi": hi_b[j], "se_ols_iid": fit["se_iid"][j],
                     "vif_full_design": vif_full.get(nm, np.nan),
                     "vif_reduced_numeric_design": vif_red.get(nm, np.nan),
                     "n_claims_nonzero": n_nz, "n_policies_nonzero": n_pol_nz,
                     "boot_frac_column_all_zero": float(frac_unid[j]),
                     "flag": flag})
        ms = R1_MS_TABLE_A1.get(nm)
        if ms:
            b_ms, se_ms, lo_ms, hi_ms, dec = ms
            chk("Table A1", f"{nm} beta", b_ms, beta[j], dec)
            chk("Table A1", f"{nm} cluster SE", se_ms, se_b[j], dec)
            chk("Table A1", f"{nm} CI low", lo_ms, lo_b[j], dec)
            chk("Table A1", f"{nm} CI high", hi_ms, hi_b[j], dec)
    tA1 = pd.DataFrame(rows)
    tA1["n"] = fit["n"]; tA1["r2"] = fit["r2"]; tA1["B"] = R1_REG_BOOT_B
    write_csv(tA1, "21_fsadj_full_regression", outfiles)
    chk("Table A1", "R^2", 0.028, fit["r2"], 3)
    for nm, v in R1_MS_VIF.items():
        chk("App A VIF (manuscript basis = reduced numeric design)",
            f"VIF {nm} reduced", v, vif_red[nm], 2)
        chk("App A VIF (full Table A1 design)", f"VIF {nm} full", v,
            vif_full[nm], 2, "manuscript value vs full-design VIF")
    log.info("  VIF (full design): " + ", ".join(
        f"{k}={v:.2f}" for k, v in vif_full.items()))
    log.info("  VIF (reduced numeric): " + ", ".join(
        f"{k}={v:.2f}" for k, v in vif_red.items()))
    # correlations
    trio = ["Arac_Bedeli", "Horse_Power", "CCM"]
    corr_rows = {}
    for level, frame in [("policy", ana), ("claim", claim)]:
        M = np.corrcoef(np.vstack([pd.to_numeric(frame[c], errors="coerce")
                                   .fillna(0).values.astype(float)
                                   for c in trio]))
        corr_rows[level] = {"value_hp": M[0, 1], "value_cc": M[0, 2],
                            "hp_cc": M[1, 2]}
    for lvl in ("claim", "policy"):
        cp = corr_rows[lvl]
        chk("App A", f"min corr value/HP/cc ({lvl} level)", 0.56,
            min(cp.values()), 2, "manuscript basis = claim level (regression "
            "sample)" if lvl == "claim" else "policy level, for reference")
        chk("App A", f"max corr value/HP/cc ({lvl} level)", 0.79,
            max(cp.values()), 2, "manuscript basis = claim level (regression "
            "sample)" if lvl == "claim" else "policy level, for reference")
    _mx = max(vif_full.values())
    chk("Sec 3.2", "max VIF below 3.8 (full Table A1 design)", 3.8, _mx, 2,
        "inequality: max VIF < 3.8", match=_mx < 3.8)
    # sensitivities (point estimate)
    sens = {}
    keep_nobody = [j for j, nm in enumerate(names)
                   if not nm.startswith("Body_Code_")]
    f1 = ols(X[:, keep_nobody], y)
    sens["drop_body_type_dummy"] = {"beta_N": float(f1["beta"][jN]),
                                    "r2": float(f1["r2"]), "n": f1["n"]}
    mpos = (pd.to_numeric(claim["exposure"], errors="coerce") > 0).values
    f2 = ols(X[mpos], y[mpos])
    sens["exclude_zero_exposure_claims"] = {"beta_N": float(f2["beta"][jN]),
                                            "r2": float(f2["r2"]),
                                            "n": f2["n"]}
    keep_noexp = [j for j, nm in enumerate(names) if nm != "log_exposure"]
    f3 = ols(X[:, keep_noexp], y)
    sens["drop_log_exposure"] = {"beta_N": float(f3["beta"][jN]),
                                 "r2": float(f3["r2"]), "n": f3["n"]}
    for k_, v_ in sens.items():
        log.info(f"  Sensitivity [{k_}]: β_N={v_['beta_N']:+.6f}, "
                 f"R²={v_['r2']:.5f}, n={v_['n']:,}")
    drows = [{"section": "fit", "quantity": q, "value": v, "note": nt}
             for q, v, nt in [
                 ("n_claims", fit["n"], "claims with positive real payment"),
                 ("n_policy_clusters", n_clusters, ""),
                 ("n_parameters", fit["p"], "incl. intercept"),
                 ("r2", fit["r2"], "OLS on log real severity"),
                 ("adj_r2", fit["adj_r2"], ""),
                 ("bootstrap_B", R1_REG_BOOT_B, "policy-cluster, percentile"),
                 ("bootstrap_seed", R1_REG_BOOT_SEED, "config.GLOBAL_SEED")]]
    for nm, v in vif_full.items():
        drows.append({"section": "vif_full_design", "quantity": nm,
                      "value": v, "note": "all Table A1 regressors "
                      "(AY, gender, body dummies included)"})
    for nm, v in vif_red.items():
        drows.append({"section": "vif_reduced_numeric_design", "quantity": nm,
                      "value": v, "note": "count + log exposure + 5 numeric "
                      "covariates only (basis of the manuscript's VIFs)"})
    for lvl, d in corr_rows.items():
        for q, v in d.items():
            drows.append({"section": f"pearson_corr_{lvl}_level",
                          "quantity": q, "value": v,
                          "note": ("818,789 policies" if lvl == "policy"
                                   else f"{len(claim):,} severity claims")})
    for k_, v_ in sens.items():
        for q in ("beta_N", "r2", "n"):
            drows.append({"section": f"sensitivity_{k_}", "quantity": q,
                          "value": v_[q], "note": "point estimate, same "
                          "design otherwise"})
    for q, v in cons06.items():
        drows.append({"section": "consistency_with_06_manifest",
                      "quantity": q, "value": json.dumps(v)
                      if isinstance(v, list) else v, "note": ""})
    write_csv(pd.DataFrame(drows), "21_fsadj_diagnostics", outfiles)

    # ------------------------------------------------------------------ [3]
    log.info("\n[3/8] Claim-count distribution, q_k, tie bounds, composition")
    ccd, ccd_s = claim_count_distribution(ana)
    write_csv(ccd, "21_claim_count_distribution", outfiles)
    for lab, ms in [("0", 79.0), ("1", 16.8), ("2", 3.3), ("3+", 0.8)]:
        v = float(ccd.loc[ccd["quantity"] == f"share_policies_N={lab}",
                          "value"].iloc[0])
        chk("Sec 2.1", f"share of policies with N={lab} (%)", ms, 100 * v, 1)
    chk("Sec 2.1", "mean claim count", 0.262, ccd_s["mean"], 3)
    chk("Sec 2.1", "variance-to-mean ratio", 1.23, ccd_s["vmr"], 2)
    comp, comp_s = composition(claim, float(beta[jN]))
    write_csv(comp, "21_claim_composition", outfiles)
    chk("Sec 2.5", "tau-b bound sqrt(1-sum q^2)", 0.72,
        comp_s["tau_b_bound_asymptotic"], 2)
    chk("Sec 2.5", "tau-a bound 1-sum q^2", 0.516,
        comp_s["tau_a_bound_asymptotic"], 3)
    chk("Sec 3.2", "share of claims on multi-claim policies (%)", 35.7,
        100 * comp_s["share_claims_on_multiclaim_policies"], 1)
    for key in ("C1_severity_weighted_multiplicative",
                "C2_count_weighted_multiplicative",
                "C3_severity_weighted_loglinear",
                "C5_severity_weighted_pctlinear"):
        chk("Sec 3.2", f"composition effect |{key}| (%)", 1.8,
            100 * abs(comp_s[key]), 1)
    log.info("  N shares (policies): " + ", ".join(
        f"{r['quantity'].split('=')[1]}={100 * r['value']:.2f}%"
        for _, r in ccd.iterrows() if r["quantity"].startswith("share"))
        + f"; mean={ccd_s['mean']:.4f} VMR={ccd_s['vmr']:.4f}")
    log.info(f"  Σq²={comp_s['sum_qk2']:.5f} → τ-a ≤ "
             f"{comp_s['tau_a_bound_asymptotic']:.4f}, τ-b ≤ "
             f"{comp_s['tau_b_bound_asymptotic']:.4f}; multi-claim share="
             f"{comp_s['share_claims_on_multiclaim_policies']:.4%}")
    log.info("  Composition: " + ", ".join(
        f"{k}={comp_s[k]:+.4%}" for k in comp_s if k.startswith("C")))

    # ------------------------------------------------------------------ [4]
    log.info("\n[4/8] Severity statistics (real & nominal, same set)")
    sev, sev_s = severity_summary(claim_all)
    write_csv(sev, "21_severity_summary", outfiles)
    s1r = sev_s["S1_real_positive|real_Dec2018"]
    s1n = sev_s["S1_real_positive|nominal"]
    s2n = sev_s["S2_nominal_positive|nominal"]
    for stat, ms in [("mean", 4243), ("median", 1529), ("sd", 9895),
                     ("p90", 9870), ("p95", 17220), ("p99", 43152)]:
        chk("Sec 2.1", f"real severity {stat}", ms, s1r[stat], 0)
    chk("Sec 2.1", "nominal mean (same claim set as real)", 3309,
        s1n["mean"], 0, "manuscript value vs same-set value")
    chk("Sec 2.1", "nominal median (same claim set as real)", 1142,
        s1n["median"], 0, "manuscript value vs same-set value")
    chk("Sec 2.1", "nominal mean (nominal-positive set)", 3309, s2n["mean"], 0,
        "reproduces the manuscript: different claim set")
    chk("Sec 2.1", "nominal median (nominal-positive set)", 1142,
        s2n["median"], 0, "reproduces the manuscript: different claim set")
    log.info(f"  Real (n={s1r['n']:,}): mean={s1r['mean']:,.1f} med="
             f"{s1r['median']:,.1f} SD={s1r['sd']:,.1f} P90/95/99="
             f"{s1r['p90']:,.0f}/{s1r['p95']:,.0f}/{s1r['p99']:,.0f}")
    log.info(f"  Nominal, SAME set (n={s1n['n']:,}): mean={s1n['mean']:,.1f} "
             f"med={s1n['median']:,.1f}; nominal>0 set (n={s2n['n']:,}): "
             f"mean={s2n['mean']:,.1f} med={s2n['median']:,.1f}")

    # ------------------------------------------------------------------ [7]
    log.info("\n[7/8] Power: i.i.d. (tie-free / tie-corrected) and cluster MDE")
    z_a = float(stats.norm.ppf(1 - POWER_ALPHA / 2))
    z_b = float(stats.norm.ppf(POWER_TARGET))
    ax_tau06 = m06.get("axis_tau", {})
    ps08 = m08.get("power_summary", []) or []

    def row08(kind):
        for r in ps08:
            e = str(r.get("eksen", "")).lower()
            if kind == "FS" and ("freq" in e or e.startswith("a_")):
                return r
            if kind == "lag" and ("delay" in e or "lag" in e
                                  or e.startswith("c_")):
                return r
        return {}

    axes = {
        "FS": {"x": claim["n_claims"].values.astype(float),
               "y": claim["severity"].values.astype(float),
               "tau06": ax_tau06.get("FS_freq_severity"),
               "ci06": m06.get("axis_FS_bca_ci"), "r08": row08("FS"),
               "x_label": "policy claim count N"},
        "lag": {"x": claim["delay"].values.astype(float),
                "y": claim["severity"].values.astype(float),
                "tau06": ax_tau06.get("lag_settlement"),
                "ci06": m06.get("axis_lag_bca_ci"), "r08": row08("lag"),
                "x_label": "mean settlement delay (years)"},
    }
    prow, pinfo = [], {}
    for ax, d in axes.items():
        n = len(d["x"])
        kt = stats.kendalltau(d["x"], d["y"])
        tau_b = float(kt.statistic)
        tx, ty = tie_sums(d["x"]), tie_sums(d["y"])
        nx_only = kendall_null(n, tx, NO_TIES)
        nxy = kendall_null(n, tx, ty)
        # verification: the formula must reproduce SciPy's p-value
        S = tau_b * nxy["den_tau_b"]
        p_mine = float(2 * stats.norm.sf(abs(S / np.sqrt(nxy["var_S"]))))
        p_rel = abs(p_mine - float(kt.pvalue)) / max(float(kt.pvalue), 1e-300)
        tau_obs = float(d["tau06"]) if d["tau06"] is not None else tau_b
        r08 = d["r08"]
        se_cl = r08.get("SE_tau_cluster")
        se_cl_src = "08_manifest power_summary.SE_tau_cluster"
        if se_cl is None and d["ci06"]:
            se_cl = (d["ci06"][1] - d["ci06"][0]) / (2 * z_a)
            se_cl_src = "06_manifest cluster CI width / (2 z_0.975)"
        mde_cl_08 = r08.get("MDE_80pct_cluster")
        se0 = se_iid_noties(n)
        bases = [
            ("iid_no_ties_formula", se0,
             "sqrt(2(2n+5)/(9n(n-1))) (08 analytic benchmark)"),
            ("iid_exact_null_ties_in_x_only", nx_only["se_tau_b"],
             "Kendall (1970) null Var(S) with the observed ties in x, y "
             "treated as continuous; / sqrt((n0-Tx) n0)"),
            ("iid_exact_null_ties_in_x_and_y", nxy["se_tau_b"],
             "Kendall (1970) null Var(S) with observed ties in x AND y "
             "(= SciPy kendalltau p-value variance); / sqrt((n0-Tx)(n0-Ty))"),
            ("policy_cluster_bootstrap", se_cl, se_cl_src),
        ]
        for bname, se, src in bases:
            if se is None:
                continue
            mde = (z_a + z_b) * se
            zt = abs(tau_obs) / se
            prow.append({
                "axis": ax, "x": d["x_label"], "n": n, "tau_obs": tau_obs,
                "se_basis": bname, "SE": se, "MDE_80pct": mde,
                "abs_tau_over_MDE": abs(tau_obs) / mde,
                "design_effect_cluster_vs_this": ((se_cl / se) ** 2
                                                  if se_cl else np.nan),
                "power_at_obs_onesided_approx": float(stats.norm.cdf(zt - z_a)),
                "power_at_obs_twosided": float(stats.norm.cdf(zt - z_a)
                                               + stats.norm.cdf(-zt - z_a)),
                "MDE_80pct_from_08": (mde_cl_08 if bname ==
                                      "policy_cluster_bootstrap" else np.nan),
                "source": src})
        pinfo[ax] = {"n": n, "tau_b_scipy": tau_b, "tau_obs_06": tau_obs,
                     "ties_x": tx, "ties_y": ty,
                     "se_iid_noties": se0,
                     "se_null_ties_x_only": nx_only["se_tau_b"],
                     "se_null_ties_xy": nxy["se_tau_b"],
                     "se_cluster": _f(se_cl), "se_cluster_source": se_cl_src,
                     "mde_cluster_08": _f(mde_cl_08),
                     "pvalue_scipy": float(kt.pvalue),
                     "pvalue_from_formula": p_mine,
                     "pvalue_rel_diff": float(p_rel)}
        log.info(f"  [{ax}] n={n:,} τ-b={tau_b:+.6f} (06: {tau_obs:+.6f}); "
                 f"SE_iid(no ties)={se0:.6f}, SE_null(ties x)="
                 f"{nx_only['se_tau_b']:.6f}, SE_null(ties x,y)="
                 f"{nxy['se_tau_b']:.6f}, SE_cluster={se_cl:.6f}; "
                 f"SciPy p-value check: relative difference={p_rel:.2e}")
    pw = pd.DataFrame(prow)
    pw["z_alpha"] = z_a; pw["z_power"] = z_b
    write_csv(pw, "21_power_mde", outfiles)

    def pget(ax, basis, col):
        s = pw[(pw["axis"] == ax) & (pw["se_basis"] == basis)][col]
        return float(s.iloc[0]) if len(s) else None
    chk("Sec 2.4", "SE0 (iid, no ties)", 0.00144,
        pget("FS", "iid_no_ties_formula", "SE"), 5)
    chk("Sec 2.4", "MDE iid", 0.0040,
        pget("FS", "iid_no_ties_formula", "MDE_80pct"), 4)
    chk("Sec 2.4", "cluster MDE FS", 0.0045,
        pget("FS", "policy_cluster_bootstrap", "MDE_80pct"), 4)
    chk("Sec 2.4", "cluster MDE lag", 0.0049,
        pget("lag", "policy_cluster_bootstrap", "MDE_80pct"), 4)
    chk("Sec 2.4", "design effect FS (vs no-ties iid)", 1.24,
        pget("FS", "iid_no_ties_formula", "design_effect_cluster_vs_this"), 2)
    chk("Sec 2.4", "design effect lag (vs no-ties iid)", 1.50,
        pget("lag", "iid_no_ties_formula", "design_effect_cluster_vs_this"), 2)
    chk("Sec 3.2", "lag tau / cluster MDE", 3.8,
        pget("lag", "policy_cluster_bootstrap", "abs_tau_over_MDE"), 1)
    _pw_cl = pget("FS", "policy_cluster_bootstrap",
                  "power_at_obs_onesided_approx")
    chk("Sec 3.2", "power at |tau_FS| exceeds 0.99 (cluster SE)", 0.99,
        _pw_cl, 3, "manuscript: '> 0.99' (claim tested as inequality); "
        "the 08 value 0.995 used the iid no-ties SE",
        match=(_pw_cl is not None and _pw_cl > 0.99))
    _pw_lag = pget("lag", "policy_cluster_bootstrap",
                   "power_at_obs_onesided_approx")
    chk("Sec 3.2", "power at |tau_lag| exceeds 0.99 (cluster SE)", 0.99,
        _pw_lag, 3, "inequality", match=(_pw_lag is not None
                                          and _pw_lag > 0.99))

    # ------------------------------------------------------------------ [6]
    log.info("\n[6/8] TOST equivalence test (τ_FS)")
    tau_fs = float(ax_tau06.get("FS_freq_severity",
                                pinfo["FS"]["tau_b_scipy"]))
    se_fs = pinfo["FS"]["se_cluster"]
    se_fs_ties = pinfo["FS"]["se_null_ties_x_only"]
    z95 = float(stats.norm.ppf(1 - R1_TOST_ALPHA))
    mde_fs = pget("FS", "policy_cluster_bootstrap", "MDE_80pct")
    trows = []
    for D in R1_TOST_MARGINS:
        rec = {"margin": D, "tau_FS": tau_fs, "SE_cluster": se_fs}
        for tag, se in [("cluster", se_fs), ("iid_ties", se_fs_ties)]:
            zl = (tau_fs + D) / se     # H0: τ ≤ −Δ
            zu = (D - tau_fs) / se     # H0: τ ≥ +Δ
            lpl = float(stats.norm.logsf(zl)); lpu = float(stats.norm.logsf(zu))
            lp = max(lpl, lpu)
            sfx = "" if tag == "cluster" else "_iid_ties"
            rec.update({f"z_lower{sfx}": zl, f"p_lower{sfx}": float(np.exp(lpl)),
                        f"z_upper{sfx}": zu, f"p_upper{sfx}": float(np.exp(lpu)),
                        f"p_tost{sfx}": float(np.exp(lp)),
                        f"log10_p_tost{sfx}": lp / np.log(10),
                        f"equivalent_5pct{sfx}": bool(np.exp(lp) < R1_TOST_ALPHA)})
        rec["ci90_lo"] = tau_fs - z95 * se_fs
        rec["ci90_hi"] = tau_fs + z95 * se_fs
        rec["margin_over_cluster_MDE"] = D / mde_fs if mde_fs else np.nan
        rec["SE_iid_ties"] = se_fs_ties
        trows.append(rec)
    tost = pd.DataFrame(trows)
    d_min = abs(tau_fs) + z95 * se_fs
    d_min_ties = abs(tau_fs) + z95 * se_fs_ties
    tost["smallest_margin_equivalent_5pct"] = d_min
    tost["smallest_margin_equivalent_5pct_iid_ties"] = d_min_ties
    tost["se_source"] = pinfo["FS"]["se_cluster_source"]
    write_csv(tost, "21_tost_equivalence", outfiles)
    ref = tost[np.isclose(tost["margin"], R1_TOST_REF_MARGIN)].iloc[0]
    log.info(f"  τ_FS={tau_fs:+.6f}, SE_cluster={se_fs:.6f}: Δ=0.02 → "
             f"p_TOST={ref['p_tost']:.3e}; smallest equivalence margin="
             f"{d_min:.5f} (with the tie-corrected iid SE {d_min_ties:.5f})")
    for _, r in tost.iterrows():
        log.info(f"    Δ={r['margin']:.4f}: p_TOST={r['p_tost']:.3e} "
                 f"({'equivalent' if r['equivalent_5pct'] else 'NOT equivalent'})")
    chk("Sec 3.2", "TOST p at Delta=0.02 below 1e-16", 1e-16,
        float(ref["p_tost"]), 0, "inequality: p < 1e-16",
        match=float(ref["p_tost"]) < 1e-16)
    chk("Sec 2.4/3.2", "equivalence holds for every margin >= 0.01 "
        "(smallest equivalence margin)", 0.01, d_min, 4,
        "inequality: smallest margin <= 0.01", match=d_min <= 0.01)

    # ------------------------------------------------------------------ [8]
    log.info("\n[8/8] Data-cleaning accounting")
    clean_df, clean_info = cleaning_accounting(paid_nom, paid, res, ana,
                                               claim_all, claim)
    write_csv(clean_df, "21_cleaning_accounting", outfiles)
    for _, r in clean_df.iterrows():
        log.info(f"  [{r['stage']}] {r['rule'][:70]}: removed="
                 f"{r['records_removed']}")

    # ------------------------------------------------------------ comparison
    chkdf = pd.DataFrame(checks)
    write_csv(chkdf, "21_manuscript_check", outfiles)
    n_mis = int((chkdf["match_at_quoted_precision"] == False).sum())  # noqa
    log.info(f"\n  Manuscript comparison: {len(chkdf)} values, "
             f"{n_mis} DO NOT MATCH at the quoted precision "
             "(see outputs/21_manuscript_check.csv)")

    # --------------------------------------------------------------- manifest
    manifest = {
        "script": SCRIPT_NAME,
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "purpose": ("R1 revision-stage quantities (Table 1, portfolio "
                    "composition, claim-count composition and tie bounds, "
                    "severity statistics, Table A1 with VIFs, TOST, power/MDE, "
                    "data-cleaning accounting) computed from data/proc; "
                    "aggregates only."),
        "params": {
            "reg_bootstrap_B": R1_REG_BOOT_B,
            "reg_bootstrap_seed": R1_REG_BOOT_SEED,
            "reg_bootstrap_scheme": ("policy-cluster (policies resampled with "
                                     "replacement, all claims kept); same rng "
                                     "and cluster order as 06 axis_D_direct"),
            "reg_se_definition": "SD of bootstrap coefficients, ddof=1",
            "reg_ci_definition": "percentile 2.5/97.5",
            "log_exposure_floor": R1_LOG_EXPOSURE_FLOOR,
            "std_eps": R1_STD_EPS,
            "desc_quantiles": list(R1_DESC_Q),
            "sev_quantiles": list(R1_SEV_Q),
            "quantile_method": "linear interpolation (pandas/numpy default)",
            "sd_definition": "ddof=1",
            "tost_margins": list(R1_TOST_MARGINS),
            "tost_reference_margin": R1_TOST_REF_MARGIN,
            "tost_alpha_each_side": R1_TOST_ALPHA,
            "power_alpha": POWER_ALPHA, "power_target": POWER_TARGET,
            "qk_topcode": R1_QK_TOPCODE,
            "degenerate_min_claims": R1_DEGENERATE_MIN_CLAIMS,
            "vehicle_value_price_basis": ("NOMINAL (policy-year TRY), not "
                                          "deflated in Table 1 or in the "
                                          "regression; real version is a "
                                          "labelled sensitivity row only"),
            "vehicle_value_deflation_sensitivity": (
                f"policy-year month {R1_VALUE_DEFLATE_MONTH:02d} CPI factor "
                f"to {INFLATION_BASE_YEAR_MONTH} from paid_long_real"),
            "cpi_factor_by_uw_year": cpi_factor_by_year,
            "tau_cluster_bootstrap_B_in_06": TAU_CLUSTER_N_BOOTSTRAP,
        },
        "design_choices": {
            "claims_basis_policy_level": (
                "N = distinct claim files per policy in the paid ledger "
                "(analytical_dataset.n_claims); the policy file has only a "
                "binary claim-status flag"),
            "severity_claim_set": "claim files with cumulative real payment > 0",
            "nominal_severity_same_set": True,
            "composition_effect_definitions": [
                "C1 severity-weighted exact multiplicative",
                "C2 claim-count-weighted exact multiplicative",
                "C3 severity-weighted log-linear",
                "C4 count-weighted log-linear",
                "C5 severity-weighted percent-linear",
                "C6 count-weighted percent-linear"],
            "vif_designs": {"full": "all Table A1 regressors",
                            "reduced": "count + log exposure + 5 numeric"},
            "kendall_null_variance": ("Kendall (1970) eq. 4.3 = SciPy "
                                      "kendalltau asymptotic p-value variance; "
                                      "verified against SciPy p-values"),
        },
        "table1": t1.to_dict(orient="records"),
        "portfolio_by_uw_year": uw.to_dict(orient="records"),
        "claims_by_accident_year": ayt.to_dict(orient="records"),
        "claim_count_distribution": ccd_s,
        "composition": comp_s,
        "severity": sev_s,
        "table_a1": {
            "n": fit["n"], "n_policy_clusters": n_clusters,
            "r2": fit["r2"], "adj_r2": fit["adj_r2"],
            "beta_N": float(beta[jN]),
            "beta_N_ci": [float(lo_b[jN]), float(hi_b[jN])],
            "beta_N_se_cluster": float(se_b[jN]),
            "pct_effect_per_claim": float(np.exp(beta[jN]) - 1),
            "coefficients": tA1.drop(columns=["n", "r2", "B"]).to_dict(
                orient="records"),
            "vif_full_design": vif_full, "vif_reduced_numeric": vif_red,
            "max_vif_full": max(v for v in vif_full.values()),
            "max_vif_reduced": max(vif_red.values()),
            "correlations": corr_rows,
            "sensitivity": sens,
            "consistency_with_06": cons06,
        },
        "tost": {"tau_FS": tau_fs, "se_cluster": se_fs,
                 "se_source": pinfo["FS"]["se_cluster_source"],
                 "smallest_margin_equivalent_5pct": d_min,
                 "smallest_margin_equivalent_5pct_iid_ties": d_min_ties,
                 "p_tost_reference_margin": float(ref["p_tost"]),
                 "log10_p_tost_reference_margin": float(ref["log10_p_tost"]),
                 "grid": tost.to_dict(orient="records")},
        "power": {"axes": pinfo, "table": pw.to_dict(orient="records"),
                  "z_alpha": z_a, "z_power": z_b},
        "cleaning": clean_info,
        "manuscript_check": {"n_values": int(len(chkdf)),
                             "n_mismatch_at_quoted_precision": n_mis},
        "output_files": outfiles,
        "environment": env_info(),
    }
    with open(OUTPUTS_DIR / "21_manifest.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(manifest), f, ensure_ascii=False, indent=2)
    log.info("\n" + "=" * 78)
    log.info("SCRIPT 21 COMPLETED")
    log.info("=" * 78)


if __name__ == "__main__":
    main()
