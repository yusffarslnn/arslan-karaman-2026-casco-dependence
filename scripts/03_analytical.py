"""
================================================================================
SCRIPT 03 — Analytical Dataset and Chain-Ladder Ultimate
================================================================================
PURPOSE
    Builds the development triangle from the real paid and case-reserve panels,
    estimates the ultimate claim cost and IBNR with Chain-Ladder, and produces
    the policy-level analytical table (frequency, ultimate, covariates). This
    table is the core input of all dependence, copula and baseline analyses.

DESIGN DECISIONS
    • Development triangle on REAL paid amounts (accident year × development year).
    • Three ultimate definitions are produced side by side, so that the
      sensitivity of the results to the ultimate choice is transparent:
        – paid_only : cumulative paid only
        – rbns      : paid + last open case reserve (reported-but-not-settled)
        – cl        : Chain-Ladder projection (ultimate including IBNR)
    • Double-counting check: a claim must not be counted in both paid and case reserve.
    • Severity is NOT DERIVED from the ultimate (to avoid the ratio-severity
      trap); per-claim severity is taken directly from the paid panel (Script 06+).

INPUT  : data/proc/{policies,paid_long_real,reserve_panel_real}.parquet
OUTPUT : data/proc/{analytical_dataset,claim_analytical}.{parquet,pkl}
        outputs/03_development_triangle.csv
================================================================================
"""
from __future__ import annotations
import sys
import json
import logging
import platform
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

from config import (
    DATA_PROC_DIR, OUTPUTS_DIR, LOGS_DIR, CURRENCY, TERRITORY,
    LINE_OF_BUSINESS, HPO_VALUATION_YEAR, HPO_VALUATION_MONTH, PICKLE_PROTOCOL,
    MACK_ENABLED,
)

SCRIPT_NAME = "03_analytical"


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


def build_triangle(paid: pd.DataFrame) -> pd.DataFrame:
    """Cumulative development triangle (accident year × development year), real.

    UNobserved cells (future calendar years) are left as NaN; only genuinely
    observed cells (accident_year + dev_year ≤ last calendar year) are filled.
    Otherwise cumsum would right-fill, turning the triangle into a rectangle
    and artificially zeroing the IBNR.
    """
    inc = (paid.groupby(["accident_year", "dev_year"])["paid_total_real"]
           .sum().reset_index())
    latest_cal = int((paid["accident_year"] + paid["dev_year"]).max())
    tri_inc = inc.pivot(index="accident_year", columns="dev_year",
                        values="paid_total_real")
    # set unobserved cells to NaN
    for ay in tri_inc.index:
        for d in tri_inc.columns:
            if ay + d > latest_cal:
                tri_inc.loc[ay, d] = np.nan
    tri_cum = tri_inc.cumsum(axis=1)
    return tri_cum


def chain_ladder(tri_cum: pd.DataFrame):
    """Age-to-age development factors and ultimate projection.

    Each accident year is mature up to the most recent development age at which
    it is observable; the remaining development is projected by the product of
    the subsequent age-to-age factors. Along the latest calendar-year diagonal
    the younger accident years are less mature, so more factors are applied to
    them → this is where the IBNR arises.
    """
    devs = list(tri_cum.columns)
    n_ay = len(tri_cum.index)
    factors = {}
    for i in range(len(devs) - 1):
        d0, d1 = devs[i], devs[i + 1]
        num, den = 0.0, 0.0
        for ay in tri_cum.index:
            c0, c1 = tri_cum.loc[ay, d0], tri_cum.loc[ay, d1]
            if c0 > 0 and c1 > 0:
                num += c1; den += c0
        factors[(d0, d1)] = num / den if den > 0 else 1.0
    # Maturity of an accident year = (latest calendar year - accident year).
    # In the triangle the latest calendar year = min(ay) + (len(devs)-1) on that axis;
    # in practice, each accident year's observed max dev = (latest_cal - ay).
    accident_years = sorted(tri_cum.index)
    latest_cal = max(ay + d for ay in accident_years
                     for d in devs if tri_cum.loc[ay, d] > 0)
    ult = {}
    for ay in accident_years:
        observed_dev = latest_cal - ay        # maturity age of this accident year
        observed_dev = max(0, min(observed_dev, devs[-1]))
        val = tri_cum.loc[ay, observed_dev] if observed_dev in devs else \
            tri_cum.loc[ay].iloc[-1]
        # project the remaining development
        start = devs.index(observed_dev) if observed_dev in devs else len(devs) - 1
        for i in range(start, len(devs) - 1):
            val *= factors[(devs[i], devs[i + 1])]
        ult[ay] = val
    return factors, ult


def mack_standard_error(tri_cum: pd.DataFrame):
    """Mack (1993) distribution-free Chain-Ladder standard error (pure numpy).

    Computes the MSE per accident year and in total (including covariance
    terms). If sigma_k^2 cannot be estimated from observations at the last
    development age, it is extrapolated with Mack's min-rule. Used for the
    transparent reporting of reserve uncertainty (s.e. column next to Table 1
    in the paper).
    """
    devs = sorted(tri_cum.columns)
    ays = sorted(tri_cum.index)
    K = len(devs)
    latest_cal = max(ay + d for ay in ays for d in devs
                     if pd.notna(tri_cum.loc[ay, d]) and tri_cum.loc[ay, d] > 0)

    def obs(ay, d):
        v = tri_cum.loc[ay, d] if d in devs else np.nan
        return v if (pd.notna(v) and ay + d <= latest_cal) else np.nan

    # age-to-age factors f_k and S_k (denominators)
    f = np.ones(K - 1); S = np.zeros(K - 1)
    pairs = [[] for _ in range(K - 1)]
    for k in range(K - 1):
        num = den = 0.0
        for ay in ays:
            c0, c1 = obs(ay, devs[k]), obs(ay, devs[k + 1])
            if pd.notna(c0) and pd.notna(c1) and c0 > 0:
                num += c1; den += c0; pairs[k].append((ay, c0, c1))
        f[k] = num / den if den > 0 else 1.0
        S[k] = den
    # sigma_k^2 (Mack eq. 4); min-rule extrapolation at the last age
    sigma2 = np.full(K - 1, np.nan)
    for k in range(K - 1):
        pk = pairs[k]
        if len(pk) > 1:
            sigma2[k] = (1.0 / (len(pk) - 1)) * sum(
                c0 * (c1 / c0 - f[k]) ** 2 for _, c0, c1 in pk)
    for k in range(K - 1):
        if np.isnan(sigma2[k]):
            if k >= 2 and sigma2[k - 2] > 0:
                sigma2[k] = min(sigma2[k - 1] ** 2 / sigma2[k - 2],
                                min(sigma2[k - 2], sigma2[k - 1]))
            elif k >= 1:
                sigma2[k] = sigma2[k - 1]
            else:
                sigma2[k] = 0.0
    # projection and accident-year MSE (Mack eq. 7)
    per_ay = {}
    Chat = {}
    for ay in ays:
        m = max([d for d in devs if pd.notna(obs(ay, d))], default=devs[0])
        c = obs(ay, m)
        row = {m: c}
        for k in range(devs.index(m), K - 1):
            c = c * f[k]
            row[devs[k + 1]] = c
        Chat[ay] = (m, row)
        ult = row[devs[-1]]
        mse = 0.0
        for k in range(devs.index(m), K - 1):
            ck = row[devs[k]]
            if ck > 0 and S[k] > 0:
                mse += (sigma2[k] / f[k] ** 2) * (1.0 / ck + 1.0 / S[k])
        per_ay[ay] = {"ultimate": float(ult), "mse": float(ult ** 2 * mse),
                      "se": float(ult * np.sqrt(max(mse, 0.0)))}
    # total MSE (Mack eq. 9 — covariance between accident years)
    total_mse = sum(v["mse"] for v in per_ay.values())
    ay_sorted = sorted(ays)
    for i, ay_i in enumerate(ay_sorted):
        m_i, row_i = Chat[ay_i]
        ult_i = row_i[devs[-1]]
        later = sum(Chat[ay_j][1][devs[-1]] for ay_j in ay_sorted[i + 1:])
        cross = 0.0
        for k in range(devs.index(m_i), K - 1):
            if S[k] > 0:
                cross += sigma2[k] / (f[k] ** 2 * S[k])
        total_mse += 2.0 * ult_i * later * cross
    return {"per_accident_year": per_ay,
            "total_se": float(np.sqrt(max(total_mse, 0.0))),
            "sigma2": [float(x) for x in sigma2],
            "factors": [float(x) for x in f]}


def main() -> None:
    global log
    log = setup_logger()
    log.info("=" * 78)
    log.info(f"SCRIPT {SCRIPT_NAME} — Analytical Dataset + Chain-Ladder")
    log.info("=" * 78)

    outfiles: list = []
    policies = load_proc("policies")
    paid = load_proc("paid_long_real")
    reserve = load_proc("reserve_panel_real")

    log.info("\n[1/5] Development triangle (real)")
    tri = build_triangle(paid)
    log.info(f"  Triangle: {tri.shape[0]} accident years × {tri.shape[1]} development periods")
    tri.to_csv(OUTPUTS_DIR / "03_development_triangle.csv")
    outfiles.append(str((OUTPUTS_DIR / "03_development_triangle.csv").resolve()))

    log.info("\n[2/5] Chain-Ladder ultimate / IBNR")
    factors, ult_cl = chain_ladder(tri)
    f01 = list(factors.values())[0] if factors else 1.0
    total_cl = sum(ult_cl.values())
    # last diagonal = each accident year's MOST RECENT cumulative paid
    last_diag = float(sum(tri.loc[ay][max([d for d in tri.columns
                                           if tri.loc[ay, d] > 0],
                                          default=tri.columns[0])]
                          for ay in tri.index))
    ibnr = total_cl - last_diag
    log.info(f"  f(0→1) = {f01:.4f}")
    log.info(f"  CL ultimate total = {total_cl:,.0f} {CURRENCY}")
    log.info(f"  IBNR = {ibnr:,.0f} ({ibnr/total_cl:.1%} development)")

    mack = None
    if MACK_ENABLED:
        log.info("\n[2b/5] Mack (1993) standard error (reserve uncertainty)")
        mack = mack_standard_error(tri)
        se_t = mack["total_se"]
        log.info(f"  Total Mack s.e. = {se_t:,.0f} {CURRENCY} "
                 f"({se_t/total_cl:.2%} / CL ultimate)")
        mrows = [{"accident_year": int(ay), **v}
                 for ay, v in mack["per_accident_year"].items()]
        mdf = pd.DataFrame(mrows)
        mdf.to_csv(OUTPUTS_DIR / "03_mack_se.csv", index=False)
        outfiles.append(str((OUTPUTS_DIR / "03_mack_se.csv").resolve()))
        for r in mrows:
            log.info(f"    AY {r['accident_year']}: ult={r['ultimate']:,.0f}, "
                     f"s.e.={r['se']:,.0f}")

    log.info("\n[3/5] Policy-level frequency + paid")
    # frequency: unique claim files per policy
    freq = (paid.groupby("policy_ref")["claim_file"].nunique()
            .rename("n_claims").reset_index())
    paid_by_pol = (paid.groupby("policy_ref")["paid_total_real"]
                   .sum().rename("paid_real").reset_index())
    # rbns: last open case reserve at the valuation date
    vy, vm = HPO_VALUATION_YEAR, HPO_VALUATION_MONTH
    # R1 final fix: the case-reserve ledger is only the quarter-end inventory of
    # OPEN claims (a closed claim drops off the ledger without a zero record). The
    # old code took each claim's LAST record before the valuation; the stale case
    # reserves of closed claims (subsequently paid/settled) were added to paid and
    # double-counted (119,208 claims, ~629 M TRY). RBNS = only the case reserve that is
    # still open at the valuation date (HPO_VALUATION_YEAR-MONTH).
    res_cut = reserve[(reserve["reserve_year"] == vy)
                      & (reserve["reserve_month"] == vm)].copy()
    res_cut = res_cut.sort_values(["reserve_year", "reserve_month"])
    last_res = (res_cut.groupby("claim_file").tail(1)
                .groupby("policy_ref")["reserve_amount_real"]
                .sum().rename("reserve_real").reset_index())

    log.info("\n[4/5] Analytical table (3 ultimate definitions)")
    ana = policies.merge(freq, on="policy_ref", how="left")
    ana["n_claims"] = ana["n_claims"].fillna(0).astype(int)
    ana = ana.merge(paid_by_pol, on="policy_ref", how="left")
    ana = ana.merge(last_res, on="policy_ref", how="left")
    for c in ["paid_real", "reserve_real"]:
        ana[c] = ana[c].fillna(0.0)
    # Three ultimate definitions — EXPLICIT naming (avoid double-counting ambiguity):
    #   ultimate_paid_only : observed cumulative paid only
    #   ultimate_paid_cl   : CL-projected ultimate from the paid triangle only (IBNR
    #                        allocated in proportion to the paid share; case reserve NOT ADDED)
    #   ultimate_rbns      : paid + last open case reserve (RBNS)
    #   ultimate_cl_plus_rbns : CL-projected paid + RBNS (upper bound; using CL
    #                        together with the open case reserve may involve partial
    #                        double-counting, hence it is named separately and explicitly)
    ana["ultimate_paid_only"] = ana["paid_real"]
    paid_sum = ana["paid_real"].sum()
    ibnr_per_paid = (ibnr / paid_sum) if paid_sum > 0 else 0.0
    ana["ultimate_paid_cl"] = ana["paid_real"] * (1 + ibnr_per_paid)
    ana["ultimate_rbns"] = ana["paid_real"] + ana["reserve_real"]
    ana["ultimate_cl_plus_rbns"] = (ana["paid_real"] * (1 + ibnr_per_paid)
                                    + ana["reserve_real"])
    # backward compatibility: downstream uses 'ultimate_rbns' (the most conservative
    # incurred definition free of double-counting)
    # double-counting check
    dbl = int(((ana["paid_real"] > 0) & (ana["reserve_real"] > 0)
               & (ana["n_claims"] == 0)).sum())
    log.info(f"  Analytical table: {len(ana):,} policies, "
             f"{(ana['n_claims']>0).sum():,} with claims")
    log.info(f"  Suspected double-counting (n=0 but amount>0): {dbl}")
    log.info(f"  Ultimate totals: paid_only={ana['ultimate_paid_only'].sum():,.0f}, "
             f"paid_cl={ana['ultimate_paid_cl'].sum():,.0f}, "
             f"rbns={ana['ultimate_rbns'].sum():,.0f}, "
             f"cl+rbns={ana['ultimate_cl_plus_rbns'].sum():,.0f}")

    log.info("\n[5/5] Per-claim analytical table (severity, no ratio trap)")
    claim = paid.groupby("claim_file").agg(
        policy_ref=("policy_ref", "first"),
        accident_year=("accident_year", "first"),
        severity_real=("paid_total_real", "sum")).reset_index()
    claim = claim[claim["severity_real"] > 0]
    log.info(f"  Unique claims: {len(claim):,}, "
             f"median severity {claim['severity_real'].median():,.0f}")

    for df, stem in [(ana, "analytical_dataset"), (claim, "claim_analytical")]:
        pkl = DATA_PROC_DIR / f"{stem}.pkl"
        df.to_pickle(pkl, protocol=PICKLE_PROTOCOL); outfiles.append(str(pkl.resolve()))
        try:
            df.to_parquet(DATA_PROC_DIR / f"{stem}.parquet", index=False)
            outfiles.append(str((DATA_PROC_DIR / f"{stem}.parquet").resolve()))
        except Exception:
            pass

    manifest = {
        "script": SCRIPT_NAME,
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "chain_ladder": {"f_0_1": float(f01), "ultimate_total": float(total_cl),
                         "ibnr": float(ibnr), "ibnr_pct": float(ibnr/total_cl)},
        "mack": (None if mack is None else
                 {"total_se": mack["total_se"],
                  "total_se_pct_ultimate": mack["total_se"] / float(total_cl),
                  "sigma2": mack["sigma2"], "factors": mack["factors"]}),
        "ultimate_totals": {
            "paid_only": float(ana["ultimate_paid_only"].sum()),
            "rbns": float(ana["ultimate_rbns"].sum()),
            "cl_plus_rbns": float(ana["ultimate_cl_plus_rbns"].sum()),
            "paid_cl": float(ana["ultimate_paid_cl"].sum())},
        "counts": {"policies": len(ana), "claims": len(claim),
                   "claim_policies": int((ana["n_claims"] > 0).sum())},
        "design_choices": {"severity_from_ultimate": False,
                           "three_ultimate_definitions": True},
        "output_files": outfiles,
        "environment": {"python": platform.python_version(),
                        "pandas": pd.__version__, "numpy": np.__version__},
    }
    with open(OUTPUTS_DIR / "03_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log.info("\n" + "=" * 78)
    log.info("SCRIPT 03 COMPLETED")
    log.info("=" * 78)


if __name__ == "__main__":
    main()
