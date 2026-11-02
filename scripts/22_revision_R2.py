"""
================================================================================
SCRIPT 22 — R2 Revision Analyses: Incurred Triangle and Case-Reserve Run-off
================================================================================
PURPOSE (Reviewer 1, round 2, point 4 — reserving)
    The reviewer noted that the gap between the paid chain-ladder reserve
    (TRY 689 thousand) and the open-file case reserves at the valuation date
    (TRY 9.4 M) weakens the practical meaning of the reserving section. This
    module explains the gap with data:
    [1] Incurred triangle: accident year × development year, real; cell = cumulative
        paid (through the calendar year-end) + OPEN case reserves in that year-end
        (December) snapshot (Dec-2018 prices). The same CL and Mack (functions in
        03_analytical) are applied to the incurred triangle; the paid triangle is
        reproduced with the same functions and verified against 03.
    [2] Case-reserve run-off: for each December snapshot (2014–2019), the case
        reserve held for open files is compared with the amount actually paid on
        those files SUBSEQUENTLY (through end-2020) (paid/case-reserve ratio, share
        of files with no payment at all, share still open). Shows to what extent
        case reserves have historically converted into payments.
    [3] Accident-year distribution of the 201 files open at the valuation date
        (Dec-2020) and the expected payment at the historical run-off ratio.

    Only data/proc is read; outputs are AGGREGATE tables only. There is no
    randomness (deterministic); no seed is needed.

INPUT : data/proc/{paid_long_real,reserve_panel_real}.parquet ;
        data/raw/muallak.parquet (only Hasar_Tarihi → accident year) ;
        outputs/03_development_triangle.csv, 03_mack_se.csv (cross-check)
OUTPUT: outputs/22_*.csv ; outputs/22_manifest.json ; logs/22_revision_R2.log
================================================================================
"""
from __future__ import annotations
import sys
import json
import logging
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

from config import (DATA_PROC_DIR, OUTPUTS_DIR, LOGS_DIR, HPO_VALUATION_YEAR,
                    HPO_VALUATION_MONTH, CURRENCY, env_info)
try:
    from config import DATA_RAW_DIR
except ImportError:  # pragma: no cover
    DATA_RAW_DIR = DATA_PROC_DIR.parent / "raw"

SCRIPT_NAME = "22_revision_R2"
R2_MATURE_RUNOFF_MAX_YEAR = 2017   # last snapshot whose run-off ratio counts as "mature"
                                   # (≥3-year observation window)


def setup_logger() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(SCRIPT_NAME)
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOGS_DIR / f"{SCRIPT_NAME}.log", mode="w",
                             encoding="utf-8"); fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger


log = logging.getLogger(SCRIPT_NAME)


def load_proc(stem: str) -> pd.DataFrame:
    pq = DATA_PROC_DIR / f"{stem}.parquet"
    return pd.read_parquet(pq) if pq.exists() else pd.read_pickle(DATA_PROC_DIR / f"{stem}.pkl")


def import_03():
    """The chain_ladder / mack_standard_error functions of 03_analytical.py
    (via importlib because the file name starts with a digit)."""
    spec = importlib.util.spec_from_file_location("analytical03", _HERE / "03_analytical.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def excel_serial_year(s: pd.Series) -> pd.Series:
    d = pd.to_datetime("1899-12-30") + pd.to_timedelta(pd.to_numeric(s, errors="coerce"), unit="D")
    return d.dt.year


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


def main() -> None:
    global log
    log = setup_logger()
    log.info("=" * 78)
    log.info(f"SCRIPT {SCRIPT_NAME} — Incurred triangle and case-reserve run-off (R2)")
    log.info("=" * 78)
    outfiles: list = []
    a03 = import_03()
    vy, vm = HPO_VALUATION_YEAR, HPO_VALUATION_MONTH

    paid = load_proc("paid_long_real")
    res = load_proc("reserve_panel_real")
    raw_res = pd.read_parquet(DATA_RAW_DIR / "muallak.parquet",
                              columns=["Hasar_Dosya_No", "Police_Ref", "Hasar_Tarihi"])

    # ------------------------------------------------------------------ [1a]
    log.info("\n[1/5] Accident year of case-reserve files (raw Hasar_Tarihi)")
    raw_res["ay_raw"] = excel_serial_year(raw_res["Hasar_Tarihi"])
    ay_map = (raw_res.dropna(subset=["ay_raw"])
              .groupby(["Police_Ref", "Hasar_Dosya_No"])["ay_raw"].min()
              .rename("accident_year").reset_index()
              .rename(columns={"Police_Ref": "policy_ref", "Hasar_Dosya_No": "claim_file"}))
    ay_map["accident_year"] = ay_map["accident_year"].astype(int)
    res = res.merge(ay_map, on=["policy_ref", "claim_file"], how="left")
    n_missing = int(res["accident_year"].isna().sum())
    # cross-check: agreement with the accident year derived from the paid panel
    ay_paid = (paid.groupby(["policy_ref", "claim_file"])["accident_year"].min()
               .rename("ay_paid").reset_index())
    chk = res.merge(ay_paid, on=["policy_ref", "claim_file"], how="left")
    both = chk.dropna(subset=["accident_year", "ay_paid"])
    agree = float((both["accident_year"] == both["ay_paid"]).mean()) if len(both) else np.nan
    log.info(f"  case-reserve rows {len(res):,}; accident year missing {n_missing}; "
             f"agreement with paid panel {agree:.4%} ({len(both):,} rows)")
    res = res.dropna(subset=["accident_year"])
    res["accident_year"] = res["accident_year"].astype(int)
    # December snapshots, one row per file
    dec = res[res["reserve_month"] == 12].copy()
    n_before = len(dec)
    dec = dec.groupby(["reserve_year", "policy_ref", "claim_file"], as_index=False).agg(
        accident_year=("accident_year", "first"),
        reserve_amount_real=("reserve_amount_real", "last"))
    log.info(f"  December snapshot rows {n_before:,} → unique files {len(dec):,}")

    # ------------------------------------------------------------------ [1b]
    log.info("\n[2/5] Paid and incurred triangles (real, accident year × development)")
    ays = sorted(int(a) for a in paid["accident_year"].unique())
    devs = list(range(0, vy - min(ays) + 1))
    paid_cum = pd.DataFrame(np.nan, index=ays, columns=devs, dtype=float)
    case_tri = pd.DataFrame(np.nan, index=ays, columns=devs, dtype=float)
    for ay in ays:
        p_ay = paid[paid["accident_year"] == ay]
        d_ay = dec[dec["accident_year"] == ay]
        for d in devs:
            cal = ay + d
            if cal > vy:
                continue
            paid_cum.loc[ay, d] = float(p_ay.loc[p_ay["payment_year"] <= cal, "paid_total_real"].sum())
            case_tri.loc[ay, d] = float(d_ay.loc[d_ay["reserve_year"] == cal, "reserve_amount_real"].sum())
    inc_tri = paid_cum + case_tri
    paid_cum.to_csv(OUTPUTS_DIR / "22_paid_triangle_check.csv")
    case_tri.to_csv(OUTPUTS_DIR / "22_case_reserve_triangle.csv")
    inc_tri.to_csv(OUTPUTS_DIR / "22_incurred_triangle.csv")
    outfiles += [str((OUTPUTS_DIR / f).resolve()) for f in
                 ("22_paid_triangle_check.csv", "22_case_reserve_triangle.csv",
                  "22_incurred_triangle.csv")]
    # cross-check against 03
    tri03 = pd.read_csv(OUTPUTS_DIR / "03_development_triangle.csv", index_col=0)
    tri03.columns = [int(c) for c in tri03.columns]; tri03.index = [int(i) for i in tri03.index]
    diff03 = float(np.nanmax(np.abs(paid_cum.values - tri03.reindex(index=ays, columns=devs).values)))
    log.info(f"  Paid triangle maximum absolute difference vs 03: {diff03:,.2f} {CURRENCY}")

    # ------------------------------------------------------------------ [1c]
    log.info("\n[3/5] Chain-ladder and Mack: paid vs incurred")
    f_paid, ult_paid = a03.chain_ladder(paid_cum.fillna(0.0))
    f_inc, ult_inc = a03.chain_ladder(inc_tri.fillna(0.0))
    mack_paid = a03.mack_standard_error(paid_cum)
    mack_inc = a03.mack_standard_error(inc_tri)
    rows = []
    for ay in ays:
        d_obs = vy - ay
        p_lat = float(paid_cum.loc[ay, d_obs]); c_lat = float(case_tri.loc[ay, d_obs])
        rows.append({
            "accident_year": ay, "dev_observed": d_obs,
            "paid_to_date": p_lat, "case_reserve_at_valuation": c_lat,
            "incurred_to_date": p_lat + c_lat,
            "paid_CL_ultimate": float(ult_paid[ay]),
            "paid_CL_reserve": float(ult_paid[ay]) - p_lat,
            "incurred_CL_ultimate": float(ult_inc[ay]),
            "incurred_CL_outstanding": float(ult_inc[ay]) - p_lat,
            "incurred_CL_IBNR_beyond_case": float(ult_inc[ay]) - (p_lat + c_lat),
            "paid_CL_mack_se": mack_paid["per_accident_year"][ay]["se"],
            "incurred_CL_mack_se": mack_inc["per_accident_year"][ay]["se"]})
    cl = pd.DataFrame(rows)
    tot = {"accident_year": "total", "dev_observed": "",
           **{c: float(cl[c].sum()) for c in cl.columns if c.startswith(("paid_", "case_", "incurred_"))
              and not c.endswith("mack_se")},
           "paid_CL_mack_se": mack_paid["total_se"], "incurred_CL_mack_se": mack_inc["total_se"]}
    cl = pd.concat([cl, pd.DataFrame([tot])], ignore_index=True)
    cl.to_csv(OUTPUTS_DIR / "22_cl_paid_vs_incurred.csv", index=False)
    outfiles.append(str((OUTPUTS_DIR / "22_cl_paid_vs_incurred.csv").resolve()))
    fac = pd.DataFrame({"dev_from": [k[0] for k in f_paid], "dev_to": [k[1] for k in f_paid],
                        "paid_factor": list(f_paid.values()), "incurred_factor": list(f_inc.values())})
    fac.to_csv(OUTPUTS_DIR / "22_development_factors.csv", index=False)
    outfiles.append(str((OUTPUTS_DIR / "22_development_factors.csv").resolve()))
    log.info(f"  Paid CL: ultimate {tot['paid_CL_ultimate']:,.0f}, reserve {tot['paid_CL_reserve']:,.0f} "
             f"(Mack s.e. {mack_paid['total_se']:,.0f})")
    log.info(f"  Incurred CL: ultimate {tot['incurred_CL_ultimate']:,.0f}, outstanding "
             f"{tot['incurred_CL_outstanding']:,.0f} (Mack s.e. {mack_inc['total_se']:,.0f}); "
             f"open case reserves {tot['case_reserve_at_valuation']:,.0f}; IBNR beyond case "
             f"{tot['incurred_CL_IBNR_beyond_case']:,.0f}")
    log.info("  Development factors (paid | incurred): " +
             ", ".join(f"{a}→{b}: {fp:.4f} | {fi:.4f}" for (a, b), fp, fi in
                       zip(f_paid.keys(), f_paid.values(), f_inc.values())))

    # ------------------------------------------------------------------ [2]
    log.info("\n[4/5] Case-reserve run-off: what did the open files in the December snapshots pay?")
    paid_f = paid.groupby(["policy_ref", "claim_file", "payment_year"])["paid_total_real"].sum().reset_index()
    open_at_val = dec[dec["reserve_year"] == vy][["policy_ref", "claim_file", "reserve_amount_real"]] \
        .rename(columns={"reserve_amount_real": "reserve_at_valuation"})
    runoff = []; runoff_age = []
    for t in sorted(dec["reserve_year"].unique()):
        if t >= vy:
            continue
        snap = dec[dec["reserve_year"] == t][["policy_ref", "claim_file", "reserve_amount_real", "accident_year"]]
        later = paid_f[paid_f["payment_year"] > t].groupby(["policy_ref", "claim_file"])["paid_total_real"].sum() \
            .rename("paid_after").reset_index()
        m = snap.merge(later, on=["policy_ref", "claim_file"], how="left") \
                .merge(open_at_val, on=["policy_ref", "claim_file"], how="left")
        m["paid_after"] = m["paid_after"].fillna(0.0); m["reserve_at_valuation"] = m["reserve_at_valuation"].fillna(0.0)
        R = float(m["reserve_amount_real"].sum()); P = float(m["paid_after"].sum())
        Rv = float(m["reserve_at_valuation"].sum())
        m["age_at_snapshot"] = (int(t) - m["accident_year"]).clip(lower=0)
        m["age_bucket"] = np.where(m["age_at_snapshot"] >= 2, "2+", m["age_at_snapshot"].astype(str))
        for b, g in m.groupby("age_bucket"):
            Rb = float(g["reserve_amount_real"].sum()); Pb = float(g["paid_after"].sum())
            Rvb = float(g["reserve_at_valuation"].sum())
            runoff_age.append({
                "snapshot_year_end": int(t), "age_bucket": b, "mature": bool(t <= R2_MATURE_RUNOFF_MAX_YEAR),
                "open_files": int(len(g)), "case_reserve_held": Rb,
                "paid_subsequently_to_valuation": Pb, "still_open_at_valuation_reserve": Rvb,
                "paid_to_reserve_ratio": Pb / Rb if Rb > 0 else np.nan,
                "share_files_no_subsequent_payment": float((g["paid_after"] <= 0).mean())})
        runoff.append({
            "snapshot_year_end": int(t), "runoff_window_years": int(vy - t),
            "mature": bool(t <= R2_MATURE_RUNOFF_MAX_YEAR),
            "open_files": int(len(m)), "case_reserve_held": R,
            "paid_subsequently_to_valuation": P,
            "still_open_at_valuation_files": int((m["reserve_at_valuation"] > 0).sum()),
            "still_open_at_valuation_reserve": Rv,
            "paid_to_reserve_ratio": P / R if R > 0 else np.nan,
            "paid_plus_open_to_reserve_ratio": (P + Rv) / R if R > 0 else np.nan,
            "share_files_no_subsequent_payment": float((m["paid_after"] <= 0).mean()),
            "share_files_paid_below_half_reserve": float((m["paid_after"] < 0.5 * m["reserve_amount_real"]).mean()),
            "median_paid_to_reserve_file_ratio": float((m["paid_after"] / m["reserve_amount_real"].replace(0, np.nan)).median()),
        })
        log.info(f"  {t}-12: open {len(m):,} files, case reserve {R:,.0f}, paid subsequently {P:,.0f} "
                 f"(ratio {P/R:.3f}; no payment {100*(m['paid_after']<=0).mean():.1f}%; "
                 f"still open {int((m['reserve_at_valuation']>0).sum())} files / {Rv:,.0f})")
    ro = pd.DataFrame(runoff)
    ro.to_csv(OUTPUTS_DIR / "22_case_reserve_runoff.csv", index=False)
    outfiles.append(str((OUTPUTS_DIR / "22_case_reserve_runoff.csv").resolve()))
    mat = ro[ro["mature"]]
    pooled_ratio = float(mat["paid_subsequently_to_valuation"].sum() / mat["case_reserve_held"].sum())
    pooled_ratio_incl_open = float((mat["paid_subsequently_to_valuation"].sum()
                                    + mat["still_open_at_valuation_reserve"].sum()) / mat["case_reserve_held"].sum())
    log.info(f"  Mature snapshots (≤{R2_MATURE_RUNOFF_MAX_YEAR}) pooled paid/case-reserve ratio: {pooled_ratio:.3f} "
             f"(including still-open case reserves {pooled_ratio_incl_open:.3f})")
    ra = pd.DataFrame(runoff_age)
    ra.to_csv(OUTPUTS_DIR / "22_case_reserve_runoff_by_age.csv", index=False)
    outfiles.append(str((OUTPUTS_DIR / "22_case_reserve_runoff_by_age.csv").resolve()))
    ra_m = ra[ra["mature"]].groupby("age_bucket").agg(
        open_files=("open_files", "sum"), case_reserve_held=("case_reserve_held", "sum"),
        paid_subsequently=("paid_subsequently_to_valuation", "sum"),
        still_open_reserve=("still_open_at_valuation_reserve", "sum")).reset_index()
    ra_m["paid_to_reserve_ratio"] = ra_m["paid_subsequently"] / ra_m["case_reserve_held"]
    ra_m["paid_plus_open_to_reserve_ratio"] = (ra_m["paid_subsequently"] + ra_m["still_open_reserve"]) / ra_m["case_reserve_held"]
    ra_m.to_csv(OUTPUTS_DIR / "22_case_reserve_runoff_by_age_pooled.csv", index=False)
    outfiles.append(str((OUTPUTS_DIR / "22_case_reserve_runoff_by_age_pooled.csv").resolve()))
    age_ratio = dict(zip(ra_m["age_bucket"], ra_m["paid_to_reserve_ratio"]))
    for _, r in ra_m.iterrows():
        log.info(f"    age {r['age_bucket']:>2}: {int(r['open_files']):,} files, case reserve {r['case_reserve_held']:,.0f}, "
                 f"paid/case-reserve {r['paid_to_reserve_ratio']:.3f} (including open {r['paid_plus_open_to_reserve_ratio']:.3f})")

    # ------------------------------------------------------------------ [3]
    log.info("\n[5/5] Files open at the valuation date (by accident year)")
    ov = dec[dec["reserve_year"] == vy]
    ov_paid = paid_f.groupby(["policy_ref", "claim_file"])["paid_total_real"].sum().rename("paid_to_date").reset_index()
    ov = ov.merge(ov_paid, on=["policy_ref", "claim_file"], how="left"); ov["paid_to_date"] = ov["paid_to_date"].fillna(0.0)
    by_ay = ov.groupby("accident_year").agg(files=("claim_file", "size"),
                                            case_reserve=("reserve_amount_real", "sum"),
                                            paid_to_date_on_open_files=("paid_to_date", "sum"),
                                            files_with_no_payment=("paid_to_date", lambda s: int((s <= 0).sum()))).reset_index()
    by_ay["age_years_at_valuation"] = vy - by_ay["accident_year"]
    by_ay["expected_payment_at_mature_runoff_ratio"] = by_ay["case_reserve"] * pooled_ratio
    by_ay["age_bucket"] = np.where(by_ay["age_years_at_valuation"] >= 2, "2+", by_ay["age_years_at_valuation"].astype(str))
    by_ay["age_matched_runoff_ratio"] = by_ay["age_bucket"].map(age_ratio)
    by_ay["expected_payment_age_matched"] = by_ay["case_reserve"] * by_ay["age_matched_runoff_ratio"]
    by_ay.to_csv(OUTPUTS_DIR / "22_open_files_at_valuation.csv", index=False)
    outfiles.append(str((OUTPUTS_DIR / "22_open_files_at_valuation.csv").resolve()))
    tot_case = float(by_ay["case_reserve"].sum())
    exp_age = float(by_ay["expected_payment_age_matched"].sum())
    log.info(f"  Open {int(by_ay['files'].sum())} files, case reserve {tot_case:,.0f}; files with no payment "
             f"{int(by_ay['files_with_no_payment'].sum())}; expected payment at the mature run-off ratio "
             f"{tot_case*pooled_ratio:,.0f}; AGE-MATCHED expected payment {exp_age:,.0f} "
             f"(paid-CL reserve {tot['paid_CL_reserve']:,.0f}; incurred-CL outstanding {tot['incurred_CL_outstanding']:,.0f})")
    for _, r in by_ay.iterrows():
        log.info(f"    AY {int(r['accident_year'])}: {int(r['files'])} files, case reserve {r['case_reserve']:,.0f}, "
                 f"age {int(r['age_years_at_valuation'])} years")

    manifest = {
        "script": SCRIPT_NAME, "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "purpose": "R2 Reviewer 1 point 4 (reserving): incurred chain-ladder and case-reserve run-off",
        "valuation": f"{vy}-{vm:02d}", "price_basis": "real, Dec-2018 (as in the paper)",
        "accident_years": ays, "development_years": devs,
        "incurred_definition": ("cumulative real paid through the calendar year-end + real open case "
                                "reserves at that December snapshot (closed claims leave the book)"),
        "accident_year_source_for_reserves": "raw Muallak file Hasar_Tarihi (Excel serial) → year; min per (policy, file)",
        "reserve_rows_missing_accident_year": n_missing,
        "accident_year_agreement_with_paid_panel": agree,
        "december_snapshot_rows_before_dedup": n_before, "december_snapshot_files": int(len(dec)),
        "paid_triangle_max_abs_diff_vs_03": diff03,
        "paid_CL": {"ultimate": tot["paid_CL_ultimate"], "reserve": tot["paid_CL_reserve"],
                    "mack_total_se": mack_paid["total_se"], "factors": list(f_paid.values())},
        "incurred_CL": {"ultimate": tot["incurred_CL_ultimate"], "outstanding": tot["incurred_CL_outstanding"],
                        "case_reserve_at_valuation": tot["case_reserve_at_valuation"],
                        "IBNR_beyond_case": tot["incurred_CL_IBNR_beyond_case"],
                        "mack_total_se": mack_inc["total_se"], "factors": list(f_inc.values())},
        "case_reserve_runoff": {"mature_snapshot_max_year": R2_MATURE_RUNOFF_MAX_YEAR,
                                "pooled_paid_to_reserve_ratio_mature": pooled_ratio,
                                "pooled_ratio_including_still_open": pooled_ratio_incl_open,
                                "per_snapshot": ro.to_dict(orient="records")},
        "open_files_at_valuation": {"files": int(by_ay["files"].sum()), "case_reserve": tot_case,
                                    "expected_payment_at_mature_runoff_ratio": tot_case * pooled_ratio,
                                    "expected_payment_age_matched": exp_age,
                                    "age_matched_ratios_mature": age_ratio,
                                    "by_accident_year": by_ay.to_dict(orient="records")},
        "output_files": outfiles, "environment": env_info(),
    }
    with open(OUTPUTS_DIR / "22_manifest.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(manifest), f, ensure_ascii=False, indent=2)
    log.info("\n" + "=" * 78); log.info("SCRIPT 22 COMPLETED"); log.info("=" * 78)


if __name__ == "__main__":
    main()
