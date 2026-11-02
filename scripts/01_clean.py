"""
================================================================================
SCRIPT 01 — Data Cleaning and Panel Construction
================================================================================
PURPOSE
    Cleans the three raw sources (production/policies, paid claims, case
    reserves) and converts them into consistent long-format panels for the
    subsequent analyses:
      • policy master table (exposure, covariates, claim status)
      • paid-claims long panel (accident year × payment year; for the development triangle)
      • case-reserve quarterly panel (valuation-date snapshots)

DESIGN DECISIONS
    • Negative paid amounts are PRESERVED: they are salvage/subrogation recoveries;
      clipping them would distort the true cash flow and the ultimate cost.
    • Development year = payment year − accident year (basis of the Chain-Ladder triangle).
    • Quarter-end case-reserve snapshots are preserved so that the reserve of each
      file can be tracked by valuation date (for the HPO and RBNS calculations).
    • All column access goes through config.COLMAP; raw column names are not hard-coded.

INPUT : data/raw/{uretim,odenen,muallak}.parquet
OUTPUT: data/proc/{policies,paid_long,reserve_panel}.{parquet,pkl}
================================================================================
"""
from __future__ import annotations
import sys
import json
import logging
import platform
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional

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
    DATA_RAW_DIR, DATA_PROC_DIR, OUTPUTS_DIR, LOGS_DIR, RAW_FILES, COLMAP,
    CURRENCY, TERRITORY, LINE_OF_BUSINESS, CLIP_NEGATIVE_PAID, PICKLE_PROTOCOL,
    STUDY_YEAR_MIN, STUDY_YEAR_MAX, CLAIM_STATUS_CLAIMED,
)

SCRIPT_NAME = "01_clean"


def setup_logger() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(SCRIPT_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOGS_DIR / f"{SCRIPT_NAME}.log", mode="w",
                             encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger


log = logging.getLogger(SCRIPT_NAME)


def get_git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT_FALLBACK(),
            stderr=subprocess.DEVNULL, timeout=5).decode().strip()
    except Exception:
        return None


def PROJECT_ROOT_FALLBACK():
    return Path(__file__).resolve().parent.parent


def read_raw(key: str) -> pd.DataFrame:
    """Parquet-first raw read (output of 00_data_ingest)."""
    cfg = RAW_FILES[key]
    pq = DATA_RAW_DIR / cfg["parquet"]
    if pq.exists():
        return pd.read_parquet(pq)
    xlsx = DATA_RAW_DIR / cfg["excel"]
    if xlsx.exists():
        log.warning(f"  {pq.name} not found; reading Excel (slow). "
                    f"Run 00_data_ingest first.")
        return pd.read_excel(xlsx, sheet_name=cfg["sheet"])
    raise FileNotFoundError(f"{key}: ne parquet ne excel bulundu")


def to_year(s: pd.Series) -> pd.Series:
    """Extract the year. The input may be an Excel serial date (int, ~40000-44000)
    or a regular date; both are handled correctly."""
    s_num = pd.to_numeric(s, errors="coerce")
    # Excel serial-date range (1990-2030 ≈ 32874-47482)
    if s_num.notna().mean() > 0.8 and s_num.dropna().between(20000, 60000).mean() > 0.8:
        # Excel origin 1899-12-30
        dt = pd.to_datetime(s_num, unit="D", origin="1899-12-30", errors="coerce")
        return dt.dt.year
    return pd.to_datetime(s, errors="coerce", dayfirst=True).dt.year


def yearmonth_to_year(s: pd.Series) -> pd.Series:
    return (pd.to_numeric(s, errors="coerce") // 100).astype("Int64")


def yearmonth_to_month(s: pd.Series) -> pd.Series:
    return (pd.to_numeric(s, errors="coerce") % 100).astype("Int64")


# =========================================================
# 1) PRODUCTION (policy master table)
# =========================================================
def clean_policies() -> pd.DataFrame:
    df = read_raw("uretim")
    log.info(f"  raw production: {df.shape}")
    out = pd.DataFrame()
    out["policy_ref"] = df[COLMAP["policy_ref"]].astype(str)
    out["policy_year"] = to_year(df[COLMAP["policy_start"]])
    out["exposure"] = pd.to_numeric(df[COLMAP["exposure"]],
                                    errors="coerce").fillna(0.0)
    out["claim_status"] = df[COLMAP["claim_status"]].astype(str)
    for c in ["premium", "premium_total", "premium_kasko"]:
        if COLMAP[c] in df.columns:
            out[COLMAP[c]] = pd.to_numeric(df[COLMAP[c]], errors="coerce")
    # covariates
    for c in ["vehicle_age", "insured_age", "horsepower", "ccm",
              "vehicle_value", "gender", "body_code"]:
        if COLMAP[c] in df.columns:
            out[COLMAP[c]] = df[COLMAP[c]]
    out = out.dropna(subset=["policy_year"])
    out = out[out["policy_year"].between(STUDY_YEAR_MIN, STUDY_YEAR_MAX)]
    log.info(f"  clean policies: {len(out):,} "
             f"(with claims: {(out['claim_status']==CLAIM_STATUS_CLAIMED).sum():,})")
    return out.reset_index(drop=True)


# =========================================================
# 2) PAID (long panel + development year)
# =========================================================
def clean_paid() -> pd.DataFrame:
    df = read_raw("odenen")
    log.info(f"  raw paid: {df.shape}")
    out = pd.DataFrame()
    out["policy_ref"] = df[COLMAP["policy_ref"]].astype(str)
    out["claim_file"] = df[COLMAP["claim_file"]].astype(str)
    out["accident_year"] = to_year(df[COLMAP["claim_date"]])
    out["payment_year"] = yearmonth_to_year(df[COLMAP["paid_yearmonth"]])
    out["payment_month"] = yearmonth_to_month(df[COLMAP["paid_yearmonth"]])
    out["paid_total"] = pd.to_numeric(df[COLMAP["paid_amount"]],
                                      errors="coerce").fillna(0.0)
    if not CLIP_NEGATIVE_PAID:
        n_neg = int((out["paid_total"] < 0).sum())
        log.info(f"  negative paid (salvage/subrogation) preserved: {n_neg:,} rows")
    else:
        out["paid_total"] = out["paid_total"].clip(lower=0)
    out = out.dropna(subset=["accident_year", "payment_year"])
    # development year (Chain-Ladder triangle)
    out["dev_year"] = (out["payment_year"] - out["accident_year"]).astype(int)
    out = out[out["dev_year"] >= 0]
    log.info(f"  clean paid: {len(out):,} "
             f"(total {out['paid_total'].sum():,.0f} {CURRENCY})")
    return out.reset_index(drop=True)


# =========================================================
# 3) CASE RESERVES (quarterly panel snapshot)
# =========================================================
def clean_reserve() -> pd.DataFrame:
    df = read_raw("muallak")
    log.info(f"  raw case reserves: {df.shape}")
    out = pd.DataFrame()
    out["policy_ref"] = df[COLMAP["policy_ref"]].astype(str)
    out["claim_file"] = df[COLMAP["claim_file"]].astype(str)
    out["reserve_year"] = yearmonth_to_year(df[COLMAP["reserve_yearmonth"]])
    out["reserve_month"] = yearmonth_to_month(df[COLMAP["reserve_yearmonth"]])
    out["reserve_amount"] = pd.to_numeric(df[COLMAP["reserve_amount"]],
                                          errors="coerce").fillna(0.0)
    out = out.dropna(subset=["reserve_year"])
    log.info(f"  clean case reserves: {len(out):,} "
             f"(total {out['reserve_amount'].sum():,.0f} {CURRENCY})")
    return out.reset_index(drop=True)


def save(df: pd.DataFrame, stem: str, outfiles: list) -> None:
    pkl = DATA_PROC_DIR / f"{stem}.pkl"
    df.to_pickle(pkl, protocol=PICKLE_PROTOCOL)
    outfiles.append(str(pkl.resolve()))
    try:
        df.to_parquet(DATA_PROC_DIR / f"{stem}.parquet", index=False)
        outfiles.append(str((DATA_PROC_DIR / f"{stem}.parquet").resolve()))
    except Exception:
        pass


def main() -> None:
    global log
    log = setup_logger()
    log.info("=" * 78)
    log.info(f"SCRIPT {SCRIPT_NAME} — Data Cleaning")
    log.info(f"Start: {datetime.now().isoformat(timespec='seconds')}")
    log.info(f"Domain: {CURRENCY}/{TERRITORY}/{LINE_OF_BUSINESS}")
    log.info("=" * 78)

    outfiles: list = []
    log.info("\n[1/3] Policy master table")
    policies = clean_policies(); save(policies, "policies", outfiles)
    log.info("\n[2/3] Paid-claims long panel")
    paid = clean_paid(); save(paid, "paid_long", outfiles)
    log.info("\n[3/3] Case-reserve quarterly panel")
    reserve = clean_reserve(); save(reserve, "reserve_panel", outfiles)

    manifest = {
        "script": SCRIPT_NAME,
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_sha": get_git_sha(),
        "domain": {"currency": CURRENCY, "territory": TERRITORY,
                   "line_of_business": LINE_OF_BUSINESS},
        "counts": {"policies": len(policies), "paid_rows": len(paid),
                   "reserve_rows": len(reserve),
                   "claims_policies": int((policies["claim_status"] == CLAIM_STATUS_CLAIMED).sum())},
        "design_choices": {
            "study_years": [STUDY_YEAR_MIN, STUDY_YEAR_MAX],
            "clip_negative_paid": CLIP_NEGATIVE_PAID,
            "dev_year": "payment_year - accident_year",
            "reserve_panel": "quarter-end snapshots preserved"},
        "output_files": outfiles,
        "environment": {"python": platform.python_version(),
                        "pandas": pd.__version__, "numpy": np.__version__},
    }
    with open(OUTPUTS_DIR / "01_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log.info("\n" + "=" * 78)
    log.info("SCRIPT 01 COMPLETED")
    log.info("=" * 78)


if __name__ == "__main__":
    main()
