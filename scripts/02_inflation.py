"""
================================================================================
SCRIPT 02 — Inflation Adjustment (Real Deflation)
================================================================================
PURPOSE
    Converts nominal paid claims and case reserves into REAL amounts by deflating
    them to a fixed base month (2018-12) with the TurkStat/CBRT CPI series. In a
    high-inflation environment (TR 2014-2020) comparing nominal amounts across
    years is misleading; all dependence and capital analyses are carried out on
    real amounts.

DESIGN DECISIONS
    • ONE-WAY deflation: each cash flow is brought to the base month exactly once,
      using the CPI of its own month of occurrence. Double deflation (e.g. first
      annual then monthly) is not performed.
    • Flexible CPI reading: cpi_monthly.xlsx (the TUFE_Veri sheet is found
      automatically) or .csv. year_month is normalised to 'YYYY-MM' even if it
      arrives as a Timestamp/number.
    • STRICT mode: an error is raised if a CPI month is missing; a hard-coded
      fallback is NOT used (for academic reproduction the CPI file is the only
      verifiable source).
    • Sensitivity: a ±5% shock is applied to the CPI slope and the sensitivity of
      real incurred is reported (so that the effect of the inflation assumption on
      the results is transparent).

INPUT : data/proc/{paid_long,reserve_panel}.parquet ; data/raw/cpi_monthly.*
OUTPUT: data/proc/{paid_long_real,reserve_panel_real}.{parquet,pkl}
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
    HPO_VALUATION_YEAR, HPO_VALUATION_MONTH,
    DATA_RAW_DIR, DATA_PROC_DIR, OUTPUTS_DIR, LOGS_DIR,
    CPI_CANDIDATES, CPI_SHEET_CANDIDATES, INFLATION_BASE_YEAR_MONTH,
    INFLATION_STRICT_MODE, CURRENCY, TERRITORY, LINE_OF_BUSINESS,
    PICKLE_PROTOCOL, CPI_SLOPE_SHOCKS, DEFAULT_PAID_MONTH, DEFAULT_RESERVE_MONTH,
)

SCRIPT_NAME = "02_inflation"


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
    pk = DATA_PROC_DIR / f"{stem}.pkl"
    if pq.exists():
        return pd.read_parquet(pq)
    return pd.read_pickle(pk)


# =========================================================
# Flexible CPI reading (xlsx first, correct sheet found automatically)
# =========================================================
def load_cpi() -> pd.DataFrame:
    cpi_path = None
    for cand in CPI_CANDIDATES:
        p = DATA_RAW_DIR / cand
        if p.exists():
            cpi_path = p
            break
    if cpi_path is None:
        msg = (f"CPI file not found ({CPI_CANDIDATES}). Download it from the CBRT "
               f"EVDS service with cpi_fetch.py or place cpi_monthly.xlsx in data/raw.")
        if INFLATION_STRICT_MODE:
            raise FileNotFoundError("[STRICT] " + msg)
        raise FileNotFoundError(msg)

    log.info(f"  CPI source: {cpi_path.name}")
    if cpi_path.suffix.lower() in (".xlsx", ".xls"):
        xl = pd.ExcelFile(cpi_path)
        cpi = None
        for sh in CPI_SHEET_CANDIDATES + xl.sheet_names:
            if sh in xl.sheet_names:
                cand = pd.read_excel(cpi_path, sheet_name=sh)
                if {"year_month", "cpi_index"}.issubset(cand.columns):
                    cpi = cand
                    log.info(f"  CPI sheet: '{sh}'")
                    break
        if cpi is None:
            raise ValueError(f"No sheet with year_month+cpi_index in the Excel file: "
                             f"{xl.sheet_names}")
    else:
        cpi = pd.read_csv(cpi_path)

    if not {"year_month", "cpi_index"}.issubset(cpi.columns):
        raise ValueError(f"Unexpected CPI schema: {list(cpi.columns)}")

    cpi = cpi[["year_month", "cpi_index"]].dropna(subset=["year_month"]).copy()
    if not pd.api.types.is_string_dtype(cpi["year_month"]):
        cpi["year_month"] = pd.to_datetime(cpi["year_month"],
                                           errors="coerce").dt.strftime("%Y-%m")
    cpi["cpi_index"] = pd.to_numeric(cpi["cpi_index"], errors="coerce")
    dt = pd.to_datetime(cpi["year_month"], format="%Y-%m")
    cpi["ym_key"] = (dt.dt.year * 100 + dt.dt.month).astype(int)
    cpi = cpi.sort_values("ym_key").reset_index(drop=True)

    n_missing = int(cpi["cpi_index"].isna().sum())
    if n_missing and INFLATION_STRICT_MODE:
        raise ValueError(f"[STRICT] CPI'da {n_missing} eksik ay")
    log.info(f"  ✓ CPI: {len(cpi)} months "
             f"({cpi['year_month'].iloc[0]}→{cpi['year_month'].iloc[-1]}), "
             f"base {INFLATION_BASE_YEAR_MONTH}="
             f"{cpi.loc[cpi.ym_key==INFLATION_BASE_YEAR_MONTH,'cpi_index'].iloc[0]:.2f}")
    return cpi


def deflation_factor(ym_key: pd.Series, cpi: pd.DataFrame,
                     base: int) -> pd.Series:
    """Deflation factor to the base month for each month = CPI_base / CPI_month."""
    base_cpi = float(cpi.loc[cpi["ym_key"] == base, "cpi_index"].iloc[0])
    cmap = dict(zip(cpi["ym_key"], cpi["cpi_index"]))
    return ym_key.map(lambda k: base_cpi / cmap.get(int(k), np.nan)
                      if pd.notna(k) else np.nan)


def main() -> None:
    global log
    log = setup_logger()
    log.info("=" * 78)
    log.info(f"SCRIPT {SCRIPT_NAME} — Real Deflation")
    log.info(f"Base month: {INFLATION_BASE_YEAR_MONTH} | STRICT={INFLATION_STRICT_MODE}")
    log.info("=" * 78)

    outfiles: list = []
    log.info("\n[1/4] Inputs + CPI")
    paid = load_proc("paid_long")
    reserve = load_proc("reserve_panel")
    cpi = load_cpi()

    log.info("\n[2/4] Paid real deflation (one-way)")
    paid["ym_key"] = (paid["payment_year"] * 100
                      + paid["payment_month"].fillna(DEFAULT_PAID_MONTH)).astype(int)
    paid["deflation_factor"] = deflation_factor(
        paid["ym_key"], cpi, INFLATION_BASE_YEAR_MONTH)
    if INFLATION_STRICT_MODE and paid["deflation_factor"].isna().any():
        raise ValueError("[STRICT] some payment months are missing from the CPI")
    paid["paid_total_real"] = paid["paid_total"] * paid["deflation_factor"]
    log.info(f"  Real paid total = {paid['paid_total_real'].sum():,.0f} "
             f"{CURRENCY} (base {INFLATION_BASE_YEAR_MONTH})")

    log.info("\n[3/4] Case-reserve real deflation")
    reserve["ym_key"] = (reserve["reserve_year"] * 100
                         + reserve["reserve_month"].fillna(DEFAULT_RESERVE_MONTH)).astype(int)
    reserve["deflation_factor"] = deflation_factor(
        reserve["ym_key"], cpi, INFLATION_BASE_YEAR_MONTH)
    reserve["reserve_amount_real"] = (reserve["reserve_amount"]
                                      * reserve["deflation_factor"])
    log.info(f"  Real case-reserve total = "
             f"{reserve['reserve_amount_real'].sum():,.0f} {CURRENCY} "
             f"(mean factor {reserve['deflation_factor'].mean():.3f})")

    log.info(f"\n[4/4] Sensitivity: {CPI_SLOPE_SHOCKS} shock to the CPI slope")
    base_cpi = float(cpi.loc[cpi["ym_key"] == INFLATION_BASE_YEAR_MONTH,
                             "cpi_index"].iloc[0])
    # R1 final: incurred = paid + open case reserves AT THE VALUATION DATE (not the
    # sum of all quarter-end records; that sum double-counts closed claims).
    val_key = HPO_VALUATION_YEAR * 100 + HPO_VALUATION_MONTH
    reserve_open = reserve[reserve["ym_key"] == val_key]
    real_incurred = (paid["paid_total_real"].sum()
                     + reserve_open["reserve_amount_real"].sum())
    sens = {}
    for shock in CPI_SLOPE_SHOCKS:
        cmap = dict(zip(cpi["ym_key"], cpi["cpi_index"]))
        # slope shock: scale the CPI in proportion to the distance from the base month
        def shocked_factor(k):
            c = cmap.get(int(k), np.nan)
            yrs = (int(k) // 100) - (INFLATION_BASE_YEAR_MONTH // 100)
            return base_cpi / (c * (1 + shock) ** yrs) if pd.notna(c) else np.nan
        p_real = (paid["paid_total"] * paid["ym_key"].map(shocked_factor)).sum()
        r_real = (reserve_open["reserve_amount"]
                  * reserve_open["ym_key"].map(shocked_factor)).sum()
        delta = (p_real + r_real) / real_incurred - 1
        sens[f"shock_{shock:+.0%}"] = float(delta)
        log.info(f"  CPI slope {shock:+.0%} → real incurred {delta:+.2%}")

    for df, stem in [(paid, "paid_long_real"), (reserve, "reserve_panel_real")]:
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
        "base_year_month": INFLATION_BASE_YEAR_MONTH,
        "strict_mode": INFLATION_STRICT_MODE,
        "real_paid_total": float(paid["paid_total_real"].sum()),
        "real_reserve_total": float(reserve["reserve_amount_real"].sum()),
        "cpi_slope_sensitivity": sens,
        "design_choices": {"deflation": "one-way to base month",
                           "double_deflation": False},
        "output_files": outfiles,
        "environment": {"python": platform.python_version(),
                        "pandas": pd.__version__, "numpy": np.__version__},
    }
    with open(OUTPUTS_DIR / "02_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log.info("\n" + "=" * 78)
    log.info("SCRIPT 02 COMPLETED")
    log.info("=" * 78)


if __name__ == "__main__":
    main()
