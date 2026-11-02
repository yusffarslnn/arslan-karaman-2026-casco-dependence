"""
================================================================================
SCRIPT 00 — Data Ingest (Excel → Parquet, Memory-Safe)
================================================================================
PURPOSE
    Converts the raw Excel files (production 818K rows / 264 MB, paid, case
    reserves) into Parquet format for analysis. Opening the large Excel file
    with pandas.read_excel is impractical (OOM) because openpyxl loads the whole
    workbook into RAM. Solution: read it row by row with read-only streaming
    into a CSV, then write Parquet from there. This step runs once; all
    subsequent scripts read fast, memory-friendly Parquet.

INPUT : data/raw/*.xlsx  (config.RAW_FILES)
OUTPUT: data/raw/*.parquet
================================================================================
"""
from __future__ import annotations
import sys
import csv
import time
import logging
from pathlib import Path

# --- Spyder-compatible path setup: look for config.py both in the same folder
# and in the parent folder (flat scripts/ layout AND src/ layout supported) ---
try:
    _HERE = Path(__file__).resolve().parent
except NameError:
    _HERE = Path.cwd()  # working directory if __file__ is undefined in Spyder
for _cand in (_HERE, _HERE.parent):
    if (_cand / "config.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

import openpyxl
import pandas as pd

from config import DATA_RAW_DIR, RAW_FILES, LOGS_DIR, INGEST_LOG_EVERY

SCRIPT_NAME = "00_data_ingest"


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


def stream_excel_to_csv(xlsx_path: Path, sheet: str, csv_path: Path) -> int:
    """Write the Excel sheet to CSV via read-only streaming; return the row count."""
    t0 = time.time()
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb[sheet]
    n = 0
    header = None
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for row in ws.iter_rows(values_only=True):
            if header is None:
                header = row
                w.writerow(row)
                continue
            w.writerow(row)
            n += 1
            if n % INGEST_LOG_EVERY == 0:
                log.info(f"    {n:,} rows... ({time.time()-t0:.0f}s)")
    wb.close()
    log.info(f"    {sheet}: {n:,} rows, {time.time()-t0:.0f}s")
    return n


def main() -> None:
    global log
    log = setup_logger()
    log.info("=" * 78)
    log.info(f"SCRIPT {SCRIPT_NAME} — Excel → Parquet (memory-safe)")
    log.info("=" * 78)

    for key, cfg in RAW_FILES.items():
        xlsx = DATA_RAW_DIR / cfg["excel"]
        pq = DATA_RAW_DIR / cfg["parquet"]
        if pq.exists():
            log.info(f"[{key}] {pq.name} already exists, skipping.")
            continue
        if not xlsx.exists():
            log.warning(f"[{key}] {xlsx.name} not found, skipping.")
            continue
        log.info(f"[{key}] {xlsx.name} [{cfg['sheet']}] processing...")
        csv_tmp = DATA_RAW_DIR / f"{key}.csv"
        stream_excel_to_csv(xlsx, cfg["sheet"], csv_tmp)
        # CSV → Parquet (pandas reads it quickly)
        df = pd.read_csv(csv_tmp, low_memory=False)
        df.to_parquet(pq)
        csv_tmp.unlink()  # delete the temporary CSV
        log.info(f"[{key}] ✓ {pq.name} ({len(df):,} rows × {df.shape[1]} columns)")

    log.info("=" * 78)
    log.info("SCRIPT 00 COMPLETED — Parquet bridge ready")
    log.info("=" * 78)


if __name__ == "__main__":
    main()
