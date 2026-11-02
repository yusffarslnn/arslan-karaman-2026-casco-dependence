"""
================================================================================
run_pipeline.py — Pipeline Orchestrator (End-to-End Reproduction)
================================================================================
PhD Thesis: Optimal Premium Pricing and Claims Reserving Strategies for
Motor Casco Insurance — YTÜ Department of Statistics (Arslan & Karaman)
================================================================================
PURPOSE
    Runs all pipeline modules (00–21, incl. 14b; the R1 revision module 21
    runs after 19 and before 20_synthesis) in the correct order with a single
    command. Tracks the status/duration of each module, stops on error and
    produces a consolidated run report at the end. For the jury/reproduction:
    "python run_pipeline.py" regenerates the whole thesis from scratch.

USAGE
    python run_pipeline.py            # whole pipeline
    python run_pipeline.py --from 10  # from 10 onwards
    python run_pipeline.py --only 14b # single module
    python run_pipeline.py --skip-ingest  # skip 00 (if Parquet already exists)

OUTPUT
    outputs/pipeline_run_report.json
    logs/pipeline_run.log
================================================================================
"""
from __future__ import annotations
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

def _resolve_src(root: Path) -> Path:
    """Locate the script directory. Order of precedence:
    1) is run_pipeline.py in the SAME folder as the scripts? (next to config.py + 01_clean.py)
    2) root/scripts/  sub-folder
    3) root/src/      sub-folder
    4) last resort: root
    """
    if (root / "config.py").exists() and (root / "01_clean.py").exists():
        return root
    for sub in ("scripts", "src"):
        if (root / sub / "config.py").exists():
            return root / sub
    return root


_ROOT = Path(__file__).resolve().parent
SRC = _resolve_src(_ROOT)
OUTPUTS = SRC / "outputs"
LOGS = SRC / "logs"
OUTPUTS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)


# Packages required by the whole pipeline: {import_name: pip_name}
REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "openpyxl": "openpyxl",        # 00_data_ingest (Excel reading)
    "pyarrow": "pyarrow",         # Parquet
    "statsmodels": "statsmodels",  # 14b (Tweedie/NegBin GLM)
    "lightgbm": "lightgbm",        # 14b
    "xgboost": "xgboost",          # 14b
}


def ensure_dependencies(packages: dict = REQUIRED_PACKAGES) -> list:
    """Installs missing packages with pip into the RUNNING interpreter.

    sys.executable is used, so that in environments such as Spyder/conda the
    packages are installed in the right place. Does nothing if everything is
    already installed. Returns the list of packages that could not be
    installed (empty = all ready)."""
    import importlib
    missing = []
    for imp_name, pip_name in packages.items():
        try:
            importlib.import_module(imp_name)
        except ImportError:
            missing.append(pip_name)
    if not missing:
        return []
    print(f"[dependency] Missing packages: {missing}")
    print(f"[dependency] Target interpreter: {sys.executable}")
    print("[dependency] installing with pip (may take a few minutes)...")
    failed = []
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", *missing])
        importlib.invalidate_caches()
        # verification
        for imp_name, pip_name in packages.items():
            try:
                importlib.import_module(imp_name)
            except ImportError:
                failed.append(pip_name)
        if failed:
            print(f"[dependency] Could not be installed: {failed}")
            if "lightgbm" in failed:
                print("[dependency] LightGBM may require 'brew install libomp' "
                      "on Mac.")
        else:
            print("[dependency] All packages ready.")
    except subprocess.CalledProcessError as e:
        print(f"[dependency] pip installation failed: {e}")
        failed = missing
    return failed


# Ordered module list (with phase labels)
PHASES = [
    ("PHASE 0 — Infrastructure", [
        ("00_data_ingest", "Excel → Parquet (memory-safe)"),
    ]),
    ("PHASE 1 — Data preparation", [
        ("01_clean", "Cleaning, panel construction"),
        ("02_inflation", "Real-terms deflation (CPI)"),
        ("03_analytical", "Chain-ladder, ultimate definitions"),
    ]),
    ("PHASE 2 — Dependence diagnostics (methodological core)", [
        ("04_eda", "Descriptives, Hill, HPO, pre-tests"),
        ("05_bias_theory", "Artifact characterisation (controlled experiment)"),
        ("06_dependence", "Dependence axes + cluster bootstrap"),
        ("07_conditional", "Conditional PITs + cross-fitted Vuong"),
        ("08_power", "Power analysis (MDE)"),
    ]),
    ("PHASE 3 — Modelling", [
        ("10_marginals", "Severity + frequency marginals"),
        ("09_positive_control", "Positive control (method validity; uses the marginals of 10)"),
        ("11_copulas", "Copula fits + family selection"),
        ("12_gof", "Goodness of fit + independence effect size"),
        ("13_capital", "Compound MC, paired capital proxy (leakage-free)"),
    ]),
    ("PHASE 4 — Pricing baselines & contribution", [
        ("14_glm", "GLM baseline (pure-numpy, aligned)"),
        ("14b_pricing_models", "Tweedie/NegBin/LightGBM/XGBoost"),
        ("15_gbm", "GBM baseline (pure-numpy)"),
        ("16_mlp", "MLP baseline (pure-numpy)"),
        ("17_augmentation", "Copula augmentation ablation"),
    ]),
    ("PHASE 5 — Decision layer (RORAC-target premium & risk-based reserve)", [
        ("18_optimal_pricing", "RORAC-target technical premium + premium principles"),
        ("22_revision_R2", "R2: incurred triangle + case-reserve run-off for 19"),
        ("19_optimal_reserving", "Mack-lognormal predictive distribution + CoC margin + newsvendor reserve"),
    ]),
    ("PHASE 5b — R1 revision additions (Table 1, Table A1, TOST, power, cleaning)", [
        ("21_revision_R1", "R1 revision numbers: descriptives, VIF, TOST, MDE"),
    ]),
    ("PHASE 5c — R2 revision additions (Reviewer 1, round 2)", [
        ("24_b2_decomposition", "R2: artifact/residual decomposition of the naive measures"),
        ("23_independent_verification", "R2: independent re-derivation of the headline results"),
    ]),
    ("PHASE 6 — Synthesis", [
        ("20_synthesis", "Consolidation, audit gates, SUMMARY"),
    ]),
]


def all_modules():
    out = []
    for _, mods in PHASES:
        out.extend([m for m, _ in mods])
    return out


def run_module(name: str, timeout: int = 900) -> dict:
    """Run one module; return status + duration + the last log lines."""
    script = SRC / f"{name}.py"
    if not script.exists():
        return {"module": name, "status": "MISSING", "seconds": 0,
                "tail": f"{script} not found"}
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(SRC), capture_output=True, text=True, timeout=timeout)
        dt = time.time() - t0
        out_lines = (proc.stdout + proc.stderr).strip().splitlines()
        tail = " | ".join(out_lines[-3:]) if out_lines else ""
        ok = (proc.returncode == 0 and ("COMPLETED" in proc.stdout or "TAMAMLANDI" in proc.stdout))  # EN edition: both completion tags
        return {"module": name, "status": "OK" if ok else "FAIL",
                "seconds": round(dt, 1), "returncode": proc.returncode,
                "tail": tail[-300:]}
    except subprocess.TimeoutExpired:
        return {"module": name, "status": "TIMEOUT",
                "seconds": round(time.time() - t0, 1),
                "tail": f"{timeout}s exceeded"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=None,
                    help="run from this module onwards (e.g. 10)")
    ap.add_argument("--only", default=None, help="only this module")
    ap.add_argument("--skip-ingest", action="store_true",
                    help="skip 00_data_ingest")
    ap.add_argument("--skip-deps", action="store_true",
                    help="skip the package check/installation")
    args = ap.parse_args()

    mods = all_modules()
    if args.only:
        mods = [m for m in mods if m.startswith(args.only)]
    elif args.start:
        idx = next((i for i, m in enumerate(mods)
                    if m.startswith(args.start)), 0)
        mods = mods[idx:]
    if args.skip_ingest:
        mods = [m for m in mods if not m.startswith("00")]

    log_lines = []
    def emit(s):
        print(s); log_lines.append(s)

    emit("=" * 78)
    emit("PIPELINE ORCHESTRATOR — doctoral-thesis reproduction")
    emit(f"Start: {datetime.now().isoformat(timespec='seconds')}")
    emit(f"{len(mods)} modules to run")
    emit("=" * 78)

    # Dependency check (once, before any module runs)
    if not args.skip_deps:
        emit("\n[dependencies] Checking packages...")
        failed = ensure_dependencies()
        if failed:
            emit(f"[dependencies] WARNING: {failed} could not be installed. 14b may be skipped; "
                 f"the other modules run with numpy/pandas/scipy.")
        else:
            emit("[dependencies] All packages available.")
    t_all = time.time()
    results = []
    # proceed with phase headers
    phase_map = {m: ph for ph, lst in PHASES for m, _ in lst}
    desc_map = {m: d for _, lst in PHASES for m, d in lst}
    current_phase = None
    for m in mods:
        ph = phase_map.get(m, "")
        if ph != current_phase:
            current_phase = ph
            emit(f"\n### {ph}")
        emit(f"  ▶ {m:22s} {desc_map.get(m,''):42s} ...")
        r = run_module(m)
        results.append(r)
        sym = {"OK": "✓", "FAIL": "✗", "TIMEOUT": "⏱", "MISSING": "?"}.get(r["status"], "?")
        emit(f"    {sym} {r['status']:8s} ({r['seconds']:6.1f}s)  {r['tail'][:90]}")
        if r["status"] != "OK":
            emit(f"\n  ⚠ {m} failed — pipeline stopped.")
            break

    total = time.time() - t_all
    n_ok = sum(1 for r in results if r["status"] == "OK")
    emit("\n" + "=" * 78)
    emit(f"SUMMARY: {n_ok}/{len(results)} modules succeeded, total {total:.0f}s")
    emit("=" * 78)
    for r in results:
        emit(f"  {r['status']:8s} {r['module']:22s} {r['seconds']:7.1f}s")

    report = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "total_seconds": round(total, 1),
        "n_modules": len(results), "n_ok": n_ok,
        "all_ok": n_ok == len(results),
        "results": results,
    }
    (OUTPUTS / "pipeline_run_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (LOGS / "pipeline_run.log").write_text("\n".join(log_lines), encoding="utf-8")
    emit(f"\nReport: outputs/pipeline_run_report.json")
    sys.exit(0 if report["all_ok"] else 1)


if __name__ == "__main__":
    main()
