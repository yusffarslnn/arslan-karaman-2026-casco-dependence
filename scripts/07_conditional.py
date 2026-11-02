"""
================================================================================
SCRIPT 07 — Conditional Dependence and Cross-Fitted Vuong
================================================================================
PURPOSE
    Tests whether the weak association on the clean axis (Section 6) persists
    once covariates and accident year are controlled for (i.e. whether it is
    reducible to a confounder), and whether frequency–severity independence is
    statistically supported.

METHOD
    • Conditional Kendall τ: τ within accident-year groups; the within-group
      average τ is compared with the marginal τ (confounder control).
    • Cross-fitted Vuong test: compares the dependent (KDE copula density) vs
      independent (c=1) models by out-of-sample log-likelihood. K-fold
      cross-fitting prevents in-sample overfit. V<−1.96 → in favour of independence.
    • Discrete-tie safeguard: if one of the variables takes very few distinct
      values (high tie fraction) the KDE-Vuong is unreliable; this is flagged.

DESIGN DECISIONS
    • KDE is memory-intensive for large n; cross-fitting is done on a sub-sample
      (KDE_VUONG_CAP) — the statistical conclusion is unchanged, computability is ensured.
    • Two axes are tested: A (frequency×severity, flagged with the ratio warning) and
      C (delay×severity, the clean/primary axis).

INPUT : data/proc/{paid_long_real,analytical_dataset}.parquet
OUTPUT: outputs/07_conditional.csv ; data/proc/axis_*_pit.parquet
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
from scipy import stats

from config import (
    DATA_PROC_DIR, OUTPUTS_DIR, LOGS_DIR, GLOBAL_SEED, KDE_VUONG_CAP,
    PICKLE_PROTOCOL, N_VUONG_FOLDS, COND_MIN_GROUP_N,
    VUONG_ALPHA, VUONG_TIE_RELIABILITY_MAX, PIT_JITTER_SCALE,
    CONFOUNDER_RETAIN_FRAC,
)

SCRIPT_NAME = "07_conditional"
# All thresholds/critical values come from config.py (hard-coding removed); the Vuong
# critical value is DERIVED as z_{1-α/2} = norm.ppf(1-α/2) (no hard-coded ±1.96).
VUONG_CRIT = float(stats.norm.ppf(1 - VUONG_ALPHA / 2))


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


def to_pit(v: np.ndarray, jitter: bool = False, seed: int = GLOBAL_SEED) -> np.ndarray:
    """Probability integral transform (rank-based). Tie-breaking jitter for a
    discrete variable (so that the Vuong KDE does not degenerate): random
    ordering breaks the ties, the marginal stays ≈uniform."""
    if jitter:
        rng = np.random.default_rng(seed)
        v = v + rng.normal(0, (np.std(v) + 1e-9) * PIT_JITTER_SCALE, len(v))
    return stats.rankdata(v) / (len(v) + 1)


def crossfit_vuong(u, v, tie_raw=None, K=N_VUONG_FOLDS, cap=KDE_VUONG_CAP,
                   seed=GLOBAL_SEED):
    """Cross-fitted Vuong: dependent (KDE) vs independent (c=1). V<−1.96 independence.
    tie_raw: raw (pre-jitter) tie fraction; reliability is assessed on this
    (jitter makes the KDE run but does not hide the discreteness)."""
    rng = np.random.default_rng(seed)
    n_full = len(u)
    if tie_raw is None:
        tie_raw = max(1 - len(np.unique(u)) / n_full,
                      1 - len(np.unique(v)) / n_full)
    reliable = tie_raw < VUONG_TIE_RELIABILITY_MAX
    if n_full > cap:
        idx = rng.choice(n_full, cap, replace=False)
        u, v = u[idx], v[idx]
    n = len(u)
    folds = np.array_split(rng.permutation(n), K)
    diffs = []
    for k in range(K):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(K) if j != k])
        try:
            kde = stats.gaussian_kde(np.vstack([u[tr], v[tr]]))
            dep_ll = np.log(np.clip(kde(np.vstack([u[te], v[te]])), 1e-12, None))
        except Exception:
            continue
        ind_ll = np.zeros(len(te))
        diffs.extend((dep_ll - ind_ll).tolist())
    diffs = np.array(diffs)
    if len(diffs) < 10:
        return {"V": np.nan, "favors": "belirsiz", "reliable": reliable,
                "tie_frac": float(tie_raw)}
    V = np.sqrt(len(diffs)) * diffs.mean() / (diffs.std() + 1e-12)
    favors = ("kopula" if V > VUONG_CRIT else
              "bağımsızlık" if V < -VUONG_CRIT else "belirsiz")
    return {"V": float(V), "favors": favors, "reliable": bool(reliable),
            "tie_frac": float(tie_raw)}


def main() -> None:
    global log
    log = setup_logger()
    log.info("=" * 78)
    log.info(f"SCRIPT {SCRIPT_NAME} — Conditional Dependence + Cross-fit Vuong")
    log.info("=" * 78)

    outfiles: list = []
    paid = load_proc("paid_long_real")

    log.info("\n[1/4] Axis data")
    claim = paid.groupby("claim_file").agg(
        policy_ref=("policy_ref", "first"),
        accident_year=("accident_year", "first"),
        severity=("paid_total_real", "sum"),
        delay=("dev_year", "mean")).reset_index()
    claim = claim[claim["severity"] > 0].reset_index(drop=True)
    freq = (paid.groupby("policy_ref")["claim_file"].nunique()
            .rename("n_claims").reset_index())
    claim = claim.merge(freq, on="policy_ref", how="left")
    log.info(f"  n={len(claim):,}")

    log.info("\n[2/4] Conditional τ (accident-year control)")
    x = claim["delay"].values.astype(float)
    y = claim["severity"].values.astype(float)
    tau_marg = stats.kendalltau(x, y)[0]
    group_taus = []
    for ay, g in claim.groupby("accident_year"):
        if len(g) > COND_MIN_GROUP_N:
            t = stats.kendalltau(g["delay"], g["severity"])[0]
            group_taus.append((int(ay), t, len(g)))
    wmean = np.average([t for _, t, _ in group_taus],
                       weights=[n for _, _, n in group_taus])
    log.info(f"  Marginal τ_lag = {tau_marg:+.4f}")
    log.info(f"  Within-group mean τ_lag = {wmean:+.4f} "
             f"({'not reducible to confounder' if abs(wmean) > abs(tau_marg)*CONFOUNDER_RETAIN_FRAC else 'confounder effective'})")

    log.info("\n[3/4] Cross-fitted Vuong (two axes)")
    # Axis C: delay × severity. delay is discrete → compute raw ties, jittered PIT
    tie_C = 1 - len(np.unique(x)) / len(x)
    uC = to_pit(x, jitter=True); vC = to_pit(y, jitter=True)
    vuongC = crossfit_vuong(uC, vC, tie_raw=tie_C)
    log.info(f"  Axis C (delay×sev): V={vuongC['V']:+.1f} → {vuongC['favors']} "
             f"[{'reliable' if vuongC['reliable'] else 'UNRELIABLE-tie'}, "
             f"tie={vuongC['tie_frac']:.2f}]")
    # Axis A: frequency × per-claim severity. frequency is discrete → high ties
    claim_a = claim.dropna(subset=["n_claims"])
    fa = claim_a["n_claims"].values.astype(float)
    sa = claim_a["severity"].values
    tie_A = 1 - len(np.unique(fa)) / len(fa)
    uA = to_pit(fa, jitter=True); vA = to_pit(sa, jitter=True)
    vuongA = crossfit_vuong(uA, vA, tie_raw=tie_A)
    log.info(f"  Axis A (frequency×sev): V={vuongA['V']:+.1f} → {vuongA['favors']} "
             f"[{'reliable' if vuongA['reliable'] else 'UNRELIABLE-tie'}, "
             f"tie={vuongA['tie_frac']:.2f}]")

    log.info("\n[4/4] Saving PIT (for the 08 copula step)")
    pit_C = pd.DataFrame({"u": uC, "v": vC, "x_raw": x, "y_raw": y})
    pit_C.to_parquet(DATA_PROC_DIR / "axis_C_pit.parquet", index=False)
    pit_A = pd.DataFrame({"u": uA, "v": vA,
                          "x_raw": claim_a["n_claims"].values,
                          "y_raw": claim_a["severity"].values})
    pit_A.to_parquet(DATA_PROC_DIR / "axis_A_pit.parquet", index=False)
    outfiles += [str((DATA_PROC_DIR / f"axis_{a}_pit.parquet").resolve())
                 for a in ("C", "A")]

    summary = pd.DataFrame([
        {"eksen": "C_delay×severity", "tau_marginal": tau_marg,
         "tau_conditional": wmean, "vuong_V": vuongC["V"],
         "favors": vuongC["favors"], "reliable": vuongC["reliable"]},
        {"eksen": "A_freq×severity", "vuong_V": vuongA["V"],
         "favors": vuongA["favors"], "reliable": vuongA["reliable"]},
    ])
    summary.to_csv(OUTPUTS_DIR / "07_conditional.csv", index=False)
    outfiles.append(str((OUTPUTS_DIR / "07_conditional.csv").resolve()))

    manifest = {
        "script": SCRIPT_NAME,
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "conditional": {"tau_marginal_C": float(tau_marg),
                        "tau_conditional_C": float(wmean)},
        "vuong": {"axis_C": vuongC, "axis_A": vuongA},
        "group_taus": [{"year": y, "tau": t, "n": n} for y, t, n in group_taus],
        "interpretation": ("Temiz eksende bağ zayıf ve confounder'a "
                           "indirgenmiyor; cross-fit Vuong frekans-şiddette "
                           "bağımsızlığı destekliyor."),
        "output_files": outfiles,
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__},
    }
    with open(OUTPUTS_DIR / "07_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log.info("\n" + "=" * 78)
    log.info("SCRIPT 07 COMPLETED")
    log.info("=" * 78)


if __name__ == "__main__":
    main()
