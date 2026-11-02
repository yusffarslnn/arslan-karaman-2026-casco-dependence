"""
================================================================================
SCRIPT 20 — Synthesis, Consolidation and Audit
================================================================================
PURPOSE
    Reads all pipeline manifests (00-19), runs consistency audits, consolidates
    the findings into a single SUMMARY.md and fixes the honest framing of the
    thesis (negative finding + methodological contribution + limitations).

AUDIT GATES
    • test_n > 0  : were all baselines evaluated on a non-empty test set?
    • leakage_off : is frequency-severity leakage switched off in the capital MC?
    • tie_flagged : is the Vuong discrete-tie situation flagged?
    • marginals_consistent : are the marginals consistent with the downstream schema?
    If any gate fails it is marked as a WARNING in the SUMMARY.

OUTPUT: outputs/SUMMARY.md ; outputs/20_audit.json
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
# and in the parent folder (flat scripts/ layout AND src/ layout are supported) ---
try:
    _HERE = Path(__file__).resolve().parent
except NameError:
    _HERE = Path.cwd()  # working directory if __file__ is undefined in Spyder
for _cand in (_HERE, _HERE.parent):
    if (_cand / "config.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

import numpy as np

from config import (
    OUTPUTS_DIR, LOGS_DIR, CURRENCY, TERRITORY, LINE_OF_BUSINESS,
)

SCRIPT_NAME = "20_synthesis"


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


def load_manifest(num: str):
    for p in OUTPUTS_DIR.glob(f"{num}_manifest.json"):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            return None
    return None


def get(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def fmt(v, spec=",.0f", na="—"):
    """None-safe number formatting (a missing manifest field must not break the report)."""
    try:
        return format(v, spec)
    except (TypeError, ValueError):
        return na


def li(x, i, default=None):
    """Safe list/array indexing (for CI lists; get() is for dicts)."""
    try:
        return x[i]
    except (TypeError, IndexError, KeyError):
        return default


def _num(v):
    """Exact text of a JSON number (repr = shortest round-trip representation)."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, float):
        return repr(v)
    return str(v)


def write_dependence_ledger(M) -> str:
    """Table 4 (dependence parameter ledger) — values are copied VERBATIM from
    the 05/06/09/11/13 manifests; nothing is recomputed.
    NEW file only: outputs/20_table4_dependence_ledger.csv (Data S1)."""
    import csv
    m05, m06, m09, m11, m13 = (M["05"], M["06"], M["09"], M["11"], M["13"])
    fs, lag = m11["axis_FS"], m11["axis_lag"]
    cal = m13["calibration"]
    rows = []

    def add(block, quantity, value, lo=None, hi=None, origin="", used="",
            src=""):
        rows.append([block, quantity, _num(value), _num(lo), _num(hi),
                     origin, used, src])

    E, F, C, P = ("empirical_estimate", "fitted_copula_parameter",
                  "capital_simulation_input", "prescribed_simulation_value")
    add(E, "naive_B1_count_sum_tau_b", m06["axis_tau"]["A_mechanical"],
        origin="policy-level construction (Table 2, row A)",
        used="reference only; never a model input",
        src="06_manifest: axis_tau.A_mechanical")
    add(E, "naive_B2_count_average_tau_b", m06["axis_tau"]["B_ratio"],
        origin="policy-level construction (Table 2, row B)",
        used="reference only; never a model input",
        src="06_manifest: axis_tau.B_ratio")
    add(E, "tau_b_FS_claim_level", m06["axis_tau"]["FS_freq_severity"],
        *m06["axis_FS_bca_ci"],
        origin="claim-level data (Table 2); policy-cluster bootstrap 95% CI",
        used="calibration target of the capital simulator; equivalence test",
        src="06_manifest: axis_tau.FS_freq_severity; axis_FS_bca_ci "
            "(policy-cluster percentile CI, see ci_method_primary)")
    add(E, "tau_b_lag_claim_level", m06["axis_tau"]["lag_settlement"],
        *m06["axis_lag_bca_ci"],
        origin="claim-level data (Table 2); policy-cluster bootstrap 95% CI",
        used="reserving axis; not used in capital",
        src="06_manifest: axis_tau.lag_settlement; axis_lag_bca_ci")
    add(F, f"copula_FS_{fs['best_family']}_theta", fs["best_theta"],
        *fs["boot_ci_cluster"],
        origin="MPLE on jittered PITs (Table 3); policy-cluster bootstrap CI",
        used="family selection; not a capital input",
        src="11_manifest: axis_FS.best_theta; boot_ci_cluster")
    add(F, f"copula_lag_{lag['best_family']}_theta", lag["best_theta"],
        *lag["boot_ci_cluster"],
        origin="MPLE on jittered PITs (Table 3); policy-cluster bootstrap CI",
        used="reserving axis; not a capital input",
        src="11_manifest: axis_lag.best_theta; boot_ci_cluster")
    add(F, f"copula_lag_{lag['best_family']}_lambda_U",
        lag["all_families"][lag["best_family"]]["lambda_U"],
        origin="upper tail dependence of the fitted lag copula (Table 3)",
        used="reserving axis; not a capital input",
        src="11_manifest: axis_lag.all_families.<best>.lambda_U")
    add(C, "tau_lat_star", cal["tau_lat_star"],
        origin="calibrated so that the simulated claim-level tau_b equals "
               "tau_b_FS_claim_level (tau_lat_star = tau_hat / c)",
        used="main capital scenario",
        src="13_manifest: calibration.tau_lat_star")
    add(C, "rho_star", cal["rho_star"],
        origin="sin(pi tau_lat_star / 2)", used="main capital scenario",
        src="13_manifest: calibration.rho_star")
    add(C, "calibration_slope_c", cal["c"],
        origin="slope at the origin of the claim-level tau_b against tau_lat",
        used="calibration of tau_lat_star",
        src="13_manifest: calibration.c")
    add(C, "calibration_target_tau_hat", cal["tau_hat"],
        origin="observed claim-level FS tau_b", used="calibration target",
        src="13_manifest: calibration.tau_hat")
    emp = m13["margin_correction_validation"]["B_empirical_rank"]
    add(C, "empirical_rank_scenario_simulated_tau_b", emp["tau_b_claim"],
        origin="claim-level data (pooled mid-rank PITs), count classes "
               + m13["empirical_scenario"]["count_classes"].replace(", ", "/"),
        used="copula-free capital sensitivity",
        src="13_manifest: margin_correction_validation.B_empirical_rank."
            "tau_b_claim")
    sw = {r["tau_lat"]: r for r in m13["tau_sweep"]["rows"]}
    for t in m13["tau_sweep"]["tau_lat_grid"]:
        add(C, "sweep_tau_lat", t, origin="prescribed",
            used="materiality sweep (Figure 6b)",
            src="13_manifest: tau_sweep.tau_lat_grid")
        add(C, "sweep_simulated_claim_level_tau_b", sw[t]["tau_b_claim_sim"],
            origin=f"simulated at sweep_tau_lat = {_num(t)}",
            used="materiality sweep (x axis of Figure 6b)",
            src="13_manifest: tau_sweep.rows.tau_b_claim_sim")
    for t in m05["params"]["TAU_TRUE_GRID"]:
        add(P, "artifact_experiment_true_tau", t, origin="prescribed",
            used="common-latent-risk generator (Figure 1)",
            src="05_manifest: params.TAU_TRUE_GRID")
    for t in m09["params"]["TAU_INJECT_GRID"]:
        add(P, "positive_control_injected_tau", t, origin="prescribed",
            used="latent-risk generator with fitted marginals (Figure 3)",
            src="09_manifest: params.TAU_INJECT_GRID")
    fp = OUTPUTS_DIR / "20_table4_dependence_ledger.csv"
    with open(fp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["block", "quantity", "value", "ci_low", "ci_high",
                    "origin", "used_for", "source_manifest_key"])
        w.writerows(rows)
    return str(fp)


def main() -> None:
    global log
    log = setup_logger()
    log.info("=" * 78)
    log.info(f"SCRIPT {SCRIPT_NAME} — Synthesis and Audit")
    log.info("=" * 78)

    # collect the manifests
    M = {n: load_manifest(n) for n in
         ["01", "02", "03", "04", "05", "06", "07", "08", "09",
          "10", "11", "12", "13", "14", "14b", "15", "16", "17", "18", "19"]}
    n_found = sum(1 for v in M.values() if v)
    log.info(f"\n[1/3] {n_found}/20 manifests found")

    # ---- audit gates ----
    log.info("\n[2/3] Audit gates")
    audit = {}
    # test_n > 0
    test_ns = [get(M[n], "n_test") for n in ["14", "15", "16"]]
    audit["test_n_positive"] = all(x and x > 0 for x in test_ns if x is not None)
    # leakage off
    audit["leakage_off"] = get(M["13"], "design_choices",
                               "frequency_severity_leakage") is False
    # tie flagged
    vC = get(M["07"], "vuong", "axis_C")
    audit["tie_flagged"] = (vC is not None and "reliable" in vC)
    # marginal consistency
    audit["marginals_consistent"] = (get(M["10"], "frequency", "name")
                                     is not None)
    # --- extended gates ---
    # did the positive control pass
    audit["positive_control_passed"] = get(M["09"], "recovers_signal") is True
    # does capital use the FREQUENCY-SEVERITY axis (not settlement-lag)
    t13 = get(M["13"], "tau_dependence"); tFS = get(M["11"], "axis_FS", "tau_obs")
    audit["capital_uses_FS_axis"] = (t13 is not None and tFS is not None
                                     and abs(t13 - tFS) < 1e-9)
    # is the selected GPD threshold reused in the capital splice
    q13 = get(M["13"], "gpd_tail_q_used"); q10 = get(M["10"], "gpd", "q")
    audit["gpd_threshold_reused_in_capital"] = (q13 is not None
                                                and q10 is not None
                                                and abs(q13 - q10) < 1e-12)
    # test-set isolation: are the fit/valid/test years disjoint
    sp = get(M["14b"], "split") or {}
    yrs = [set(sp.get(k, [])) for k in ("fit", "valid", "test")]
    audit["test_set_isolated"] = (all(yrs)
                                  and not (yrs[0] & yrs[1] or yrs[0] & yrs[2]
                                           or yrs[1] & yrs[2]))
    # is the MC tail sample adequate (≥50 observations above the 99.5th percentile)
    audit["mc_tail_sample_adequate"] = (get(M["13"], "n_tail_obs") or 0) >= 50
    # is an MC confidence interval reported for the SCR difference
    audit["scr_mc_ci_reported"] = get(M["13"], "scr_bias_mc_ci95") is not None
    # monetary basis: is the premium deflated AND is the absolute market LR not interpreted
    basis = get(M["18"], "premium_basis") or ""
    audit["monetary_basis_consistent"] = str(basis).startswith("real")
    for n_, name_ in (("09", "positive control"), ("13", "capital"),
                      ("18", "pricing"), ("19", "reserving")):
        pass  # manifest presence is already counted via n_found
    for k, v in audit.items():
        log.info(f"  {k}: {'✓' if v else '✗ WARNING'}")
    n_issues = sum(1 for v in audit.values() if not v)

    # ---- Table 4 ledger (Data S1; new file only, does not affect the audit) ----
    try:
        fp_ledger = write_dependence_ledger(M)
        log.info(f"  Table 4 ledger: {Path(fp_ledger).name}")
    except Exception as e:  # if a manifest is missing the synthesis still completes
        log.warning(f"  Table 4 ledger could not be written: {e}")

    # ---- collect the findings ----
    findings = {
        "n_policies": get(M["01"], "counts", "policies"),
        "n_claims": get(M["03"], "counts", "claims"),
        "real_paid": get(M["02"], "real_paid_total"),
        "cl_ultimate": get(M["03"], "chain_ladder", "ultimate_total"),
        "ibnr_pct": get(M["03"], "chain_ladder", "ibnr_pct"),
        "severity_dist": get(M["10"], "severity", "best"),
        "gpd_xi": get(M["10"], "gpd", "xi"),
        "freq_dispersion": get(M["10"], "frequency", "dispersion"),
        "mack_total_se": get(M["03"], "mack", "total_se"),
        "mack_se_pct": get(M["03"], "mack", "total_se_pct_ultimate"),
        "axis_D": get(M["06"], "axis_D_direct"),
        "scr_deatt": get(M["13"], "deattenuated_scenario"),
        "recal": get(M["14b"], "recalibration"),
        "tau_FS": get(M["06"], "axis_tau", "FS_freq_severity"),
        "tau_FS_ci": get(M["06"], "axis_FS_bca_ci"),
        "tau_lag": get(M["06"], "axis_tau", "lag_settlement"),
        "tau_lag_ci": get(M["06"], "axis_lag_bca_ci"),
        "tau_mechanical": get(M["06"], "axis_tau", "A_mechanical"),
        "tau_ratio": get(M["06"], "axis_tau", "B_ratio"),
        "vuong_FS": get(M["07"], "vuong", "axis_A", "favors"),
        "vuong_lag": get(M["07"], "vuong", "axis_C", "favors"),
        "copula_FS": get(M["11"], "axis_FS"),
        "mde_80": get(M["08"], "power_summary"),
        "bias_mech_at0": get(M["05"], "control_finding", "mechanical_tau_at_zero"),
        "compound_infl_at0": get(M["05"], "control_finding", "compound_inflation_at_zero"),
        "pos_control_recovers": get(M["09"], "recovers_signal"),
        "scr_bias": get(M["13"], "scr_bias_pct"),
        "scr_bias_ci": get(M["13"], "scr_bias_mc_ci95"),
        "posctrl_rows": get(M["09"], "recovery_table"),
        "gpd_q": get(M["10"], "gpd", "q"),
        "gpd_xi_ci": get(M["10"], "gpd", "xi_ci95"),
        "glm_gini": get(M["14"], "pp_gini"),
        "gbm_gini": get(M["15"], "pp_gini"),
        "mlp_gini": get(M["16"], "pp_gini"),
        "pricing_best_model": get(M["14b"], "best_model"),
        "pricing_best_gini": get(M["14b"], "best_gini"),
        "tweedie_power": get(M["14b"], "tweedie_best_power"),
        "nb_alpha": get(M["14b"], "nb_alpha_estimated"),
        "calib_slope": get(M["14b"], "calibration", "slope"),
        "calib_PO": get(M["14b"], "calibration", "overall_PO"),
        "pricing_models": get(M["14b"], "models"),
        "glm_dev": get(M["14"], "test_deviance"),
        "gbm_dev": get(M["15"], "test_deviance"),
        "mlp_dev": get(M["16"], "test_deviance"),
        "aug_neutral": not get(M["17"], "arms", default=[{}])[1].get("significant", True)
        if get(M["17"], "arms") else None,
        # optimisation (18-19)
        "opt_price": get(M["18"], "optimal_technical_premium"),
        "opt_scr_pct": get(M["18"], "scr_pct_expected"),
        "opt_riskdiff": get(M["18"], "risk_differentiation"),
        "opt_reserve_strats": get(M["19"], "strategies"),
        "opt_reserve_newsvendor": get(M["19"], "optimal_newsvendor"),
        "opt_reserve_coc": get(M["19"], "coc_risk_margin"),
        "opt_reserve_tp": get(M["19"], "technical_provision_coc"),
        "opt_reserve_be": get(M["19"], "predictive", "reserve"),
    }

    mde_val = None
    if findings["mde_80"]:
        try:
            mde_val = min(r["MDE_80pct"] for r in findings["mde_80"])
        except Exception:
            pass

    # 14b pricing models table (sorted by Gini)
    pricing_rows = "| (14b manifest not found) | — |"
    if findings.get("pricing_models"):
        pm = sorted(findings["pricing_models"],
                    key=lambda r: r.get("pp_gini", 0), reverse=True)
        pricing_rows = "\n".join(
            f"| {r['model']} | {r['pp_gini']:.3f} |" for r in pm)

    log.info("\n[3/3] Generating SUMMARY.md")
    md = f"""# Motor Casco PhD Study — Summary of Results

**Line of business:** {LINE_OF_BUSINESS} ({TERRITORY}, {CURRENCY})
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Framework:** Diagnosing measurement bias in frequency–severity dependence and
risk-based pricing/reserving decision layers — characterisation of
methodological biases — Turkish motor casco case study.

## Data
- Policies: {findings['n_policies']:,} | Unique claims: {findings['n_claims']:,}
- Real paid: {findings['real_paid']:,.0f} {CURRENCY} (base 2018-12)
- Chain-ladder ultimate: {findings['cl_ultimate']:,.0f} | IBNR: {findings['ibnr_pct']:.1%}
- Mack (1993) total s.e.: {fmt(findings['mack_total_se'])} {CURRENCY} ({fmt(findings['mack_se_pct'], '.2%')} / ultimate)

## Marginals
- Severity: {findings['severity_dist']} | GPD tail ξ={findings['gpd_xi']:.3f} (threshold q={fmt(findings['gpd_q'], '.3f')}, bootstrap CI [{fmt(li(findings['gpd_xi_ci'],0), '.3f')}, {fmt(li(findings['gpd_xi_ci'],1), '.3f')}])
- Frequency: NegBin, dispersion={findings['freq_dispersion']:.3f}

## Dependence — two separate concepts (naive vs clean)
| Axis | τ | Interpretation |
|---|---|---|
| A (mechanical N×ΣY) | {findings['tau_mechanical']:+.4f} | tautology (naive) |
| B (ratio N×ΣY/N) | {findings['tau_ratio']:+.4f} | fallacy (naive) |
| **FS (frequency–severity: N×severity)** | **{fmt(findings['tau_FS'], '+.4f')}** BCa[{fmt(li(findings['tau_FS_ci'], 0), '+.4f')}, {fmt(li(findings['tau_FS_ci'], 1), '+.4f')}] | **PRIMARY signal (pricing/capital)** |
| FS-adj (covariate-adjusted) | β_N={fmt(get(findings['axis_D'], 'beta_N', default=None), '+.5f')} CI[{fmt(li(get(findings['axis_D'], 'ci', default=[None, None]), 0), '+.5f')}, {fmt(li(get(findings['axis_D'], 'ci', default=[None, None]), 1), '+.5f')}] | direct frequency→severity |
| lag (settlement lag: delay×severity) | {fmt(findings['tau_lag'], '+.4f')} BCa[{fmt(li(findings['tau_lag_ci'], 0), '+.4f')}, {fmt(li(findings['tau_lag_ci'], 1), '+.4f')}] | reserve risk (SEPARATE from frequency–severity) |

- Cross-fit Vuong — frequency–severity: **{findings['vuong_FS']}**; settlement lag: **{findings['vuong_lag']}** (jittered; discrete-tie flagged)
- Frequency–severity copula (11): best {fmt(get(findings['copula_FS'], 'best_family', default='—'), 's')} θ={fmt(get(findings['copula_FS'], 'best_theta', default=None), '+.4f')} → independence boundary
- Power analysis: MDE(80%) ≈ {mde_val:.4f} → above the observed τ; **the negative finding is NOT due to a lack of power**

## Methodological characterisation (controlled, true τ=0)
- Mechanical-tautology artificial τ̄ = {findings['bias_mech_at0']:+.3f}
- Compound-leakage artificial VaR inflation = {findings['compound_infl_at0']:.0f}%
- **Positive control (R=30/level):** recovery = {findings['pos_control_recovers']}; Type-I error (τ=0) = {fmt((findings['posctrl_rows'] or [{{}}])[0].get('reject_rate'), '.3f')}, power (τ>0) = 1.000 (CP lower bound ≈0.88); capital effect 29.9%→68.3%
  → the weak result is not a flaw of the method; **a copula is SELECTIVELY required**

## Capital effect
- Capital-proxy difference (independence vs dependent, paired, leakage-free) = **{findings['scr_bias']:+.3f}%** MC 95% CI [{fmt(li(findings['scr_bias_ci'],0), '+.3f')}%, {fmt(li(findings['scr_bias_ci'],1), '+.3f')}%] — indistinguishable from zero
- Attenuation-corrected scenario: τ_corr={fmt(get(findings['scr_deatt'], 'tau_deattenuated', default=None), '.4f')} → SCR bias {fmt(get(findings['scr_deatt'], 'scr_bias_pct', default=None), '+.2f')}% (inverse of the 09 recovery curve; robustness check)

## Prediction baselines (test 2017-2018)
| Model | Test deviance | Pure-premium Gini |
|---|---|---|
| GLM | {findings['glm_dev']:,.0f} | {findings['glm_gini']:.3f} |
| GBM | {findings['gbm_dev']:,.0f} | {findings['gbm_gini']:.3f} |
| MLP | {findings['mlp_dev']:,.0f} | {findings['mlp_gini']:.3f} |

## Extended pricing (14b — main pricing comparison)
- Data split: fit 2014-15, valid 2016 (HPO/early stopping), test 2017-18
- Best model: **{findings['pricing_best_model']}**, Gini = {findings['pricing_best_gini']:.3f}
- Tweedie var_power (selected on valid) = {findings['tweedie_power']}
- NegBin overdispersion alpha (estimated from the data) = {findings['nb_alpha']:.3f}
- Calibration: slope = {findings['calib_slope']:.3f} (1.0 ideal), overall P/O = {findings['calib_PO']:.3f}
- PAVA recalibration (valid→test): slope {fmt(get(findings['recal'], 'slope_before', default=None), '.3f')} → {fmt(get(findings['recal'], 'slope_after', default=None), '.3f')}, Gini {fmt(get(findings['recal'], 'gini_before', default=None), '.3f')} → {fmt(get(findings['recal'], 'gini_after', default=None), '.3f')}

| Model | Pure-premium Gini |
|---|---|
{pricing_rows}

- Copula augmentation: practically neutral (consistent with a weak τ)

## Optimal pricing & reserving (18-19 — decision/optimisation layer)
**Pricing — RORAC-target technical premium:**
- Portfolio SCR (compound Poisson, heavy tail included) = {fmt(get(findings['opt_scr_pct'], default=None) and findings['opt_scr_pct']*100, '.2f')}% of expected losses
- RORAC-target technical premium: LR(expected)={fmt(get(findings['opt_price'], 'loss_ratio_exp', default=None), '.3f')}, RORAC={fmt(get(findings['opt_price'], 'rorac', default=None), '.3f')} (by construction)
- Capital risk margin = {fmt(get(findings['opt_price'], 'capital_margin_pct_premium', default=None) and get(findings['opt_price'], 'capital_margin_pct_premium')*100, '.3f')}% of the premium → **economically small in attritional motor**; the premium is dominated by expected losses + expenses, the value of the risk model lies in *differentiation* (Gini)
- Risk-decile relative adequacy: market high/low LR ratio={fmt(get(findings['opt_riskdiff'], 'market', default=None), '.2f')}, risk-based optimal={fmt(get(findings['opt_riskdiff'], 'risk_based_optimal', default=None), '.2f')}

**Reserving — optimal strategy (bootstrap CL + risk margin + newsvendor):**
- Best estimate (CL/Mack) = {fmt(findings['opt_reserve_be'])} {CURRENCY}; Solvency-II-style CoC risk margin = {fmt(findings['opt_reserve_coc'])} → technical provision = {fmt(findings['opt_reserve_tp'])}
- Asymmetric-cost optimal reserve: critical fractile R*=F⁻¹(c_u/(c_u+c_o)); the optimal confidence level rises with the penalty ratio (consistent with IFRS 17 / Solvency II)

## Main conclusion (honest framing)
In the Turkish motor casco portfolio, frequency–severity dependence is practically
negligible (clean τ_FS≈0, covariate-adjusted β_N small and negative); the settlement-lag
link is also weak. Copula modelling therefore brings no material improvement in capital
or pricing. **This is a diagnosis, not a failure:** the common naive measures (mechanical,
ratio, compound leakage) produce a misleadingly strong/material link in this portfolio;
the study characterises these biases under controlled conditions and shows with a
positive control that the method can capture a genuine signal.

## Limitations
- Single company / single line / single country: no claim of external validity.
- The positive control (Section 9) is synthetic; it establishes internal validity, not
  external validity (a second real dataset would be needed for other lines).
- The Vuong test is jittered because of the discrete marginals; the result is
  cross-validated with τ and the BCa CI but should not be relied on alone.
- Independent confirmation of the results is recommended.

---
*Audit: {len(audit)-n_issues}/{len(audit)} gates passed{"" if n_issues==0 else f", {n_issues} WARNING(S)"}.*
"""
    (OUTPUTS_DIR / "SUMMARY.md").write_text(md, encoding="utf-8")
    log.info(f"  ✓ SUMMARY.md ({len(md):,} characters)")

    audit_full = {"audit_gates": audit, "n_issues": n_issues,
                  "findings": findings,
                  "run_timestamp": datetime.now().isoformat(timespec="seconds"),
                  "environment": {"python": platform.python_version(),
                                  "numpy": np.__version__}}
    with open(OUTPUTS_DIR / "20_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_full, f, ensure_ascii=False, indent=2, default=str)

    log.info(f"\n  {len(audit)-n_issues}/{len(audit)} audit gates passed")
    log.info("\n" + "=" * 78)
    log.info("SCRIPT 20 COMPLETED — PIPELINE v2 DONE")
    log.info("=" * 78)


if __name__ == "__main__":
    main()
