"""
================================================================================
config.py — Central Configuration
================================================================================
PhD Thesis — Optimal Premium Pricing and Claims Reserving Strategies
for Motor Casco Insurance
(Diagnosing Frequency–Severity Dependence: From Measurement Bias to Risk-Based
Premium Pricing and Claims Reserving Strategies — YTÜ Department of Statistics)

All scripts read their configuration from this file. No constant is hard-coded
elsewhere; this is the single source of truth for reproduction.
================================================================================
"""
from __future__ import annotations
import os
from pathlib import Path

# --- Directory layout -------------------------------------------------------
try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    # Spyder does not define __file__ in some modes → use the working directory
    PROJECT_ROOT = Path.cwd()
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROC_DIR = PROJECT_ROOT / "data" / "proc"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "figures"
LOGS_DIR = PROJECT_ROOT / "logs"
for _d in (DATA_RAW_DIR, DATA_PROC_DIR, OUTPUTS_DIR, FIGURES_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Line of business / context ---------------------------------------------
CURRENCY = "TRY"
TERRITORY = "TR"
LINE_OF_BUSINESS = "Motor-Casco"

# --- Raw data files (Excel) + Parquet bridge --------------------------------
# The large Excel file (264 MB) cannot be opened directly with pandas (memory);
# 00_data_ingest converts these to Parquet via read-only streaming, and the
# subsequent scripts read Parquet. RAW_FILES holds both the Excel and the
# derived Parquet name.
RAW_FILES = {
    "uretim": {"excel": "2014_2018_Kasko_veri.xlsx", "sheet": "Üretim",
               "parquet": "uretim.parquet"},
    "odenen": {"excel": "Odenen_Hasar_2014_2020.xlsx", "sheet": "Ödenen_Hasar",
               "parquet": "odenen.parquet"},
    "muallak": {"excel": "Muallak_Hasar_2014_2020.xlsx", "sheet": "Muallak_Hasar",
                "parquet": "muallak.parquet"},
}
CPI_CANDIDATES = ["cpi_monthly.xlsx", "cpi_monthly.xls", "cpi_monthly.csv"]
CPI_SHEET_CANDIDATES = ["TUFE_Veri", "TÜFE_Veri", "data", "cpi", "Sheet1"]

# --- Central column map (COLMAP) --------------------------------------------
COLMAP = {
    "policy_ref": "Police_Ref",
    "policy_start": "Ilk_Police_Baslangic",
    "policy_end": "Ilk_Police_Bitis",
    "exposure": "Kazanikan_Adet",
    "claim_status": "Hasar_Durum",
    "premium": "Prim_TL",
    "premium_total": "Toplam_Prim_TL",
    "premium_kasko": "Kasko_Prim_TL",
    # paid
    "paid_amount": "Ödenen_Hasar_TL",
    "paid_yearmonth": "Ödenen_Yıl_Ay",
    "claim_date": "Hasar_Tarihi",
    "claim_file": "Hasar_Dosya_No",
    "salvage": "Sovtaj_Tutar_TL",
    # case reserve
    "reserve_amount": "Muallak_Tutar_TL",
    "reserve_yearmonth": "Muallak_Yıl_Ay",
    # covariates
    "vehicle_age": "Arac_Yasi",
    "insured_age": "Sigortali_Yasi",
    "horsepower": "Horse_Power",
    "ccm": "CCM",
    "vehicle_value": "Arac_Bedeli",
    "gender": "Cinsiyet",
    "body_code": "Body_Code",
}
NUMERIC_COVARS = ["Arac_Yasi", "Sigortali_Yasi", "Horse_Power", "CCM",
                  "Arac_Bedeli"]
CATEG_COVARS = ["Cinsiyet", "Body_Code"]

# --- Time split -------------------------------------------------------------
TRAIN_YEARS = (2014, 2015, 2016)
TEST_YEARS = (2017, 2018)
# Inner validation split for hyperparameter selection / early stopping
# (used by the pricing models in 14b): train ⊂ TRAIN_YEARS
FIT_YEARS = (2014, 2015)      # model fitting
VALID_YEARS = (2016,)         # hyperparameter / early-stopping validation

# --- Inflation / deflation --------------------------------------------------
INFLATION_BASE_YEAR_MONTH = 201812     # real base month
INFLATION_STRICT_MODE = True           # error on missing CPI; no hard-coded fallback
CLIP_NEGATIVE_PAID = False             # salvage (negative paid) is preserved

# --- HPO (loss ratio) valuation date ----------------------------------------
HPO_VALUATION_YEAR = 2020
HPO_VALUATION_MONTH = 12

# --- Risk / capital ---------------------------------------------------------
VAR_LEVELS = (0.95, 0.99, 0.995)
SOLVENCY_ALPHA = 0.995                 # Solvency II SCR level
N_MONTE_CARLO = 200_000

# --- Reproducibility --------------------------------------------------------
GLOBAL_SEED = 20240517
PICKLE_PROTOCOL = 4
FIG_DPI = 150

# --- Performance caps (large n; do not alter the statistics, only make them tractable)
KDE_VUONG_CAP = 8000      # cross-fit KDE-Vuong sub-sample
GOF_BOOTSTRAP_CAP = 5000  # copula GoF bootstrap sample
GOF_DIST_CAP = 20000      # marginal GoF bootstrap sample
TREE_TRAIN_CAP = None     # R2 (Reviewer 1, point 4): NO cap — tree/MLP models are
                          # trained on the SAME full fit sample as the GLMs. In R1
                          # this was a random sub-sample of 200_000 policies (for speed).

# --- External API -----------------------------------------------------------
EVDS_API_KEY = os.environ.get("EVDS_API_KEY", "")

# =============================================================================
# SCRIPT-LEVEL PARAMETERS (hard-code removal — this is the single source of truth)
# The constants below used to be embedded inside the respective scripts; all
# scripts now read them from config. To change a value, editing this file
# alone is sufficient.
# =============================================================================

# --- 00_data_ingest ----------------------------------------------------------
INGEST_LOG_EVERY = 100_000        # progress-log interval during streaming

# --- 01_clean ----------------------------------------------------------------
STUDY_YEAR_MIN = 2014             # study window (policy year)
STUDY_YEAR_MAX = 2018
CLAIM_STATUS_CLAIMED = "HASARLI"  # claimed-policy label in the production file

# --- 02_inflation ------------------------------------------------------------
CPI_SLOPE_SHOCKS = (-0.05, 0.05)  # CPI slope sensitivity shocks
DEFAULT_PAID_MONTH = 6            # default when the payment month is missing (mid-year)
DEFAULT_RESERVE_MONTH = 12        # default when the case-reserve month is missing (year-end)

# --- 03_analytical (Chain-Ladder + Mack 1993) --------------------------------
MACK_ENABLED = True               # Mack distribution-free standard error calculation

# --- 04_eda ------------------------------------------------------------------
HILL_K_GRID = [25, 50, 100, 200, 500, 1000, 2000]

# --- 05_bias_theory ----------------------------------------------------------
BIAS_N_GRID = [1000, 5000, 25000, 100000]
BIAS_TAU_TRUE_GRID = [0.0, 0.05, 0.10, 0.20]
BIAS_N_REP = 200
BIAS_SEV_MEANLOG = 8.0
BIAS_SEV_SDLOG = 1.0
BIAS_FREQ_LAMBDA = 0.35
BIAS_MC_CAP = 20000
BIAS_NMAX = 20                    # maximum claims per policy (simulation)

# --- 06_dependence -----------------------------------------------------------
TAU_N_BOOTSTRAP = 1000            # iid BCa bootstrap replications (point CI)
TAU_BCA_CAP = 50000               # BCa jackknife sub-sample cap
AXIS_D_N_BOOTSTRAP = 500          # axis FS-adj cluster (policy) bootstrap replications
TAU_CLUSTER_N_BOOTSTRAP = 500     # POLICY-CLUSTER bootstrap replications for τ: the
                                  # claims of the same policy are resampled together;
                                  # the claim-level iid CI ignores within-cluster
                                  # correlation and may be too NARROW → cluster CI is primary

# --- 07_conditional ----------------------------------------------------------
N_VUONG_FOLDS = 5
COND_MIN_GROUP_N = 100            # minimum group size for conditional tau
VUONG_ALPHA = 0.05               # Vuong two-sided decision threshold; the critical
                                 # value is DERIVED as z_{1-α/2} = norm.ppf(1-α/2)
                                 # (hard-coded ±1.96 removed)
VUONG_TIE_RELIABILITY_MAX = 0.30 # if the raw (pre-jitter) tie rate exceeds this
                                 # threshold, KDE-Vuong is flagged UNRELIABLE
PIT_JITTER_SCALE = 1e-4          # discrete PIT tie-breaking jitter: σ = scale·std(v)
CONFOUNDER_RETAIN_FRAC = 0.5     # if within-group τ retains this fraction of marginal τ,
                                 # the association is not deemed reducible to the confounder

# --- 08_power ----------------------------------------------------------------
POWER_ALPHA = 0.05
POWER_TARGET = 0.80
POWER_TAU_GRID = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15]
N_POWER_SIM = 300
POWER_SIM_CAP = 5000

# --- 09_positive_control -----------------------------------------------------
TAU_INJECT_GRID = [0.0, 0.10, 0.25, 0.40]
N_SYNTH_POSCTRL = 200_000
POSCTRL_N_REP = 30                # independent synthetic replications at each τ level:
                                  # mean recovery, bias, RMSE, Type-I error at τ=0
                                  # and power at positive τ are reported
POSCTRL_NMAX = 30                 # maximum claims per policy (synthetic)
POSCTRL_DEFAULT_NB = (1.1244, 0.811)        # NegBin (r, p) — fallback if 10 is absent
POSCTRL_DEFAULT_LN = (1.4623, 0.0, 1463.45) # LogNormal (s, loc, scale) fallback

# --- 10_marginals -------------------------------------------------------------
# The POT-GPD threshold is now selected in a DATA-DRIVEN way (fixed 0.90 hard-code → formula).
# A GPD is fitted to the exceedances over a grid of candidate quantiles; the threshold
# that MINIMISES the KS distance of the exceedances to the fitted GPD (while leaving
# at least GPD_MIN_EXCEED exceedances) is selected. The tail regime is therefore not
# masked by a single-k choice (Pickands/POT).
GPD_THRESHOLD_Q = 0.90            # backward-compatibility / fallback threshold (quantile)
GPD_THRESHOLD_Q_GRID = [0.85, 0.90, 0.925, 0.95, 0.975]  # candidate POT thresholds
GPD_MIN_EXCEED = 500             # minimum number of exceedances in threshold selection (variance)
FREQ_FIT_METHOD = "mle"          # frequency estimation: "mle" (likelihood) | "mom"
                                 # (moments). MoM: p=μ/σ², r=μp/(1−p); MLE is more
                                 # efficient (optimises the dispersion directly).
GPD_XI_BOOT = 500                # exceedance-bootstrap CI replications for ξ
GPD_SENS_MC = 200_000            # threshold sensitivity: MC moment sample from the splice

# --- 11_copulas ----------------------------------------------------------------
COPULA_N_BOOTSTRAP = 500
COPULA_TAU_CLAMP = 0.95          # upper bound on |τ| in the τ→θ mapping (numerical
                                 # stability; hard-coded 0.95 removed)
COPULA_THETA_MAX = 60.0          # Frank/Joe brentq upper search bound
JOE_SERIES_TERMS = 200           # number of terms in the truncated Joe τ(θ) series
CLAYTON_THETA_MAX = 1.0e3        # Clayton θ ceiling (degenerate τ→1 limit)
# Copula dependence is reported on TWO separate axes:
#   axis_FS  : FREQUENCY-SEVERITY (policy N × per-claim severity) — the PRIMARY axis
#              for pricing and compound capital (13_capital consumes it).
#   axis_lag : settlement-LAG × severity (settlement-lag) — for RESERVE risk;
#              the pkl backward-compatibility alias "axis_C" refers to this axis.
COPULA_PRIMARY_AXIS = "axis_FS"  # dependence axis read by 13_capital
CAPITAL_COPULA_MAIN = "gauss"    # main capital scenario: AIC-selected family
                                 # (Gauss; its conditional is analytic). Sensitivity run
                                 # with Frank's closed-form bisection conditional
NB_DISP_BOOT = 200               # policy-bootstrap replications for NB dispersion
                                 # parameter uncertainty (capital sensitivity)
COPULA_FIT_METHOD = "mple"       # "mple": maximum pseudo-likelihood initialised by
                                 # τ-inversion; "tau": inversion only.
                                 # The independence copula is an explicit candidate; AIC+BIC

# --- 12_gof --------------------------------------------------------------------
N_GOF_BOOT = 300
GOF_GRID_POINTS = 50              # number of CvM grid points

# --- 13_capital ----------------------------------------------------------------
N_POLICIES_SIM = 100_000          # portfolio size in each MC replication
# Tail sample for the 99.5% quantile: N_MC_PAIRED×(1−α) observations lie above
# the quantile. 2,000 replications→~10 observations is INSUFFICIENT; 20,000→~100 observations.
N_MC_PAIRED = 20_000              # number of paired replications (tail-sufficient)
CAPITAL_SEED_BLOCKS = 4           # independent seed blocks (seed, seed+1000, ...)
                                  # the block-wise distribution of the SCR difference is reported
CAPITAL_SCR_BOOT = 1000           # paired-replication bootstrap CI for the SCR difference
CAPITAL_NMAX = 30                 # maximum claims per policy (MC)
# The compound capital copula is parameterised by the FREQUENCY-SEVERITY dependence
# (NOT the settlement-lag τ — see 06/11 for the conceptual distinction). Fallback is the per-claim (N×severity) Kendall τ.
TAU_FS_FALLBACK = 0.0065          # frequency-severity τ fallback if copula_fits cannot be read
TAU_CLEAN_FALLBACK = 0.0185       # (backward-compatibility) settlement-lag τ; backward-compatibility only
SEV_TAIL_Q = 0.90                 # fallback for the body→GPD transition quantile; if the
                                  # GPD threshold quantile selected by the marginal fit
                                  # (gpd['q']) exists, THAT is used (splice consistency)
DEATTENUATION_ENABLED = True      # attenuation-corrected τ scenario using the 09
                                  # recovery curve (additional SCR run)

# --- 14_glm --------------------------------------------------------------------
GLM_N_ITER = 25
GLM_RIDGE = 1e-6
GINI_BOOTSTRAP_B = 500

# --- 14b_pricing_models --------------------------------------------------------
TWEEDIE_P_MIN = 1.1               # var_power profile grid
TWEEDIE_P_MAX = 1.9
TWEEDIE_P_STEP = 0.05
TWEEDIE_DEFAULT_P = 1.5           # fixed p for the GBM/package models
GBM14B_MAX_TREES = 1000           # R2: at 300 the frequency GBM on the full sample stopped
                                  # at tree 291 (the ceiling ran out before patience → the ceiling was binding);
                                  # raised so that the number of trees is determined by early stopping alone
GBM14B_LR = 0.08
GBM14B_MAX_DEPTH = 3
GBM14B_N_BINS = 32
GBM14B_PATIENCE = 15
PKG_NUM_BOOST_ROUND = 500         # lightgbm/xgboost maximum rounds
PKG_EARLY_STOPPING = 20
LGB_PARAMS_BASE = {"learning_rate": 0.05, "num_leaves": 31,
                   "min_data_in_leaf": 100}
XGB_PARAMS_BASE = {"eta": 0.05, "max_depth": 5, "min_child_weight": 100}
TWOSTAGE_GBM_ENABLED = True       # two-stage GBM (frequency×severity) — completes the
                                  # structure (direct vs two-stage) × learner (GLM vs
                                  # GBM) 2×2 design
RECALIBRATION_ENABLED = True      # PAVA isotonic recalibration (fitted on valid,
                                  # applied to test). HONEST FINDING: PAVA corrects the
                                  # overall bias (P/O→1); the slope compression
                                  # REMAINS under the valid→test distribution shift
                                  # (0.648→0.554). Reported as such in the paper.
RECAL_N_BINS = 1000               # number of quantile bins before PAVA (speed)
PAIRED_GINI_B = 500               # paired bootstrap replications for the model difference

# --- 15_gbm --------------------------------------------------------------------
GBM_N_TREES = 60
GBM_MAX_DEPTH = 3
GBM_LR = 0.1
GBM_N_BINS = 32

# --- 16_mlp --------------------------------------------------------------------
MLP_HIDDEN = 16
MLP_EPOCHS = 300
MLP_LR = 0.05
MLP_MOMENTUM = 0.9

# --- 17_augmentation -----------------------------------------------------------
AUG_FRACTION = 0.20               # fraction of synthetic rows added
AUG_PAIRED_B = 500                # paired bootstrap replications
AUG_PRACTICAL_DELTA = 0.01        # practical significance threshold (ΔGini)

# =============================================================================
# OPTIMISATION MODULES (18-19) — "Optimal Premium Pricing & Claims Reserving"
# The decision/optimisation layer that ACTUALLY delivers the "optimal ... strategies"
# claim of the thesis/title. The pricing side builds on the validated pure-premium
# model (14b) and the capital model (13); the reserving side builds on
# Chain-Ladder/Mack (03). All constants are here; the scripts read only from here.
# =============================================================================

# --- 18_optimal_pricing --------------------------------------------------------
COC_RATE = 0.06                   # Solvency II cost of capital (annual, CoC)
EXPENSE_RATIO = 0.25              # administration+distribution expense ratio (combined ratio)
TARGET_RORAC = 0.12               # target risk-adjusted return on capital (RORAC)
PRICING_OPT_SOURCE = "best"       # risk-score source: "best" (14b best-preds)
                                  # | "gbm" (15) | "glm" (14)
# R2 (Reviewer 1, round 2, point 4b): on-level (level/trend) adjustment.
# 14b computes THREE scale factors for the selected model's test predictions; none
# uses test-year information, none changes the ranking (Gini):
#   "none"                        : raw prediction (2014-15 level; same as R1)
#   "validation_level"            : actual/predicted ratio in the 2016 validation year
#   "frequency_trend"             : 2014-16 log-linear frequency trend, from the fit
#                                   centre to the test centre
#   "validation_level_then_trend" : first level to 2016, then the 2016→test trend
#                                   (standard actuarial on-level + trend)
# 18_optimal_pricing uses the prediction MULTIPLIED by this factor in the RORAC layer;
# the headline figures obtained with the raw prediction are written to the manifest as a sensitivity.
PRICING_LEVEL_ADJUSTMENT = "validation_level_then_trend"
# Premium principles: for each one, a grid of loading parameters θ (the RORAC
# frontier) is scanned; the θ* reaching the target RORAC is selected.
LOADING_GRID_N = 60               # number of ρ grid points (frontier resolution)
PRICING_SCR_MC_REPS = 2000        # compound-Poisson MC cross-validation of the
                                  # Cornish-Fisher SCR: number of replications
PREMIUM_DEFLATE = True            # deflate the market premium to the 2018-12 real base
                                  # using the policy-year mid-month CPI; the absolute
                                  # market LR is still not interpreted (column
                                  # semantics unclear) — RELATIVE adequacy only

# --- 19_optimal_reserving ------------------------------------------------------
RESERVE_BOOTSTRAP_N = 5000        # England-Verrall bootstrap CL replications
RESERVE_PROCESS_ERROR = True      # whether to add process error (Gamma) to the bootstrap
RESERVE_CONF_LEVELS = [0.75, 0.95, 0.995]   # IFRS17/Solvency confidence levels
RESERVE_DISCOUNT_RATE = 0.0       # real discount rate (with real amounts; 0 = undiscounted)
RESERVE_TRIANGLE_VARIANTS = ("paid", "incurred")  # R2: 19 runs for both triangles;
                                  # "paid" output names are unchanged, "incurred" → 19_incurred_*
# Asymmetric-cost (newsvendor) optimal reserve: critical fractile
# R* = F⁻¹(c_u/(c_u+c_o)); c_u=under-reserving (shortfall/solvency) penalty,
# c_o=over-reserving (capital holding) cost. A grid of penalty ratios is scanned.
RESERVE_PENALTY_GRID = [1.0, 2.0, 3.0, 5.0, 10.0]   # c_u/c_o
RESERVE_COC_RATE = 0.06           # CoC for the risk margin (Solvency II)

# =============================================================================
# R1 REVISION (mathematics-4559487, September 2026) — shared helpers
# All modules use these for publication figures and the manifest environment record.
# =============================================================================
# Publication figure widths (inches): MDPI full width ≈ 18.4 cm; text width
# ≈ 13.8 cm. Figures are drawn AT PRINT SIZE → fonts are not rescaled.
FIG_W_FULL = 7.1          # multi-panel, full page width (extends into the left margin)
FIG_W_TEXT = 5.3          # single-panel, text width
FIG_FONT_PT = 8           # body font size in print (pt); title 8.5, tick labels 7
FIG_DPI_PUB = 600         # MDPI raster requirement (for the PNG copies)


def apply_pub_style():
    """Print-size publication style (MDPI): serif 8 pt, no offset/scientific notation."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.size": FIG_FONT_PT,
        "axes.titlesize": FIG_FONT_PT + 0.5, "axes.labelsize": FIG_FONT_PT,
        "xtick.labelsize": FIG_FONT_PT - 1, "ytick.labelsize": FIG_FONT_PT - 1,
        "legend.fontsize": FIG_FONT_PT - 1, "legend.frameon": False,
        "axes.formatter.useoffset": False, "axes.formatter.limits": (-7, 9),
        "axes.unicode_minus": True, "lines.linewidth": 1.2,
        "savefig.dpi": FIG_DPI_PUB, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
        "pdf.fonttype": 42, "ps.fonttype": 42})


def thousands_formatter(decimals: int = 0):
    """MDPI: comma thousands separator for numbers with ≥5 digits (e.g. 10,000)."""
    from matplotlib.ticker import FuncFormatter
    return FuncFormatter(lambda v, _pos: f"{v:,.{decimals}f}")


def env_info() -> dict:
    """Manifest 'environment' block: interpreter + all critical package versions."""
    import platform, importlib
    info = {"python": platform.python_version()}
    for m in ("numpy", "pandas", "scipy", "matplotlib", "pyarrow",
              "statsmodels", "lightgbm", "xgboost"):
        try:
            info[m] = importlib.import_module(m).__version__
        except Exception:
            info[m] = None
    return info
