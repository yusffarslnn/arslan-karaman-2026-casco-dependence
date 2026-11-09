# Frequency–severity dependence diagnostics in motor casco insurance — analysis code

This repository contains the complete analysis pipeline behind the article

> Arslan, Y.; Karaman, F. **Frequency–Severity Dependence Diagnostics in Motor Casco Insurance: From Measurement Bias to Risk-Based Premium Pricing and Claims Reserving Strategies.** *Mathematics* **2026**, *14*(21), 4104. https://doi.org/10.3390/math14214104 (MDPI; published 9 November 2026; manuscript mathematics-4559487).

It is the code deposit announced in the article's Data Availability Statement. Release `v1.0-mathematics-4559487` is the exact code that produced every number, table and figure of the published version. The archived release carries a persistent identifier: Zenodo DOI https://doi.org/10.5281/zenodo.23894833 (this version) and https://doi.org/10.5281/zenodo.23894832 (all versions).

## Purpose

The pipeline (i) measures frequency–severity dependence in a Turkish motor casco portfolio on *clean* axes and shows how the usual naive constructions manufacture spurious dependence, (ii) characterises those measurement biases in a controlled experiment with a known truth and a synthetic positive control, (iii) fits marginal and copula models with policy-cluster bootstrap uncertainty, and (iv) carries the dependence diagnosis through to capital, pricing and reserving decisions (paired capital experiment, eight pricing models, RORAC-target premiums, Mack/CoC/newsvendor reserves). Every manuscript element is traceable to one module (table below), and an independent verification module re-derives the 32 headline results from the saved intermediate data.

## What is included — and what is not

| Path | Content |
|---|---|
| `scripts/` | 28 Python files: 26 analysis modules (`00`–`24`, including `14b`), the single configuration file `config.py` and the orchestrator `run_pipeline.py` |
| `requirements.txt` | Pinned package versions of the final clean run |
| `pipeline_run_report.json` | Report of the final clean end-to-end run (26/26 modules OK, 26 October 2026): status, duration and log tail of every module |
| `data_s1/` | Data S1, the *minimal dataset* published with the article as Supplementary Material: 83 aggregate CSV files (the numbers behind every table and figure, the verification table `23_verification_table.csv`, Data S1 `README.md` and `FILES.txt`) |
| `LICENSE`, `CITATION.cff` | MIT licence; citation metadata |

**No insurance data are included.** Modules `00`–`24` read the three administrative files of the insurer (policy production, paid-claims ledger, case-reserve ledger) from `scripts/data/raw/`. Those files are policy-level records provided by the insurer under a data-use agreement and are subject to Turkish data-protection law (KVKK); they cannot be shared, and neither can any file derived from them at policy or claim level (`scripts/data/proc/`). Consequently:

- **The full empirical pipeline cannot be run from this repository.** It requires the restricted data in `scripts/data/raw/` (see `RAW_FILES` and `COLMAP` in `scripts/config.py` for the expected files, sheets and column names). Access to the underlying data may be requested through the corresponding author, subject to the data provider's approval.
- **Two modules run without any data** and reproduce two manuscript figures exactly: `05_bias_theory.py` (controlled experiment on synthetic portfolios; Figure 1) and `09_positive_control.py` (synthetic positive control with injected dependence; Figure 4; it falls back to its default marginals when `data/proc/marginal_fits.pkl` is absent). Both take a few seconds.
- **Everything else can be inspected and compared.** Each module documents in its docstring the manuscript sections, tables and figures it produces, and the aggregate outputs of the real run are in `data_s1/`, so every code path can be checked against the published numbers.

## Environment

Python 3.13.9 with the versions pinned in `requirements.txt` (NumPy 2.3.5, pandas 2.3.3, SciPy 1.16.3, statsmodels 0.14.5, LightGBM 4.6.0, XGBoost 3.3.0, matplotlib 3.10.6, pyarrow 21.0.0, openpyxl 3.1.5). Results were produced on macOS (Apple silicon); LightGBM and XGBoost are run single-threaded with fixed seeds, so the pricing results are deterministic on a given platform. The XGBoost Gini index is sensitive to the `xgboost` version at the third decimal.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

All seeds, grids, thresholds, bootstrap sizes and data splits are fixed in `scripts/config.py` (global seed 20240517). No constant is hard-coded elsewhere. The CPI series used for deflation is read from `scripts/data/raw/cpi_monthly.*`; the optional download helper reads the CBRT EVDS key from the environment variable `EVDS_API_KEY` (never from the code).

## Running

Modules are run from the `scripts/` directory; each writes `outputs/<nn>_manifest.json` (parameters, environment, list of output files), `outputs/<nn>_*.csv`, `figures/<nn>_*.{png,pdf}` and `logs/<nn>_*.log`. The directories are created on first use.

```bash
cd scripts
python 05_bias_theory.py          # runs without data (Figure 1)
python 09_positive_control.py     # runs without data (Figure 4)
python run_pipeline.py            # full run; requires the restricted data in data/raw/
```

**Run order** (as executed by `run_pipeline.py`; about 20 minutes on a laptop with the restricted data):
`00, 01, 02, 03, 04, 05, 06, 07, 08, 10, 09, 11, 12, 13, 14, 14b, 15, 16, 17, 18, 22, 19, 21, 24, 23, 20`.
Module `09` runs after `10` because it reuses the fitted marginals; `23` (independent verification) runs after every result module; `20` consolidates the manifests, runs the eleven audit gates and writes `outputs/SUMMARY.md`.

## Module – manuscript element map

| Module | Role (manuscript element) |
|---|---|
| `00_data_ingest.py` | Excel -> Parquet bridge for the three administrative files (memory-safe) |
| `01_clean.py` | Typing, validation, linking of claims to policies, paid-ledger and case-reserve panels |
| `02_inflation.py` | CPI deflation to December-2018 prices (payments, case reserves, premiums); CPI-slope sensitivity |
| `03_analytical.py` | Paid development triangle, chain-ladder, Mack standard error; per-policy pricing target (paid + open case reserves) |
| `04_eda.py` | Descriptive statistics, Hill estimator, per-year HPO diagnostics |
| `05_bias_theory.py` | Controlled experiment under a known truth: count-sum, count-average and shared-uniform constructions (Figure 1) |
| `06_dependence.py` | Naive and clean dependence axes, policy-cluster bootstrap, covariate-adjusted severity regression (Table 2, Figure 2, Table A1) |
| `07_conditional.py` | Conditional PITs with tie jitter; cross-fitted Vuong test |
| `08_power.py` | Minimum detectable effects (i.i.d. tie-corrected and cluster-adjusted) |
| `09_positive_control.py` | Synthetic positive control with injected dependence (Figure 4); runs after 10 |
| `10_marginals.py` | Severity body + GPD tail, negative binomial frequency (Figures 5-6, Table 4 inputs) |
| `11_copulas.py` | Copula families by MPLE, policy-cluster bootstrap with jitter redraw (Table 4) |
| `12_gof.py` | Goodness of fit and Cramer-von Mises independence diagnostics |
| `13_capital.py` | Paired, leakage-free capital experiment with count-coupled Gaussian link and margin correction (Table 6, Figure 7) |
| `14_glm.py` | Pure-numpy GLM baseline (aligned with 15/16) |
| `14b_pricing_models.py` | Eight pricing models on identical samples, Tweedie profile likelihood, paired bootstrap, 2x2 factorial, calibration and on-level factors (Tables 7-8, Figures 8-9, Table B1) |
| `15_gbm.py` | Pure-numpy GBM baseline |
| `16_mlp.py` | Pure-numpy MLP baseline |
| `17_augmentation.py` | Copula-augmentation ablation (not reported in the paper; retained for completeness) |
| `18_optimal_pricing.py` | RORAC-target premium, Cornish-Fisher capital proxy with Monte Carlo check, premium principles, relative adequacy (Section 3.7, Text S1 S4) |
| `19_optimal_reserving.py` | Mack-lognormal predictive distribution, confidence-level reserves, CoC risk margin, newsvendor reserves; paid and incurred bases (Text S1 S5) |
| `20_synthesis.py` | Consolidation and the eleven audit gates; SUMMARY.md |
| `21_revision_R1.py` | Round-1 revision numbers: descriptive table, VIFs, TOST, MDE |
| `22_revision_R2.py` | Incurred triangle and case-reserve run-off (Table 9) |
| `23_independent_verification.py` | Independent re-derivation of the 32 headline results (Text S1 S6, Table S5) |
| `24_b2_decomposition.py` | Permutation decomposition of the naive measures (Table 3, Figure 3) |
| `config.py` | Single configuration file: paths, seeds, grids, thresholds, splits |
| `run_pipeline.py` | Orchestrator: python run_pipeline.py runs 00-24 in dependency order |

## Data S1 (`data_s1/`)

Data S1 is the minimal dataset published with the article. It contains, for every table and figure, the aggregate numbers as produced by the modules above (file names carry the module number, e.g. `06_dependence_summary.csv`, `11_copulas.csv`, `23_verification_table.csv`), together with its own `README.md` (file-by-file description, aggregation rules and the minimum-cell-size policy that protects policyholder privacy) and `FILES.txt`. The files are the direct outputs of the final run and can be compared one-to-one with the code paths that wrote them. Because the pipeline wrote them, the CSV headers and a few categorical cell values that originate in the insurer's files are in Turkish (the raw column names, e.g. `Ödenen_Hasar_TL`, are identifiers of the source data and were deliberately left unchanged); Data S1 `README.md` translates them.

## Language note

Docstrings, comments, log messages, error messages, console output, the exploratory figure labels of module 04 and the `SUMMARY.md` digest template are in English. Three kinds of strings were intentionally kept in their original Turkish form so that the outputs of this edition are byte-identical to those behind the published article: (i) the raw column and sheet names of the insurer's files in `config.py`; (ii) a small number of categorical cell values and free-text `note`/`interpretation` fields that modules write into their CSV outputs and JSON manifests; and (iii) the regular expressions of module 21 that parse the cleaning logs, which accept both wordings. The English edition was derived from the original scripts under an AST-based equivalence check: once docstrings and message strings are blanked, every module is syntactically identical to the original.

## Citing

Please cite the article (above) and, for the code itself, the archived release:

> Arslan, Y.; Karaman, F. (2026). Analysis code for "Frequency–Severity Dependence Diagnostics in Motor Casco Insurance" (v1.0-mathematics-4559487). Zenodo. https://doi.org/10.5281/zenodo.23894833

Machine-readable metadata are in `CITATION.cff`.

## Licence and contact

Code: MIT License (see `LICENSE`), © 2026 Yusuf Arslan and Filiz Karaman. Data S1 is published with the article under the article's licence (CC BY 4.0).

Corresponding author: Yusuf Arslan, Department of Statistics, Yıldız Technical University, Istanbul — yusuf.arslan1@std.yildiz.edu.tr
