"""
================================================================================
SCRIPT 10 — Marginal Distributions (Severity and Frequency)
================================================================================
PURPOSE
    Estimates the marginals for the copula and the compound capital simulation:
    per-claim severity (body + POT-GPD tail) and policy frequency. The marginals
    are estimated SEPARATELY from the copula (IFM approach). marginal_fits.pkl is
    saved with a consistent schema for downstream use: frequency{name,params},
    severity{name,params}, gpd{xi,beta,threshold}.
INPUT : data/proc/{claim_analytical,analytical_dataset}.parquet
OUTPUT: data/proc/marginal_fits.pkl ; outputs/10_marginals.csv ; figures/10_*.png
================================================================================
"""
from __future__ import annotations
import sys, json, logging, pickle
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
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import stats
from config import (DATA_PROC_DIR, OUTPUTS_DIR, FIGURES_DIR, LOGS_DIR,
                    GLOBAL_SEED, FIG_DPI, GOF_DIST_CAP, PICKLE_PROTOCOL,
                    GPD_THRESHOLD_Q, GPD_THRESHOLD_Q_GRID, GPD_MIN_EXCEED,
                    FREQ_FIT_METHOD, GPD_XI_BOOT, GPD_SENS_MC, SOLVENCY_ALPHA,
                    FIG_W_FULL, FIG_DPI_PUB, apply_pub_style,
                    thousands_formatter, env_info)

SCRIPT_NAME = "10_marginals"
# The GPD threshold is selected data-driven; the frequency fit method is read from config

# --- R1 revision constants ---------------------------------------------------
# FIGURE design only (Figure 4 = 10_marginals, Figure 5 = 10_gpd_threshold).
# Estimates, the RNG stream and the pkl/CSV outputs are NOT AFFECTED by this block.
FIG4_H = 2.5                  # Figure 4 height (inches), width FIG_W_FULL
FIG5_H = 2.5                  # Figure 5 height (inches), width FIG_W_FULL
FIG4_SEV_TRUNC_Q = 0.99       # (a) histogram/pdf plotting range: [0, P99] (visual only)
FIG4_SEV_BINS = 80            # (a) number of equal-width histogram bins
FIG4_DENS_PER = 1000.0        # (a) density unit: per TRY 1,000 (no offset)
FIG_TRY_TICK = 10_000         # major tick spacing on TRY axes (avoids crowding)
FIG4_COUNT_POOL_K = 7         # (b) "≥7" pooled last category (small-cell
                              #     confidentiality: every bar ≥ 10 policies)
FIG5_ME_Q = (0.50, 0.99, 40)  # (b) mean-excess threshold grid: quantiles (lo, hi, n)
FIG5_ME_Z = 1.959963984540054 # (b) pointwise 95% band: ±z·sd(excess)/√n_u
C_DATA = "#9ECAE1"; C_DATA_EDGE = "#6BAED6"; C_FIT = "#B2182B"
C_GRID = "#2C7FB8"; C_REF = "#737373"; C_ME = "#238B45"

# --- Data S1 exports (R1 final): CSV of the plotted AGGREGATE data ----------
# Writes NEW files only; estimates, the RNG stream, the pkl, 10_marginals.csv and
# the figures are NOT AFFECTED by this block. Confidentiality (statistical
# disclosure control): no cell identifies fewer than 10 policies/claims.
DS1_MIN_CELL = 10             # smallest publishable cell (policies/claims)
DS1_FIG4B_TOPCODE = 7         # (b) OBSERVED counts are released exactly for the
                              #     categories in the figure (0..6 and ≥7; every
                              #     cell ≥ 10 policies); model probabilities (NB,
                              #     Poisson) are given for every category in the figure


_DS1_INFO = {}                # export metadata (for the manifest)


def _ds1_write(df, name):
    """Data S1 export: outputs/<name> (new file only)."""
    fp = OUTPUTS_DIR / name
    df.to_csv(fp, index=False)
    return str(fp.resolve())


def _merge_small_bins(edges, cnt, min_cell=DS1_MIN_CELL):
    """Merges neighbouring bins until every bin contains ≥ min_cell claims
    (right to left; for the export only, the figure is UNCHANGED)."""
    e = list(edges); c = list(cnt)
    i = len(c) - 1
    while i >= 0 and len(c) > 1:
        if c[i] < min_cell:
            j = i - 1 if i > 0 else i + 1
            lo, hi = min(i, j), max(i, j)
            c[lo] = c[lo] + c[hi]; del c[hi]; del e[hi]
            i = min(i, len(c) - 1)
            continue
        i -= 1
    return np.asarray(e), np.asarray(c)


def _save_fig(fig, stem):
    """PDF (vector) + PNG (600 dpi), same stem."""
    out = []
    for ext, kw in ((".pdf", {}), (".png", {"dpi": FIG_DPI_PUB})):
        fp = FIGURES_DIR / f"{stem}{ext}"
        fig.savefig(fp, **kw); out.append(str(fp.resolve()))
    plt.close(fig)
    return out


def draw_gpd_threshold_figure(sev, gpd, xi_lo, xi_hi):
    """Figure 5 — (a) threshold stability of ξ + exceedance-bootstrap 95% CI at the
    selected threshold; (b) empirical mean excess e(u)=E[Y−u|Y>u] (95% pointwise
    band) + selected threshold + linear e(u) implied by the fitted GPD (u ≥ threshold).
    Estimates are UNCHANGED."""
    tf = thousands_formatter()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(FIG_W_FULL, FIG5_H),
                                   layout="constrained")
    grid = gpd.get("grid", [])
    qs = [t["q"] for t in grid]; xis = [t["xi"] for t in grid]
    axL.plot(qs, xis, "o-", color=C_GRID, ms=4, label="Estimate on candidate grid")
    axL.errorbar([gpd["q"]], [gpd["xi"]],
                 yerr=[[gpd["xi"] - xi_lo], [xi_hi - gpd["xi"]]],
                 fmt="s", ms=5, color=C_FIT, capsize=3, lw=1.2, zorder=5,
                 label=f"Selected, q = {gpd['q']:.3f} (95% bootstrap CI)")
    # Two-line ticks: quantile q and (in parentheses) the real TRY threshold u —
    # direct link to the x axis of panel (b); no top axis (alignment)
    axL.set_xticks(qs)
    axL.set_xticklabels([(f"{t['q']:.3f}" if round(t["q"] * 100, 6) % 1
                          else f"{t['q']:.2f}") + f"\n({tf(t['threshold'], None)})"
                         for t in grid])        # 0.85 (6,816), 0.925 (12,500), ...
    axL.set_xlim(min(qs) - 0.012, max(qs) + 0.012)
    axL.set_xlabel("Threshold quantile q (threshold u, real TRY)")
    axL.set_ylabel("GPD shape parameter ξ")
    axL.set_title("(a) Threshold stability of ξ", loc="left")
    axL.legend(loc="upper right")

    lo, hi, n = FIG5_ME_Q
    uq = np.quantile(sev, np.linspace(lo, hi, n))
    me = np.empty(len(uq)); se = np.empty(len(uq))
    for i, u0 in enumerate(uq):
        e = sev[sev > u0] - u0
        me[i] = e.mean(); se[i] = e.std(ddof=1) / np.sqrt(len(e))
    axR.fill_between(uq, me - FIG5_ME_Z * se, me + FIG5_ME_Z * se,
                     color=C_ME, alpha=0.18, lw=0, label="95% pointwise band")
    axR.plot(uq, me, "-", color=C_ME, lw=1.3, label="Empirical mean excess")
    u_sel, xi_s, b_s = gpd["threshold"], gpd["xi"], gpd["beta"]
    uu = np.linspace(u_sel, uq.max(), 100)
    axR.plot(uu, (b_s + xi_s * (uu - u_sel)) / (1 - xi_s), ls=(0, (4, 2)),
             color=C_REF, lw=1.1, label="Implied by fitted GPD")
    axR.axvline(u_sel, color=C_FIT, ls="--", lw=0.9,
                label=f"Selected threshold u = {u_sel:,.0f}")
    axR.xaxis.set_major_locator(MultipleLocator(FIG_TRY_TICK))
    axR.xaxis.set_major_formatter(tf); axR.yaxis.set_major_formatter(tf)
    axR.set_xlim(0, uq.max() * 1.02)
    axR.set_xlabel("Threshold u (real TRY)")
    axR.set_ylabel("E[Y − u | Y > u] (real TRY)")
    axR.set_title("(b) Mean-excess plot", loc="left")
    axR.legend(loc="lower right")
    fig.align_xlabels([axL, axR])
    out = _save_fig(fig, "10_gpd_threshold")

    # Data S1: panel (b) — empirical mean-excess curve, band and GPD line
    n_exc = np.array([int(np.sum(sev > u0)) for u0 in uq])
    assert n_exc.min() >= DS1_MIN_CELL, "mean-excess: fewer than DS1_MIN_CELL exceedances"
    _DS1_INFO["fig5b_min_exceedances"] = int(n_exc.min())
    implied = np.where(uq >= u_sel, (b_s + xi_s * (uq - u_sel)) / (1 - xi_s),
                       np.nan)
    me_df = pd.DataFrame({
        "threshold_quantile_q": np.linspace(lo, hi, n),
        "threshold_u_real_TRY": uq,
        "n_exceedances": n_exc,
        "mean_excess_real_TRY": me,
        "se_mean_excess": se,
        "band95_low": me - FIG5_ME_Z * se,
        "band95_high": me + FIG5_ME_Z * se,
        "gpd_implied_mean_excess_real_TRY": implied})
    out.append(_ds1_write(me_df, "10_fig5b_mean_excess.csv"))
    return out


def draw_marginals_figure(sev, sev_name, sev_params, gpd, nc, freq):
    """Figure 4 — (a) real severity histogram [0, P99] + AIC-selected body
    density (histogram normalised by the FULL sample: same scale as the pdf);
    (b) distribution of the number of claims per policy vs NB (MLE) [+ Poisson
    reference], log-y. Estimates are UNCHANGED."""
    tf = thousands_formatter()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W_FULL, FIG4_H),
                                   layout="constrained")
    p_tr = float(np.quantile(sev, FIG4_SEV_TRUNC_Q))
    edges = np.linspace(0.0, p_tr, FIG4_SEV_BINS + 1)
    cnt, _ = np.histogram(sev[sev < p_tr], bins=edges)
    dens = cnt / (len(sev) * np.diff(edges)) * FIG4_DENS_PER
    ax1.stairs(dens, edges, fill=True, color=C_DATA, alpha=0.85,
               label="Observed")
    dist = {"LogNormal": stats.lognorm, "Gamma": stats.gamma,
            "Weibull": stats.weibull_min}[sev_name]
    xx = np.unique(np.concatenate([np.geomspace(1.0, p_tr, 600),
                                   np.linspace(0.0, p_tr, 400)[1:]]))
    ax1.plot(xx, dist.pdf(xx, *sev_params) * FIG4_DENS_PER, "-", color=C_FIT,
             lw=1.2, label={"LogNormal": "Lognormal", "Gamma": "Gamma",
                            "Weibull": "Weibull"}[sev_name] + " body (ML)")
    ax1.axvline(gpd["threshold"], color=C_REF, ls="--", lw=0.9,
                label=f"GPD threshold (q = {gpd['q']:.3f})")
    ax1.xaxis.set_major_locator(MultipleLocator(FIG_TRY_TICK))
    ax1.xaxis.set_major_formatter(tf)
    ax1.set_xlim(0, p_tr); ax1.set_ylim(bottom=0)
    ax1.set_xlabel("Severity (real TRY), up to the 99th percentile")
    ax1.set_ylabel("Density (per TRY 1,000)")
    ax1.set_title("(a) Severity body", loc="left")
    ax1.legend(loc="upper right")

    K = FIG4_COUNT_POOL_K; ks_ = np.arange(0, K + 1)
    obs = np.array([(nc == k).mean() for k in ks_[:-1]] + [(nc >= K).mean()])
    if freq["name"] == "NegativeBinomial":
        r, p = freq["params"]
        fit_pmf = np.r_[stats.nbinom.pmf(ks_[:-1], r, p), stats.nbinom.sf(K - 1, r, p)]
        fit_lab = "Negative binomial fit"
    else:
        fit_pmf = np.r_[stats.poisson.pmf(ks_[:-1], freq["params"][0]),
                        stats.poisson.sf(K - 1, freq["params"][0])]
        fit_lab = "Poisson fit"
    mu = float(nc.mean())
    poi = np.r_[stats.poisson.pmf(ks_[:-1], mu), stats.poisson.sf(K - 1, mu)]
    ax2.bar(ks_, obs, 0.62, color=C_DATA, edgecolor=C_DATA_EDGE, lw=0.4,
            label="Observed")
    ax2.plot(ks_, fit_pmf, "o-", color=C_FIT, ms=3.5, lw=1.0, label=fit_lab)
    if freq["name"] == "NegativeBinomial":
        ax2.plot(ks_, poi, "^", ls=(0, (3, 2)), color=C_REF, ms=3.5, lw=0.9,
                 mfc="white", label="Poisson fit (reference)")
    ax2.set_yscale("log")
    # lower limit: decade floor of the smallest probability; headroom for the
    # legend at the top (no tick label in the >1 region, 10^0 is the top label)
    ax2.set_ylim(10 ** np.floor(np.log10(min(obs.min(), fit_pmf.min(), poi.min()))), 4.0)
    ax2.set_xticks(ks_)
    ax2.set_xticklabels([str(k) for k in ks_[:-1]] + [f"≥{K}"])
    ax2.set_xlabel("Number of claims per policy")
    ax2.set_ylabel("Probability (log scale)")
    ax2.set_title("(b) Claim-count distribution", loc="left")
    ax2.legend(loc="upper right")
    out = _save_fig(fig, "10_marginals")

    # Data S1: panel (a) — histogram bins (same as the plotted density) +
    # value of the selected body density at the bin midpoint
    e_x, c_x = _merge_small_bins(edges, cnt)
    assert c_x.min() >= DS1_MIN_CELL
    _DS1_INFO["fig4a_bins_merged_for_disclosure"] = int(len(cnt) - len(c_x))
    _DS1_INFO["fig4a_min_bin_count"] = int(c_x.min())
    w_x = np.diff(e_x); mid = 0.5 * (e_x[:-1] + e_x[1:])
    h_df = pd.DataFrame({
        "bin": np.arange(1, len(c_x) + 1),
        "bin_left_real_TRY": e_x[:-1], "bin_right_real_TRY": e_x[1:],
        "n_claims": c_x.astype(int),
        "density_per_TRY1000": c_x / (len(sev) * w_x) * FIG4_DENS_PER,
        "fitted_body_density_per_TRY1000_at_bin_mid":
            dist.pdf(mid, *sev_params) * FIG4_DENS_PER})
    out.append(_ds1_write(h_df, "10_fig4a_severity_histogram.csv"))

    # Data S1: panel (b) — observed counts 0..T-1 and top-coded "≥T";
    # the model's plotted probabilities for every category in the figure (0..K-1, ≥K)
    T = DS1_FIG4B_TOPCODE
    n_pol = [int(np.sum(nc == k)) for k in range(T)] + [int(np.sum(nc >= T))]
    assert min(n_pol) >= DS1_MIN_CELL, "Figure 4b: fewer than DS1_MIN_CELL policies"
    _DS1_INFO["fig4b_min_cell_policies"] = int(min(n_pol))
    if freq["name"] == "NegativeBinomial":
        fit_T = stats.nbinom.sf(T - 1, *freq["params"])
    else:
        fit_T = stats.poisson.sf(T - 1, freq["params"][0])
    rows = []
    for k in range(T):                       # observed + model, same row
        rows.append({"category": str(k),
                     "n_policies": n_pol[k], "observed_share": n_pol[k] / len(nc),
                     "fit_prob": fit_pmf[k], "poisson_ref_prob": poi[k],
                     "plotted_in_figure": "observed bar and model points"})
    for k in (range(T, K + 1) if T < K else []):   # model only (observed is top-coded)
        rows.append({"category": (str(k) if k < K else f">={K}"),
                     "n_policies": np.nan,
                     "observed_share": np.nan, "fit_prob": fit_pmf[k],
                     "poisson_ref_prob": poi[k],
                     "plotted_in_figure": (f"model points; observed bar pooled "
                                           f"in row >={T}")})
    rows.append({"category": f">={T}",
                 "n_policies": n_pol[T], "observed_share": n_pol[T] / len(nc),
                 "fit_prob": fit_T,
                 "poisson_ref_prob": stats.poisson.sf(T - 1, mu),
                 "plotted_in_figure": ("observed bar and model points" if T == K else
                                       f"top-coded observed total of the bars "
                                       f"{T}..>={K} (disclosure control)")})
    c_df = pd.DataFrame(rows)
    c_df["n_policies"] = c_df["n_policies"].astype("Int64")
    c_df.insert(3, "fit_model", fit_lab)
    out.append(_ds1_write(c_df, "10_fig4b_claim_count_distribution.csv"))
    return out


def setup_logger():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(SCRIPT_NAME); lg.setLevel(logging.INFO); lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOGS_DIR / f"{SCRIPT_NAME}.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt); sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    lg.addHandler(fh); lg.addHandler(sh); return lg


log = logging.getLogger(SCRIPT_NAME)


def set_style():
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "savefig.dpi": FIG_DPI,
        "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25})


def load_proc(stem):
    pq = DATA_PROC_DIR / f"{stem}.parquet"
    return pd.read_parquet(pq) if pq.exists() else pd.read_pickle(DATA_PROC_DIR / f"{stem}.pkl")


def fit_severity(sev):
    cands = {"LogNormal": stats.lognorm, "Gamma": stats.gamma, "Weibull": stats.weibull_min}
    res = {}; rng = np.random.default_rng(GLOBAL_SEED)
    ks_s = sev if len(sev) <= GOF_DIST_CAP else rng.choice(sev, GOF_DIST_CAP, replace=False)
    for name, dist in cands.items():
        try:
            par = dist.fit(sev, floc=0)
            ll = np.sum(dist.logpdf(sev, *par)); aic = 2*len(par) - 2*ll
            ks = stats.kstest(ks_s, dist.cdf, args=par).statistic
            res[name] = {"params": list(par), "aic": float(aic), "ks": float(ks)}
        except Exception as e:
            log.warning(f"  {name} error: {e}")
    best = min(res, key=lambda n: res[n]["aic"])
    return best, res


def _fit_gpd_at(sev, u):
    """GPD exceedance fit at the given threshold + KS fit of the exceedances."""
    exc = sev[sev > u] - u
    xi, loc, beta = stats.genpareto.fit(exc, floc=0)
    ks = float(stats.kstest(exc, "genpareto", args=(xi, loc, beta)).statistic)
    return {"threshold": float(u), "xi": float(xi), "beta": float(beta),
            "n_exceed": int(len(exc)), "ks": ks}


def fit_gpd(sev, q_grid=GPD_THRESHOLD_Q_GRID, min_exceed=GPD_MIN_EXCEED,
            q_fallback=GPD_THRESHOLD_Q):
    """DATA-DRIVEN POT-GPD threshold (fixed-quantile hardcode → formula).

    The GPD is fitted to the exceedances on a candidate quantile grid; the
    threshold that MINIMISES the KS distance of the exceedances from the fitted
    GPD is selected (subject to at least `min_exceed` exceedances). Since all
    candidates are fitted in-sample in the same way, KS is a relative selection
    criterion (Pickands-type threshold-stability logic). The quantile of the
    selected threshold is stored in the `q` field; downstream (13_capital)
    aligns the body→GPD splice quantile with it."""
    trace = []
    for q in q_grid:
        u = float(np.quantile(sev, q))
        fit = _fit_gpd_at(sev, u)
        if fit["n_exceed"] < min_exceed:
            continue
        fit["q"] = float(q)
        trace.append(fit)
    if trace:
        best = min(trace, key=lambda d: d["ks"])
    else:  # backward-compatibility fixed quantile if the grid is empty
        u = float(np.quantile(sev, q_fallback))
        best = _fit_gpd_at(sev, u); best["q"] = float(q_fallback)
    best = dict(best)
    best["selection"] = "ks_min_over_grid"
    best["grid"] = trace
    return best


def _nb_mle(nc, mu):
    """Intercept-only NB2 MLE: μ fixed at the sample mean, the dispersion (r) is
    optimised by likelihood. nbinom(r, p), p=r/(r+μ)."""
    from scipy.optimize import minimize_scalar
    def negll(logr):
        r_ = np.exp(logr); p_ = r_ / (r_ + mu)
        return -np.sum(stats.nbinom.logpmf(nc, r_, p_))
    opt = minimize_scalar(negll, bounds=(np.log(1e-3), np.log(1e6)),
                          method="bounded")
    r = float(np.exp(opt.x))
    return r, float(r / (r + mu))


def fit_frequency(nc, method=FREQ_FIT_METHOD):
    """Frequency marginal. method='mle' → NB dispersion parameter by likelihood
    (optimises the dispersion directly); 'mom' → moments (p=μ/σ², r=μp/(1−p))."""
    mean = nc.mean(); var = nc.var(); disp = var/mean if mean > 0 else np.nan
    ll_p = np.sum(stats.poisson.logpmf(nc, mean)); aic_p = 2 - 2*ll_p
    if var > mean:  # overdispersion → NB meaningful
        if method == "mle":
            r, p = _nb_mle(nc, mean)
        else:
            p = mean/var; r = mean*p/(1-p)
        ll_nb = np.sum(stats.nbinom.logpmf(nc, r, p)); aic_nb = 4 - 2*ll_nb
    else:
        r, p, aic_nb = np.nan, np.nan, np.inf
    best = "NegativeBinomial" if aic_nb < aic_p else "Poisson"
    params = [float(r), float(p)] if best == "NegativeBinomial" else [float(mean)]
    return {"name": best, "params": params, "dispersion": float(disp),
            "aic_poisson": float(aic_p), "aic_negbin": float(aic_nb),
            "fit_method": method}


def main():
    global log
    log = setup_logger(); set_style(); apply_pub_style()
    log.info("=" * 78); log.info(f"SCRIPT {SCRIPT_NAME} — Marginal Distributions"); log.info("=" * 78)
    outfiles = []
    claim = load_proc("claim_analytical"); ana = load_proc("analytical_dataset")

    log.info("\n[1/3] Severity body")
    sev = claim["severity_real"].values; sev = sev[sev > 0]
    best_sev, sev_res = fit_severity(sev)
    log.info(f"  Best (AIC): {best_sev}, KS={sev_res[best_sev]['ks']:.4f}")
    for n, r in sev_res.items():
        log.info(f"    {n}: AIC={r['aic']:,.0f}, KS={r['ks']:.4f}")

    log.info("\n[2/3] Severity tail (POT-GPD, data-driven threshold)")
    gpd = fit_gpd(sev)
    log.info(f"  Selected threshold q={gpd['q']:.3f} (KS-min grid); threshold={gpd['threshold']:,.0f}, "
             f"ξ={gpd['xi']:.3f}, β={gpd['beta']:,.0f}, n_exceed={gpd['n_exceed']:,}, KS={gpd['ks']:.4f}")
    for t in gpd.get("grid", []):
        log.info(f"    q={t['q']:.3f}: ξ={t['xi']:+.3f}, n={t['n_exceed']:,}, KS={t['ks']:.4f}")

    log.info("\n[2b/3] GPD robustness analysis: ξ bootstrap CI + threshold sensitivity")
    rng = np.random.default_rng(GLOBAL_SEED)
    exc = sev[sev > gpd["threshold"]] - gpd["threshold"]
    xi_b = np.empty(GPD_XI_BOOT)
    for b in range(GPD_XI_BOOT):
        eb = rng.choice(exc, len(exc), replace=True)
        xi_b[b] = stats.genpareto.fit(eb, floc=0)[0]
    xi_lo, xi_hi = np.percentile(xi_b, [2.5, 97.5])
    gpd["xi_ci95"] = [float(xi_lo), float(xi_hi)]
    log.info(f"  ξ bootstrap 95% CI = [{xi_lo:.3f}, {xi_hi:.3f}] "
             f"(B={GPD_XI_BOOT} exceedance-bootstrap)")
    # Threshold sensitivity: under each candidate threshold, MC moments from the
    # body+GPD splice → compound-Poisson Cornish-Fisher SCR RATIO (reference annual volume Λ_ref)
    bp = tuple(sev_res[best_sev]["params"])
    lam_ref = len(sev) / 5.0                      # ~annual claim volume
    z995 = stats.norm.ppf(SOLVENCY_ALPHA)
    sens = []
    for t in gpd.get("grid", []):
        u01 = rng.uniform(size=GPD_SENS_MC)
        y = stats.lognorm.ppf(np.clip(u01, 1e-9, 1 - 1e-9), *bp)
        m_tail = u01 > t["q"]
        if m_tail.any():
            pe = (u01[m_tail] - t["q"]) / (1 - t["q"])
            y[m_tail] = t["threshold"] + stats.genpareto.ppf(
                np.clip(pe, 1e-9, 1 - 1e-9), t["xi"], scale=t["beta"])
        E1, E2, E3 = y.mean(), np.mean(y**2), np.mean(y**3)
        mu = lam_ref * E1; var = lam_ref * E2; m3 = lam_ref * E3
        g = m3 / var**1.5
        scr_pct = (np.sqrt(var) * (z995 + (z995**2 - 1) / 6 * g)) / mu * 100
        sens.append({"q": t["q"], "xi": t["xi"], "scr_pct_of_mean": float(scr_pct)})
        log.info(f"  q={t['q']:.3f}: ξ={t['xi']:.3f} → SCR proxy "
                 f"= {scr_pct:.2f}% × E[S]")
    gpd["threshold_sensitivity"] = sens
    if sens:
        rngv = [d["scr_pct_of_mean"] for d in sens]
        log.info(f"  → SCR proxy range across threshold choices: "
                 f"[{min(rngv):.2f}%, {max(rngv):.2f}%] (narrow → robust)")
    # Figure 5: threshold stability + mean excess (plotting only; uses no RNG)
    outfiles.extend(draw_gpd_threshold_figure(sev, gpd, xi_lo, xi_hi))

    log.info("\n[3/3] Frequency")
    nc = ana["n_claims"].values; freq = fit_frequency(nc)
    log.info(f"  Best: {freq['name']}, params={np.round(freq['params'],4)}, dispersion={freq['dispersion']:.3f}")

    bundle = {
        "severity": {"name": best_sev, "params": list(sev_res[best_sev]["params"]),
                     "ks": sev_res[best_sev]["ks"], "all": sev_res},
        "gpd": gpd,
        "frequency": {"name": freq["name"], "params": freq["params"], "dispersion": freq["dispersion"]},
        "n_claims": int(len(sev))}
    with open(DATA_PROC_DIR / "marginal_fits.pkl", "wb") as f:
        pickle.dump(bundle, f, protocol=PICKLE_PROTOCOL)
    outfiles.append(str((DATA_PROC_DIR / "marginal_fits.pkl").resolve()))

    summary = pd.DataFrame([
        {"bilesen": "severity", "dagilim": best_sev, "ks": sev_res[best_sev]["ks"]},
        {"bilesen": "gpd_tail", "dagilim": "GPD", "xi": gpd["xi"], "beta": gpd["beta"]},
        {"bilesen": "frequency", "dagilim": freq["name"], "dispersion": freq["dispersion"]}])
    summary.to_csv(OUTPUTS_DIR / "10_marginals.csv", index=False)
    outfiles.append(str((OUTPUTS_DIR / "10_marginals.csv").resolve()))

    # Figure 4: severity body + claim-count distribution (plotting only)
    outfiles.extend(draw_marginals_figure(sev, best_sev, sev_res[best_sev]["params"],
                                          gpd, nc, freq))

    # Tail-index terminology (the text uses both ξ and 1/ξ). NOT ADDED to the
    # gpd dict (so the pkl stays byte-identical) — separate key in the manifest only.
    tail_index = {
        "convention": ("xi = GPD shape parameter (plotted in Figure 5a, reported "
                       "as xi in text/tables); tail index alpha = 1/xi "
                       "(Pareto-type tail P(Y>y) = L(y) y^(-1/xi)); E[Y^k] < inf "
                       "iff k < 1/xi. The text's 'tail index of about 3.2' refers "
                       "to 1/xi, not to xi."),
        "xi_hat": float(gpd["xi"]), "xi_ci95": [float(xi_lo), float(xi_hi)],
        "tail_index_hat": float(1.0 / gpd["xi"]),
        "tail_index_ci95": [float(1.0 / xi_hi), float(1.0 / xi_lo)],
        "tail_index_ci95_method": ("monotone (decreasing) transform of the "
                                   "exceedance-bootstrap percentile CI for xi"),
        "finite_moment_orders": [k for k in (1, 2, 3, 4) if k < 1.0 / gpd["xi"]]}
    figure_design = {
        "style": "config.apply_pub_style(); width FIG_W_FULL; PDF + PNG@600dpi",
        "fig4_10_marginals": {
            "size_in": [float(FIG_W_FULL), FIG4_H],
            "a": (f"histogram of real severity on [0, P{int(FIG4_SEV_TRUNC_Q*100)}], "
                  f"{FIG4_SEV_BINS} equal-width bins, density normalised by the FULL "
                  f"sample size (comparable to the full-support pdf), shown per "
                  f"TRY {FIG4_DENS_PER:,.0f}; AIC-selected body pdf "
                  f"({best_sev}, ML); dashed line = selected GPD threshold"),
            "a_truncation_value_try": float(np.quantile(sev, FIG4_SEV_TRUNC_Q)),
            "b": (f"claim counts per policy 0..{FIG4_COUNT_POOL_K-1} and pooled "
                  f">={FIG4_COUNT_POOL_K} (small-cell pooling); fitted NB pmf/sf "
                  f"(ML); Poisson with the sample mean shown as reference; log y")},
        "fig5_10_gpd_threshold": {
            "size_in": [float(FIG_W_FULL), FIG5_H],
            "a": ("xi on candidate quantile grid (two-line tick labels: q and "
                  "threshold u in real TRY); selected q with exceedance-bootstrap 95% "
                  f"percentile CI (B={GPD_XI_BOOT})"),
            "b": (f"empirical mean excess at {FIG5_ME_Q[2]} thresholds = quantiles "
                  f"{FIG5_ME_Q[0]}..{FIG5_ME_Q[1]}; 95% pointwise band "
                  "+/-1.96 sd(excess)/sqrt(n_u); GPD-implied mean excess "
                  "(beta + xi (u - u*))/(1 - xi) for u >= u*; dashed line = u*")}}

    manifest = {"script": SCRIPT_NAME, "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "severity": {"best": best_sev, "ks": sev_res[best_sev]["ks"]}, "gpd": gpd, "frequency": freq,
        "tail_index_convention": tail_index, "figure_design": figure_design,
        "data_s1_exports": {
            "files": ["10_fig4a_severity_histogram.csv",
                      "10_fig4b_claim_count_distribution.csv",
                      "10_fig5b_mean_excess.csv"],
            "purpose": ("aggregate data plotted in Figure 4 (a, b) and "
                        "Figure 5 (b); written after the figures, no estimate, "
                        "seed or existing output affected"),
            "min_cell": DS1_MIN_CELL,
            **_DS1_INFO,
            "fig4b_observed_topcode": DS1_FIG4B_TOPCODE,
            "fig4b_note": ("observed counts released for exactly the plotted "
                           "categories N = 0..6 and N >= 7 (every cell >= "
                           "min_cell policies); model probabilities are released "
                           "for every plotted category")},
        "output_files": outfiles, "environment": env_info()}
    with open(OUTPUTS_DIR / "10_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log.info("\n" + "=" * 78); log.info("SCRIPT 10 COMPLETED"); log.info("=" * 78)


if __name__ == "__main__":
    main()
