"""Trend and distribution tests chosen for small, skewed, non-normal samples.

Counts (incidents per covered month)
    * Poisson dispersion (chi-square homogeneity) test of equal daily rates.
    * Mann-Kendall trend test with an *exact* permutation p-value (n is tiny).
    * Exact conditional (binomial) comparison of two periods' rates.
    * Poisson and negative-binomial GLMs with a log(days) offset, as a
      parametric cross-check that also yields an interpretable monthly rate ratio.

Durations (minutes per incident; heavily right-skewed)
    * Shapiro-Wilk on raw and log durations to document non-normality.
    * Kruskal-Wallis across months; Mann-Whitney U between two months.
    * Jonckheere-Terpstra for an *ordered* (monotone) shift across months.
    * Spearman correlation with date and Theil-Sen slope of log-duration.

Composition (hazard mix, multi-AV share)
    * Monte-Carlo permutation chi-square test of hazard x period.
    * Cochran-Armitage test for a monotone trend in a proportion.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from . import config

RNG = np.random.default_rng(config.RANDOM_SEED)


# --- monthly counts --------------------------------------------------------------
def monthly_counts(df: pd.DataFrame, days: pd.Series, group: str | None = None) -> pd.DataFrame:
    """Counts per covered month (zeros kept), with days and month index for trend tests."""
    months = list(days.index)
    sub = df[df["month"].isin(months)]
    if group is None:
        counts = sub.groupby("month").size().reindex(months, fill_value=0).to_frame("n")
    else:
        counts = (
            sub.groupby(["month", group]).size().unstack(fill_value=0).reindex(months, fill_value=0)
        )
    counts.insert(0, "days", days.values)
    counts.insert(1, "t", [pd.Period(m).month for m in months])  # keeps the Apr-May gap
    return counts


def dispersion_test(n: np.ndarray, days: np.ndarray) -> dict:
    """Chi-square test that all months share one daily rate (Poisson homogeneity)."""
    n, days = np.asarray(n, float), np.asarray(days, float)
    expected = n.sum() * days / days.sum()
    chi2 = float(((n - expected) ** 2 / expected).sum())
    dof = len(n) - 1
    return {
        "test": "Poisson homogeneity (chi-square)",
        "statistic": chi2,
        "df": dof,
        "p_value": float(stats.chi2.sf(chi2, dof)),
    }


def _mk_s(x: np.ndarray) -> float:
    return float(sum(np.sign(x[j] - x[i]) for i in range(len(x)) for j in range(i + 1, len(x))))


def mann_kendall(x, exact_max_n: int = 9) -> dict:
    """Mann-Kendall S, Kendall's tau, normal-approx p (tie-corrected), exact permutation p."""
    x = np.asarray(x, float)
    n = len(x)
    s = _mk_s(x)
    _, tie_counts = np.unique(x, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - sum(t * (t - 1) * (2 * t + 5) for t in tie_counts)) / 18
    z = 0.0 if s == 0 else (s - np.sign(s)) / math.sqrt(var)
    out = {
        "test": "Mann-Kendall",
        "n": n,
        "S": s,
        "tau": s / (n * (n - 1) / 2),
        "z": z,
        "p_normal": float(2 * stats.norm.sf(abs(z))),
    }
    if n <= exact_max_n:
        perm = np.array([_mk_s(np.array(p)) for p in itertools.permutations(x)])
        out["p_exact"] = float(np.mean(np.abs(perm) >= abs(s) - 1e-9))
    return out


def sen_slope(y, t) -> float:
    y, t = np.asarray(y, float), np.asarray(t, float)
    slopes = [
        (y[j] - y[i]) / (t[j] - t[i])
        for i in range(len(y))
        for j in range(i + 1, len(y))
        if t[j] != t[i]
    ]
    return float(np.median(slopes))


def exact_rate_comparison(n1: int, t1: float, n2: int, t2: float) -> dict:
    """Conditional binomial test of H0: rate1 == rate2 (exact, no normality)."""
    res = stats.binomtest(n2, n1 + n2, t2 / (t1 + t2))
    ci = res.proportion_ci(method="exact")

    def to_ratio(p):
        return (p / (1 - p)) * (t1 / t2) if p < 1 else np.inf

    return {
        "test": "Exact conditional rate comparison",
        "n1": n1,
        "n2": n2,
        "rate_ratio": (n2 / t2) / (n1 / t1),
        "rr_ci_lo": to_ratio(ci.low),
        "rr_ci_hi": to_ratio(ci.high),
        "p_value": float(res.pvalue),
    }


def count_glms(counts: pd.DataFrame) -> pd.DataFrame:
    """Poisson and NB2 log-linear trend in month with log(days) offset."""
    X = sm.add_constant(counts["t"].astype(float))
    offset = np.log(counts["days"].astype(float))
    rows = []
    pois = sm.GLM(counts["n"], X, family=sm.families.Poisson(), offset=offset).fit()
    disp = float(pois.pearson_chi2 / pois.df_resid)
    for name, fit in [("Poisson", pois)]:
        rows.append(_glm_row(name, fit, disp))
    try:
        nb = sm.NegativeBinomial(counts["n"], X, offset=offset).fit(disp=0)
        b, se = nb.params["t"], nb.bse["t"]
        rows.append(
            {
                "model": "Negative binomial (NB2)",
                "rate_ratio_per_month": math.exp(b),
                "ci_lo": math.exp(b - 1.96 * se),
                "ci_hi": math.exp(b + 1.96 * se),
                "p_value": float(nb.pvalues["t"]),
                "pearson_dispersion": np.nan,
                "alpha": float(nb.params["alpha"]),
            }
        )
    except Exception:  # NB fails to converge when there is no overdispersion
        pass
    return pd.DataFrame(rows)


def _glm_row(name, fit, disp) -> dict:
    b, se = fit.params["t"], fit.bse["t"]
    return {
        "model": name,
        "rate_ratio_per_month": math.exp(b),
        "ci_lo": math.exp(b - 1.96 * se),
        "ci_hi": math.exp(b + 1.96 * se),
        "p_value": float(fit.pvalues["t"]),
        "pearson_dispersion": disp,
        "alpha": np.nan,
    }


# --- durations -------------------------------------------------------------------
def normality(x) -> pd.DataFrame:
    x = np.asarray(pd.Series(x).dropna(), float)
    rows = []
    for label, v in [("raw minutes", x), ("log minutes", np.log(x))]:
        w, p = stats.shapiro(v)
        rows.append(
            {
                "scale": label,
                "n": len(v),
                "skewness": stats.skew(v),
                "excess_kurtosis": stats.kurtosis(v),
                "shapiro_W": w,
                "p_value": p,
            }
        )
    return pd.DataFrame(rows)


def kruskal_by(df: pd.DataFrame, value: str, group: str, order: list) -> dict:
    samples = [df.loc[df[group] == g, value].dropna().values for g in order]
    h, p = stats.kruskal(*samples)
    n = sum(len(s) for s in samples)
    eps2 = (h - len(samples) + 1) / (n - len(samples))  # epsilon-squared effect size
    return {
        "test": "Kruskal-Wallis",
        "H": float(h),
        "df": len(samples) - 1,
        "p_value": float(p),
        "epsilon_sq": float(eps2),
    }


def mann_whitney(a, b) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    res = stats.mannwhitneyu(a, b, alternative="two-sided", method="auto")
    u = float(res.statistic)
    # W (rank-sum of sample a) as reported by R's wilcox.test is U for sample a.
    return {
        "test": "Mann-Whitney U",
        "n1": len(a),
        "n2": len(b),
        "U": u,
        "p_value": float(res.pvalue),
        "rank_biserial": 1 - 2 * u / (len(a) * len(b)),
        "prob_superiority": u / (len(a) * len(b)),
    }


def _jt_stat(samples: list[np.ndarray]) -> float:
    total = 0.0
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            a, b = samples[i][:, None], samples[j][None, :]
            total += (a < b).sum() + 0.5 * (a == b).sum()
    return float(total)


def jonckheere_terpstra(samples: list, n_perm: int = 20000, rng=RNG) -> dict:
    """JT statistic for increasing order; permutation p-value (two-sided)."""
    samples = [np.asarray(s, float) for s in samples]
    obs = _jt_stat(samples)
    sizes = np.cumsum([len(s) for s in samples])[:-1]
    pooled = np.concatenate(samples)
    ns = np.array([len(s) for s in samples])
    n = ns.sum()
    mean = (n**2 - (ns**2).sum()) / 4
    perm = np.empty(n_perm)
    for k in range(n_perm):
        perm[k] = _jt_stat(np.split(rng.permutation(pooled), sizes))
    p = (np.sum(np.abs(perm - mean) >= abs(obs - mean)) + 1) / (n_perm + 1)
    return {
        "test": "Jonckheere-Terpstra",
        "JT": obs,
        "expected": float(mean),
        "z": float((obs - mean) / perm.std()),
        "p_value": float(p),
    }


def duration_vs_time(df: pd.DataFrame) -> dict:
    d = df.dropna(subset=["duration_min"])
    days = (d["date"] - d["date"].min()).dt.days.values.astype(float)
    rho, p = stats.spearmanr(days, d["duration_min"])
    ts = stats.theilslopes(np.log(d["duration_min"].values), days)
    return {
        "test": "Spearman (duration vs date)",
        "rho": float(rho),
        "p_value": float(p),
        "theil_sen_pct_per_30d": float(np.expm1(ts.slope * 30) * 100),
        "ts_lo_pct": float(np.expm1(ts.low_slope * 30) * 100),
        "ts_hi_pct": float(np.expm1(ts.high_slope * 30) * 100),
    }


def bootstrap_median_ci(x, n_boot: int = 10000, level: float = 0.95, rng=RNG):
    x = np.asarray(pd.Series(x).dropna(), float)
    meds = np.median(rng.choice(x, size=(n_boot, len(x)), replace=True), axis=1)
    a = (1 - level) / 2
    return float(np.median(x)), float(np.quantile(meds, a)), float(np.quantile(meds, 1 - a))


# --- composition -----------------------------------------------------------------
def permutation_chi2(labels, groups, n_perm: int = 20000, rng=RNG) -> dict:
    """Chi-square test of independence with a Monte-Carlo permutation p-value
    (valid with the sparse expected counts that make the asymptotic test unreliable)."""
    labels, groups = np.asarray(labels), np.asarray(groups)

    def chi2_of(g):
        tab = pd.crosstab(labels, g).values
        exp = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum()
        return float(((tab - exp) ** 2 / exp).sum())

    obs = chi2_of(groups)
    perm = np.array([chi2_of(rng.permutation(groups)) for _ in range(n_perm)])
    tab = pd.crosstab(labels, groups)
    exp = tab.sum(1).values[:, None] * tab.sum(0).values[None, :] / tab.values.sum()
    return {
        "test": "Permutation chi-square",
        "chi2": obs,
        "p_value": float((np.sum(perm >= obs - 1e-12) + 1) / (n_perm + 1)),
        "share_expected_lt5": float((exp < 5).mean()),
    }


def cochran_armitage(successes, totals, scores) -> dict:
    """Cochran-Armitage test for a linear trend in proportions (two-sided)."""
    x, n, s = (np.asarray(v, float) for v in (successes, totals, scores))
    p = x.sum() / n.sum()
    t = (s * (x - n * p)).sum()
    var = p * (1 - p) * ((n * s**2).sum() - (n * s).sum() ** 2 / n.sum())
    z = t / math.sqrt(var)
    return {"test": "Cochran-Armitage", "z": float(z), "p_value": float(2 * stats.norm.sf(abs(z)))}
