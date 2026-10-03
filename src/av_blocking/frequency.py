r"""Incident-frequency estimation and risk-matrix placement.

Model
-----
Incidents of a given kind are treated as a Poisson process with rate
:math:`\lambda` per day over the covered observation window of :math:`T` days.
With :math:`n` observed events:

* MLE: :math:`\hat\lambda = n / T`, with an exact (Garwood) confidence interval.
* Bayesian: Jeffreys prior :math:`\lambda \sim \mathrm{Gamma}(1/2, 0)` gives the
  posterior :math:`\mathrm{Gamma}(n + 1/2,\ T)`.
* Posterior predictive probability of at least one event in the next
  :math:`h` days (negative-binomial predictive):
  :math:`P(N_h \ge 1) = 1 - \left(\tfrac{T}{T+h}\right)^{n+1/2}`.

There is no exposure denominator (AV miles or trips) in the DEM log, so rates
are per calendar day of a covered month, not per mile. If fleet-mileage data
become available, pass them as ``exposure`` instead of days.

Risk-curve placement
--------------------
A hazard rarely has a single consequence. For hazard *h* and impact tier *k*,
let :math:`n_{h,\ge k}` count incidents of *h* whose impact score is at least
*k*. Each tier gives a candidate cell (probability band of
:math:`n_{h,\ge k}`, impact *k*). The hazard is placed in the candidate cell
with the highest risk level (ties go to the higher impact). Only tiers that
were actually observed are considered, so an unobserved worst case never
drives the placement. Probability and impact therefore always describe the
*same* event, which pairing the hazard's total frequency with its worst
consequence would not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .risk_matrix import LEVEL_RANK, PROB_LABELS, Placement, level_of, probability_band
from .severity import IMPACT_LABELS
from .taxonomy import HAZARD_BY_CODE, HAZARD_CODES

JEFFREYS_SHAPE = 0.5


def garwood_ci(n: int, exposure: float, level: float = 0.95) -> tuple[float, float]:
    a = 1 - level
    lo = 0.0 if n == 0 else stats.chi2.ppf(a / 2, 2 * n) / 2
    hi = stats.chi2.ppf(1 - a / 2, 2 * n + 2) / 2
    return lo / exposure, hi / exposure


def posterior(n: int, exposure: float):
    """Gamma posterior for the daily rate under the Jeffreys prior."""
    return stats.gamma(a=n + JEFFREYS_SHAPE, scale=1 / exposure)


def predictive_p_any(n: int, exposure: float, horizon: float = config.HORIZON_DAYS) -> float:
    """Posterior-predictive P(at least one event in the next ``horizon`` days)."""
    return 1 - (exposure / (exposure + horizon)) ** (n + JEFFREYS_SHAPE)


def mle_p_any(n: int, exposure: float, horizon: float = config.HORIZON_DAYS) -> float:
    return 1 - np.exp(-n / exposure * horizon)


def probability(
    n: int, exposure: float, basis: str = "bayesian", horizon: float = config.HORIZON_DAYS
) -> float:
    if basis == "bayesian":
        return predictive_p_any(n, exposure, horizon)
    if basis == "per_period":
        return mle_p_any(n, exposure, horizon)
    raise ValueError(f"unknown probability basis {basis!r}")


def rate_summary(n: int, exposure: float, horizon: float = config.HORIZON_DAYS) -> dict:
    lo, hi = garwood_ci(n, exposure)
    post = posterior(n, exposure)
    cr_lo, cr_hi = post.ppf([0.025, 0.975])
    p = predictive_p_any(n, exposure, horizon)
    return {
        "n": n,
        "rate_per_30d": n / exposure * 30,
        "ci_lo_30d": lo * 30,
        "ci_hi_30d": hi * 30,
        "post_mean_30d": post.mean() * 30,
        "cri_lo_30d": cr_lo * 30,
        "cri_hi_30d": cr_hi * 30,
        "mean_days_between": exposure / n if n else np.inf,
        "p_any_horizon": p,
        # Band under the credible-interval ends of the rate: how robust is the placement?
        "band_lo": probability_band(1 - np.exp(-cr_lo * horizon)),
        "band_hi": probability_band(1 - np.exp(-cr_hi * horizon)),
        "band": probability_band(p),
    }


def hazard_frequency_table(
    df: pd.DataFrame, exposure: float, horizon: float = config.HORIZON_DAYS
) -> pd.DataFrame:
    """Rate, uncertainty, and probability band for every hazard class."""
    rows = []
    for code in HAZARD_CODES:
        n = int((df["hazard"] == code).sum())
        rows.append(
            {
                "hazard": code,
                "name": HAZARD_BY_CODE[code].name,
                **rate_summary(n, exposure, horizon),
            }
        )
    total = rate_summary(len(df), exposure, horizon)
    rows.append({"hazard": "All", "name": "All blocking incidents", **total})
    out = pd.DataFrame(rows)
    out["share"] = out["n"] / len(df)
    out["probability"] = out["band"].map(lambda b: PROB_LABELS[b - 1])
    return out


def risk_curve(
    df: pd.DataFrame,
    hazard: str,
    exposure: float,
    basis: str = "bayesian",
    horizon: float = config.HORIZON_DAYS,
) -> pd.DataFrame:
    """Candidate (probability, impact) cells for each observed impact tier of a hazard."""
    sub = df[df["hazard"] == hazard]
    rows = []
    if sub.empty:
        return pd.DataFrame(columns=["hazard", "impact", "n_ge", "p_any", "prob", "level"])
    for k in range(1, int(sub["impact"].max()) + 1):
        n_ge = int((sub["impact"] >= k).sum())
        p = probability(n_ge, exposure, basis, horizon)
        prob = probability_band(p)
        rows.append(
            {
                "hazard": hazard,
                "impact": k,
                "n_ge": n_ge,
                "p_any": p,
                "prob": prob,
                "level": level_of(prob, k),
            }
        )
    return pd.DataFrame(rows)


def place_hazards(
    df: pd.DataFrame, exposure: float, basis: str = "bayesian", horizon: float = config.HORIZON_DAYS
) -> pd.DataFrame:
    """Risk-curve placement for every hazard that occurred at least once."""
    rows = []
    for code in HAZARD_CODES:
        curve = risk_curve(df, code, exposure, basis, horizon)
        if curve.empty:
            continue
        curve = curve.assign(rank=curve["level"].map(LEVEL_RANK))
        best = curve.sort_values(["rank", "impact"], ascending=False).iloc[0]
        sub = df[df["hazard"] == code]
        rows.append(
            {
                "hazard": code,
                "name": HAZARD_BY_CODE[code].name,
                "n": len(sub),
                "median_impact": int(np.median(sub["impact"])),
                "max_impact": int(sub["impact"].max()),
                "governing_impact": int(best["impact"]),
                "n_at_or_above": int(best["n_ge"]),
                "p_any_horizon": float(best["p_any"]),
                "prob": int(best["prob"]),
                "impact_label": IMPACT_LABELS[int(best["impact"]) - 1],
                "prob_label": PROB_LABELS[int(best["prob"]) - 1],
                "level": best["level"],
                "score": int(best["prob"] * best["impact"]),
            }
        )
    return pd.DataFrame(rows)


def to_placements(placed: pd.DataFrame) -> list[Placement]:
    return [Placement(r.hazard, r.prob, r.governing_impact) for r in placed.itertuples()]


def conventional_placement(
    df: pd.DataFrame,
    exposure: float,
    quantile: float = 0.9,
    basis: str = "bayesian",
    horizon: float = config.HORIZON_DAYS,
) -> pd.DataFrame:
    """Sensitivity check: total hazard frequency paired with a high-quantile impact."""
    rows = []
    for code in HAZARD_CODES:
        sub = df[df["hazard"] == code]
        if sub.empty:
            continue
        impact = int(np.quantile(sub["impact"], quantile, method="higher"))
        prob = probability_band(probability(len(sub), exposure, basis, horizon))
        rows.append(
            {"hazard": code, "prob": prob, "impact": impact, "level": level_of(prob, impact)}
        )
    return pd.DataFrame(rows)
