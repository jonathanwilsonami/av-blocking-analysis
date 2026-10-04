"""Per-incident consequence (impact) scoring on the 5-level risk-matrix scale.

Each incident gets the *maximum* of several component scores, so one serious
attribute (e.g. a delayed ambulance) is never averaged away by benign ones.
The rubric is ordinal and judgement-based; it is written down here so it can
be argued with, and :func:`rubric_table` renders it for the paper.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

IMPACT_LABELS = ["Insignificant", "Minor", "Significant", "Major", "Severe"]

RUBRIC = [
    (1, "Insignificant", "General-traffic obstruction cleared in under 10 minutes."),
    (
        2,
        "Minor",
        "General-traffic obstruction of 10-60 minutes; 2-4 AVs clustered; "
        "non-medical occupant issue (asleep rider, door left open).",
    ),
    (
        3,
        "Significant",
        "Obstruction lasting 60 minutes or more; public-transit or rail "
        "obstruction; crosswalk / curb-ramp / bike-lane obstruction; 5+ AVs "
        "gridlocked; occupant medical-welfare check.",
    ),
    (
        4,
        "Major",
        "Obstruction of emergency responders or an active emergency scene; "
        "AV-involved collision with no reported injury.",
    ),
    (
        5,
        "Severe",
        "Emergency response obstructed during a life-threatening incident "
        "(unconscious patient, fire, persons trapped, shots fired), or an "
        "AV-involved collision with reported injury.",
    ),
]

DURATION_EDGES = (10, 60)  # minutes: <10 -> 1, 10-60 -> 2, >=60 -> 3

_LIFE_THREAT = re.compile(r"unconc?s?cious|trapped|\bfire\b|shots", re.I)
_MEDICAL = re.compile(r"unconc?s?cious|medic|911", re.I)
_INJURY = re.compile(r"\binj(?:ury|uries|ured)?\b", re.I)
_NO_INJURY = re.compile(r"\b(?:no|non)[ -]?(?:reported )?inj", re.I)


def duration_score(minutes: float, edges: tuple[int, int] = DURATION_EDGES) -> int:
    if minutes is None or np.isnan(minutes):
        return 1
    lo, hi = edges
    return 1 if minutes < lo else 2 if minutes < hi else 3


def cluster_score(n_avs: int) -> int:
    return 3 if n_avs >= 5 else 2 if n_avs >= 2 else 1


def hazard_score(hazard: str, text) -> int:
    t = "" if text is None or pd.isna(text) else str(text)
    if hazard == "H1":
        return 5 if _INJURY.search(t) and not _NO_INJURY.search(t) else 4
    if hazard == "H2":
        return 5 if _LIFE_THREAT.search(t) else 4
    if hazard in ("H3", "H4"):
        return 3
    if hazard == "H5":
        return 3 if _MEDICAL.search(t) else 2
    return 1


def score_incidents(
    df: pd.DataFrame, duration_edges: tuple[int, int] = DURATION_EDGES
) -> pd.DataFrame:
    """Add component scores, ``impact`` (1-5), ``impact_label`` and the driving component.

    ``duration_edges`` = (Minor, Significant) thresholds in minutes; varied only in the
    sensitivity analysis.
    """
    out = df.copy()
    out["impact_duration"] = out["duration_min"].map(lambda m: duration_score(m, duration_edges))
    out["impact_cluster"] = out["n_avs_filled"].map(cluster_score)
    out["impact_hazard"] = [
        hazard_score(h, t) for h, t in zip(out["hazard"], out["narrative"], strict=True)
    ]
    comps = out[["impact_hazard", "impact_duration", "impact_cluster"]]
    out["impact"] = comps.max(axis=1).astype(int)
    out["impact_driver"] = comps.idxmax(axis=1).str.replace("impact_", "", regex=False)
    out["impact_label"] = out["impact"].map(lambda k: IMPACT_LABELS[k - 1])
    return out


def rubric_table() -> pd.DataFrame:
    return pd.DataFrame(RUBRIC, columns=["Level", "Impact", "Criteria (any one suffices)"])
