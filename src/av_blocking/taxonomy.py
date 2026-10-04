"""Hazard taxonomy and the transparent rule-based narrative classifier.

The taxonomy has two axes, kept separate on purpose. Mixing "why the AV
stopped" with "what the stopped AV endangered" is the most common flaw in
ad-hoc incident coding.

* **Hazard** (consequence axis, one label per incident): what the immobilized
  AV obstructed or put at risk. Classes are mutually exclusive and assigned by
  a severity-first *precedence* rule: an incident that matches several classes
  takes the most severe one (H1 before H2 ... before H7).
* **Contributing cause** (mechanism axis, one label per incident): why the AV
  became immobilized, as far as the narrative says.

Rules are deliberately simple keyword/regex patterns over the DEM dispatcher
shorthand (``blkng``, ``ntfyd``, ``LP`` ...). With 123 short and often
truncated narratives and no gold-standard labels, an auditable rule set is
more defensible than a trained model; see :mod:`av_blocking.nlp` for the
comparison against learned classifiers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Hazard:
    code: str
    name: str
    short: str
    definition: str


HAZARDS: list[Hazard] = [
    Hazard(
        "H1",
        "AV-involved collision",
        "Collision",
        "The AV made (or was reported to make) contact with another road user, vehicle, "
        "or animal, leaving the AV or the scene blocking the roadway.",
    ),
    Hazard(
        "H2",
        "Emergency-response obstruction",
        "Emergency obstruction",
        "The AV obstructed an emergency response (medics, fire, police) or became stuck "
        "inside an active emergency scene (fire, crash scene, police investigation).",
    ),
    Hazard(
        "H3",
        "Transit / rail obstruction",
        "Transit obstruction",
        "The AV blocked a public-transit vehicle, lane, or track (Muni bus or rail, cable "
        "car) or a railroad track.",
    ),
    Hazard(
        "H4",
        "Pedestrian / accessibility obstruction",
        "Ped/accessibility",
        "The AV blocked a crosswalk, curb ramp, or bike lane, pushing vulnerable road "
        "users into traffic.",
    ),
    Hazard(
        "H5",
        "Occupant-related immobilization",
        "Occupant-related",
        "A passenger condition or action immobilized the AV: unresponsive/asleep rider, "
        "occupant trapped or calling for help, or a door left open.",
    ),
    Hazard(
        "H6",
        "Multi-AV clustering / gridlock",
        "Multi-AV gridlock",
        "Two or more AVs immobilized together at one location (a common-mode, fleet-level "
        "failure) with none of the above consequences.",
    ),
    Hazard(
        "H7",
        "Single-AV traffic obstruction",
        "Single-AV obstruction",
        "One AV stalled in a travel lane, intersection, or roadway, obstructing general "
        "traffic only. Default class.",
    ),
]
HAZARD_BY_CODE = {h.code: h for h in HAZARDS}
HAZARD_CODES = [h.code for h in HAZARDS]


CAUSES = {
    "collision": "AV collision / contact",
    "external_scene": "External event or scene (fire, crash, protest, police activity)",
    "infrastructure": "Infrastructure failure (signal or power outage)",
    "occupant": "Passenger condition or action",
    "vehicle_fault": "Vehicle fault (hardware, battery, tire, sensor damage)",
    "tampering": "Tampering / vandalism",
    "maneuver": "Maneuver or routing edge case (turn, parking lot, cul-de-sac, queue)",
    "unstated": "Unstated (stalled/stuck with no cause given)",
}


def _rx(*patterns: str) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)


_AV = r"(?:waymos?|av'?s?|zoox)"

RULES: dict[str, re.Pattern] = {
    "H1": _rx(
        rf"{_AV}\W+vs\.?\b",  # no leading \b: narratives contain typos like "SWaymo"
        rf"\bvs\.?\W+{_AV}\b",
        r"hit the door of (?:a |the )?waymo",
        rf"\b{_AV}\b[^.]{{0,15}}\bhit\b",
        r"involved in a[^.]{0,20}accident",
    ),
    "H2": _rx(
        r"\bmedics?\b",
        r"\bambulance",
        r"\bsffd",
        r"\bfire\b",
        r"\btrapped",
        r"shots",
        r"actively investigating",
        r"(?:veh(?:icle)?\.?|multi veh)[^.]{0,10}accident",
    ),
    "H3": _rx(
        r"\bmuni\b",
        r"cable car",
        r"\btrains?\b",
        r"\btrack\b",
        r"railroad",
        r"\b\d{1,2} line\b",
        r"bus lines?",
    ),
    "H4": _rx(
        r"handicap",
        # curb/accessibility ramps only; bare "ramp" in this log usually means a freeway ramp
        r"(?:curb|wheelchair|ada|accessib\w*) ramp",
        r"crosswalk",
        r"bike lane",
        r"pedestrian",
    ),
    "H5": _rx(
        r"passengers?",
        r"\brider\b",
        r"asleep",
        r"unconc?s?cious rider",
        r"door (?:left )?open|left the door open",
        r"subj inside",
        r"rp inside",
        r"pssngrs? ",
    ),
}

# Occupant terms that do NOT indicate an occupant hazard ("no driver unk if pssngrs").
_H5_NEGATIONS = _rx(r"unk if pss?ngrs?", r"no driver")

CAUSE_RULES: dict[str, re.Pattern] = {
    "external_scene": _rx(
        r"\bfire\b",
        r"protest",
        r"critical mass",
        r"shots",
        r"investigating",
        r"geo ?fence",
        r"accident",
        r"towing another",
    ),
    "infrastructure": _rx(
        r"signals?\b[^.]{0,20}(?:out|red|malfunction)",
        r"(?:stop |traffic )?lights are out",
        r"power outage",
        r"signal repair",
    ),
    "occupant": RULES["H5"],
    "vehicle_fault": _rx(
        r"disabl+ed",
        r"malfunctioning\b(?! ?on)(?!.*signal)",
        r"tech difficult",
        r"battery",
        r"\bflat\b",
        r"sensors? (?:are|is) damaged",
    ),
    "tampering": _rx(r"put something on the sensor", r"vandal"),
    "maneuver": _rx(
        r"trying to (?:make|enter)",
        r"cul de sac",
        r"alley",
        r"parking lot",
        r"stuck behind",
        r"double parked",
        r"changed lanes",
        r"through the stop light",
        r"right turn",
        r"cannot go",
        r"lined up",
        r"pulled up behind",
        r"tunnel",
        r"left lane poss trying",
    ),
}
_CAUSE_ORDER = [
    "collision",
    "tampering",
    "occupant",
    "infrastructure",
    "external_scene",
    "vehicle_fault",
    "maneuver",
]


def hazard_matches(text: str | None, n_avs: int = 1) -> dict[str, bool]:
    """Which hazard rules fire for one narrative (before precedence)."""
    t = "" if text is None or pd.isna(text) else str(text)
    hits = {code: bool(rx.search(t)) for code, rx in RULES.items()}
    if hits["H5"] and _H5_NEGATIONS.search(t) and not re.search(r"passenger|rider|asleep", t, re.I):
        hits["H5"] = False
    hits["H6"] = n_avs >= 2
    hits["H7"] = True
    return hits


def classify_hazard(text: str | None, n_avs: int = 1) -> str:
    hits = hazard_matches(text, n_avs)
    return next(code for code in HAZARD_CODES if hits[code])


def classify_cause(text: str | None, hazard: str) -> str:
    t = "" if text is None or pd.isna(text) else str(text)
    if hazard == "H1":
        return "collision"
    for cause in _CAUSE_ORDER[1:]:
        if CAUSE_RULES[cause].search(t):
            return cause
    return "unstated"


def apply_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``hazard``, ``hazard_name``, ``cause``, and per-rule match columns."""
    out = df.copy()
    matches = [
        hazard_matches(t, n) for t, n in zip(out["narrative"], out["n_avs_filled"], strict=True)
    ]
    for code in HAZARD_CODES[:-1]:
        out[f"match_{code}"] = [m[code] for m in matches]
    out["n_hazard_matches"] = out[[f"match_{c}" for c in HAZARD_CODES[:-1]]].sum(axis=1)
    out["hazard"] = [next(c for c in HAZARD_CODES if m[c]) for m in matches]
    out["hazard_name"] = out["hazard"].map(lambda c: HAZARD_BY_CODE[c].name)
    out["cause"] = [
        classify_cause(t, h) for t, h in zip(out["narrative"], out["hazard"], strict=True)
    ]
    out["cause_name"] = out["cause"].map(CAUSES)
    return out


def taxonomy_table() -> pd.DataFrame:
    return pd.DataFrame(
        [(h.code, h.name, h.definition) for h in HAZARDS],
        columns=["Code", "Hazard", "Definition"],
    )
