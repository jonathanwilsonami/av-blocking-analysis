"""Load, clean, and enrich the DEM AV-blocking incident log.

The primary source is the ``DEM Incidents`` sheet of ``AV-Brick-Report.xlsx``.
Two secondary sheets in ``AV-Brick-Report-analyzed.xlsx`` contribute fields the
primary sheet lacks:

* ``DEM Incidents`` (analyzed copy): ETA provided by the AV company, other
  parties involved, and emergency-responder / transit impact flags, plus a few
  incidents that have a narrative but no start/clear times.
* ``DEM Incidents Quant``: a hand-coded San Francisco neighborhood zone (1-10).
"""

from __future__ import annotations

import datetime as dt
import re

import numpy as np
import pandas as pd

from . import config

NEIGHBORHOODS = {
    1: "Downtown / Tenderloin / Civic Center",
    2: "Financial District / Embarcadero",
    3: "Chinatown / North Beach / Nob Hill",
    4: "SoMa / Mission Bay",
    5: "Mission District",
    6: "Castro / Noe Valley / Bernal Heights",
    7: "Hayes Valley / Haight",
    8: "Richmond / Presidio",
    9: "Sunset",
    10: "Bayview / Hunters Point / Excelsior",
}

_PRIMARY_COLS = {
    "Date": "date",
    "Location (include Street Type: St, Ave, Blvd, etc)": "location",
    "Summary/Narrative": "narrative",
    "Number of AVs Involved": "n_avs",
    "Time Event Started": "time_started",
    "Time Event Cleared": "time_cleared",
}

_ANALYZED_COLS = {
    "ETA Provided by AV Company in Minutes": "eta_min",
    "Other Parties Involved (ex. Passengers)": "other_parties",
    "Emergency Responder Impact": "er_impact_flag",
    "Transit Impact": "transit_impact_flag",
}


def _norm_location(s: pd.Series) -> pd.Series:
    return s.astype("string").str.upper().str.replace(r"\s+", " ", regex=True).str.strip()


def _join_key(df: pd.DataFrame, loc_col: str) -> pd.Series:
    return df["date"].dt.strftime("%Y-%m-%d") + "|" + _norm_location(df[loc_col])


def parse_clock(value) -> float:
    """Return minutes after midnight for a ``datetime.time`` or ``'H:MM'`` string."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NaT:
        return np.nan
    if isinstance(value, dt.datetime):
        value = value.time()
    if isinstance(value, dt.time):
        return value.hour * 60 + value.minute + value.second / 60
    m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*", str(value))
    if not m:
        return np.nan
    h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
    return h * 60 + mi + s / 60


def duration_minutes(start_min: pd.Series, end_min: pd.Series) -> pd.Series:
    """Clear minus start, wrapping past midnight (an event never spans > 24 h here)."""
    return (end_min - start_min) % 1440


def clean_narrative(s: pd.Series) -> pd.Series:
    return s.astype("string").str.replace(r"\s+", " ", regex=True).str.strip().replace("", pd.NA)


def _likely_truncated(text) -> bool:
    """Heuristic: the export cut many narratives at ~45-60 characters mid-phrase."""
    if text is None or pd.isna(text):
        return False
    t = str(text).rstrip()
    return len(t) <= 62 and not re.search(r"[.!?)]$", t)


def operator_of(text) -> str:
    if text is None or pd.isna(text):
        return "Unknown"
    t = str(text).lower()
    if "zoox" in t:
        return "Zoox"
    if "waymo" in t or "wamo" in t or "waumo" in t:
        return "Waymo"
    return "Unknown"


def load_primary(path=config.RAW_PRIMARY) -> pd.DataFrame:
    """Read the primary DEM sheet and derive times, durations, and calendar fields."""
    raw = pd.read_excel(path, sheet_name=config.PRIMARY_SHEET, engine="openpyxl")
    df = raw.rename(columns=_PRIMARY_COLS)[list(_PRIMARY_COLS.values())].copy()
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    df.insert(0, "incident_id", [f"INC-{i + 1:03d}" for i in range(len(df))])

    df["narrative"] = clean_narrative(df["narrative"])
    df["narrative_missing"] = df["narrative"].isna()
    df["narrative_truncated"] = df["narrative"].map(_likely_truncated)
    df["operator"] = df["narrative"].map(operator_of)

    df["start_min"] = df["time_started"].map(parse_clock)
    df["clear_min"] = df["time_cleared"].map(parse_clock)
    df["duration_min"] = duration_minutes(df["start_min"], df["clear_min"])
    df["crosses_midnight"] = df["clear_min"] < df["start_min"]

    df["n_avs_missing"] = df["n_avs"].isna()
    df["n_avs"] = df["n_avs"].astype("Int64")
    # At least one AV is involved in every logged incident.
    df["n_avs_filled"] = df["n_avs"].fillna(1).astype(int)
    df["multi_av"] = df["n_avs_filled"] >= 2

    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["weekday"] = df["date"].dt.day_name()
    df["start_hour"] = (df["start_min"] // 60).astype("Int64")
    df["in_covered_window"] = df["month"].isin(config.COVERED_MONTHS)
    return df.drop(columns=["time_started", "time_cleared"])


def load_analyzed(path=config.RAW_ANALYZED) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=config.PRIMARY_SHEET, engine="openpyxl")
    df = raw.rename(columns={**_PRIMARY_COLS, **_ANALYZED_COLS})
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    df["narrative"] = clean_narrative(df["narrative"])
    df["key"] = _join_key(df, "location")
    return df


def load_quant_neighborhoods(path=config.RAW_ANALYZED) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=config.QUANT_SHEET, engine="openpyxl")
    raw = raw.rename(columns={"Date": "date", raw.columns[1]: "location", raw.columns[2]: "zone"})
    raw = raw[pd.to_datetime(raw["date"], errors="coerce").notna()].copy()
    raw["date"] = pd.to_datetime(raw["date"])
    raw["key"] = _join_key(raw, "location")
    raw["zone"] = pd.to_numeric(raw["zone"], errors="coerce").astype("Int64")
    return raw[["key", "zone"]].drop_duplicates("key")


def _yes(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.upper().eq("Y").fillna(False).astype(bool)


def scope_exclusions() -> pd.DataFrame:
    """Records excluded as out of scope (``record`` = incident_id, or date|location key)."""
    if not config.SCOPE_EXCLUSIONS.exists():
        return pd.DataFrame(columns=["record", "source", "reason"])
    return pd.read_csv(config.SCOPE_EXCLUSIONS, dtype=str)


def build_incidents() -> pd.DataFrame:
    """Primary incidents enriched with the secondary-sheet fields (left joins)."""
    df = load_primary()
    df["key"] = _join_key(df, "location")

    analyzed = load_analyzed()
    extra = analyzed[["key", *_ANALYZED_COLS.values()]].drop_duplicates("key")
    df = df.merge(extra, on="key", how="left", validate="one_to_one")
    df["eta_min"] = pd.to_numeric(df["eta_min"], errors="coerce")
    df["er_impact_flag"] = _yes(df["er_impact_flag"])
    df["transit_impact_flag"] = _yes(df["transit_impact_flag"])
    df["other_parties"] = df["other_parties"].astype("string")

    zones = load_quant_neighborhoods()
    df = df.merge(zones, on="key", how="left", validate="one_to_one")
    df["neighborhood"] = df["zone"].map(NEIGHBORHOODS).fillna("Unassigned")

    excl = scope_exclusions().set_index("record")["reason"]
    df["in_scope"] = ~df["incident_id"].isin(excl.index)
    df["scope_note"] = df["incident_id"].map(excl).astype("string")
    return df.drop(columns=["key"])


def supplementary_incidents() -> pd.DataFrame:
    """Incidents in the analyzed workbook but not the primary sheet that have a narrative.

    They lack start/clear times, so they are excluded from duration analysis
    but are useful for a frequency sensitivity check.
    """
    primary_keys = set(_join_key(load_primary(), "location"))
    a = load_analyzed()
    out_of_scope = set(scope_exclusions()["record"])
    out = a[
        ~a["key"].isin(primary_keys) & a["narrative"].notna() & ~a["key"].isin(out_of_scope)
    ].copy()
    out["month"] = out["date"].dt.to_period("M").astype(str)
    out["n_avs_filled"] = pd.to_numeric(out["n_avs"], errors="coerce").fillna(1).astype(int)
    return out[["key", "date", "month", "location", "narrative", "n_avs_filled"]].reset_index(
        drop=True
    )


def observation_days(months: list[str] | None = None) -> pd.Series:
    """Calendar days in each covered month (the exposure for rate estimates)."""
    months = months or config.COVERED_MONTHS
    return pd.Series({m: pd.Period(m).days_in_month for m in months}, name="days")
