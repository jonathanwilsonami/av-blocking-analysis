"""One-call construction of the analysis-ready incident table."""

from __future__ import annotations

import pandas as pd

from . import config, data, severity, taxonomy


def prepare(include_out_of_scope: bool = False) -> pd.DataFrame:
    """Load + enrich incidents, classify hazards and causes, score impact, attach audit labels.

    Out-of-scope records (``data/labels/scope_exclusions.csv``) are dropped unless
    ``include_out_of_scope``. H1/H2 labels whose consequence is inferred rather than
    stated get ``hazard_evidence = "inferred"`` and a ``strict_hazard`` alternative
    (``data/labels/evidence_flags.csv``); everything else keeps its own hazard.
    """
    df = data.build_incidents()
    if not include_out_of_scope:
        df = df[df["in_scope"]].reset_index(drop=True)
    df = taxonomy.apply_taxonomy(df)
    df = severity.score_incidents(df)
    if config.MANUAL_LABELS.exists():
        labels = pd.read_csv(config.MANUAL_LABELS, dtype=str, keep_default_na=False)
        df = df.merge(
            labels.rename(columns={"note": "manual_note"}),
            on="incident_id",
            how="left",
            validate="one_to_one",
        )
    df["hazard_evidence"] = "explicit"
    df["strict_hazard"] = df["hazard"]
    if config.EVIDENCE_FLAGS.exists():
        flags = pd.read_csv(config.EVIDENCE_FLAGS, dtype=str).set_index("incident_id")
        hit = df["incident_id"].isin(flags.index)
        df.loc[hit, "hazard_evidence"] = df.loc[hit, "incident_id"].map(flags["hazard_evidence"])
        df.loc[hit, "strict_hazard"] = df.loc[hit, "incident_id"].map(flags["strict_hazard"])
        df["evidence_note"] = df["incident_id"].map(flags["note"]).astype("string")
    df["weak_av_evidence"] = False
    if config.WEAK_AV_EVIDENCE.exists():
        weak = pd.read_csv(config.WEAK_AV_EVIDENCE, dtype=str).set_index("incident_id")["note"]
        df["weak_av_evidence"] = df["incident_id"].isin(weak.index)
        df["weak_av_note"] = df["incident_id"].map(weak).astype("string")
    return df
