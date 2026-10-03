"""One-call construction of the analysis-ready incident table."""

from __future__ import annotations

import pandas as pd

from . import config, data, severity, taxonomy


def prepare() -> pd.DataFrame:
    """Load + enrich incidents, classify hazards and causes, score impact, attach audit labels."""
    df = data.build_incidents()
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
    return df
