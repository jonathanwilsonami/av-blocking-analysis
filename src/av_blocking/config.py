"""Project paths and analysis-wide constants.

Everything that a reader might want to change when re-running the analysis
(observation window, reference horizon, probability bands) lives here so the
notebook and the paper stay consistent.
"""

from __future__ import annotations

from pathlib import Path

# --- paths --------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_PRIMARY = DATA_DIR / "AV-Brick-Report.xlsx"
RAW_ANALYZED = DATA_DIR / "AV-Brick-Report-analyzed.xlsx"
LABELS_DIR = DATA_DIR / "labels"
MANUAL_LABELS = LABELS_DIR / "manual_hazard_labels.csv"
PROCESSED_DIR = DATA_DIR / "processed"
ZEROSHOT_CACHE = PROCESSED_DIR / "zeroshot_predictions.csv"

SITE_DIR = ROOT / "project-site"
ASSETS_DIR = SITE_DIR / "assets"
FIG_DIR = ASSETS_DIR / "figures"
TABLE_DIR = ASSETS_DIR / "tables"
SNIPPET_DIR = ASSETS_DIR / "snippets"
VARIABLES_FILE = SITE_DIR / "_variables.yml"

PRIMARY_SHEET = "DEM Incidents"
QUANT_SHEET = "DEM Incidents Quant"

# --- observation window ---------------------------------------------------------
# The DEM log is not continuous: February holds only three incidents on its
# first three days, and April-May are absent entirely. Treating absent months
# as zero-incident months would bias every rate and trend estimate downward, so
# rates and trends use only months judged to be fully covered.
COVERED_MONTHS = ["2025-03", "2025-06", "2025-07", "2025-08", "2025-09", "2025-10"]

# --- risk-matrix probability basis ---------------------------------------------
# Probability = P(at least one event of the given kind in the next HORIZON_DAYS).
HORIZON_DAYS = 30
# Upper edges of the Rare / Unlikely / Moderate / Likely bands on that
# probability (same bands as the team's reference risk-matrix template).
PROB_BAND_EDGES = [0.05, 0.25, 0.50, 0.90]

RANDOM_SEED = 568
