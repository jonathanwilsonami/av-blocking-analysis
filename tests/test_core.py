import numpy as np
import pandas as pd
import pytest

from av_blocking import data, frequency, risk_matrix, severity, taxonomy, trends


@pytest.mark.parametrize(
    "text,n,expected",
    [
        ("Waymo was blkng medics way", 1, "H2"),
        ("SWaymo vs Muni, unclear if contact was actually", 1, "H1"),
        ("WAYMO blocking MUNI track. WAYMO tech en route", 1, "H3"),
        ("Waymo blkng handicap ramp..Waymo ntfyd", 1, "H4"),
        ("Unoccupyd AV is stalled right off of Bayshore off ramp", 1, "H7"),
        ("Rider asleep in the back of Waymo..reqs PD wake up", 1, "H5"),
        ("On Brannan Waymo stalled..no driver unk if pssngrs", 1, "H7"),
        ("5 Waymo Av's blocking traffic", 5, "H6"),
        (None, 1, "H7"),
    ],
)
def test_classify_hazard(text, n, expected):
    assert taxonomy.classify_hazard(text, n) == expected


def test_precedence_prefers_more_severe():
    # Matches both the emergency (fire) and multi-AV rules -> H2 wins.
    assert taxonomy.classify_hazard("Working fire, 2 WAYMO vehicles stuck", 2) == "H2"


def test_duration_wraps_midnight():
    d = data.duration_minutes(pd.Series([23 * 60 + 50]), pd.Series([26]))
    assert d.iloc[0] == 36


def test_parse_clock_formats():
    import datetime as dt

    assert data.parse_clock("7:29") == 449
    assert data.parse_clock(dt.time(14, 18)) == 858
    assert np.isnan(data.parse_clock(None))


def test_severity_components():
    assert severity.hazard_score("H1", "Muni vs Waymo non inj accident") == 4
    assert severity.hazard_score("H2", "fire with persons trapped") == 5
    assert severity.duration_score(9) == 1 and severity.duration_score(60) == 3
    assert severity.cluster_score(5) == 3


def test_level_lookup_is_not_score_threshold():
    assert risk_matrix.level_of(2, 2) == "Low"
    assert risk_matrix.level_of(1, 4) == "Medium"


@pytest.mark.parametrize("p,band", [(0.01, 1), (0.05, 1), (0.2, 2), (0.6, 4), (0.95, 5)])
def test_probability_band(p, band):
    assert risk_matrix.probability_band(p) == band


def test_predictive_probability_monotone_and_bounded():
    ps = [frequency.predictive_p_any(n, 184) for n in range(0, 30, 5)]
    assert all(0 < a < b < 1 for a, b in zip(ps, ps[1:], strict=False))


def test_risk_curve_ignores_unobserved_tiers():
    df = pd.DataFrame({"hazard": ["H7"] * 5, "impact": [1, 1, 2, 2, 2]})
    curve = frequency.risk_curve(df, "H7", exposure=184)
    assert curve["impact"].max() == 2


def test_mann_kendall_exact_monotone():
    res = trends.mann_kendall([1, 2, 3, 4, 5, 6])
    assert res["S"] == 15 and res["p_exact"] == pytest.approx(2 / 720)


def test_cochran_armitage_null():
    assert trends.cochran_armitage([5, 5, 5], [10, 10, 10], [1, 2, 3])["p_value"] == 1.0


def test_mcnemar_counts_discordant_pairs():
    from av_blocking import nlp

    y = ["H1", "H2", "H7", "H7"]
    res = nlp.mcnemar_exact(y, ["H1", "H2", "H7", "H1"], ["H7", "H2", "H7", "H7"])
    assert (res["only_a_correct"], res["only_b_correct"]) == (1, 1)
    assert res["p_value"] == 1.0


def test_paired_bootstrap_identical_predictions_is_zero():
    from av_blocking import nlp

    y = ["H1", "H2", "H7", "H7", "H6"]
    res = nlp.paired_bootstrap_f1(y, y, y, n_boot=200)
    assert res["delta_macro_F1"] == 0 and res["ci_lo"] == 0 and res["ci_hi"] == 0


def test_scores_to_frame_picks_argmax():
    from av_blocking import nlp

    df = pd.DataFrame({"incident_id": ["INC-001", "INC-002"]})
    scores = np.zeros((2, len(nlp.ZEROSHOT_LABELS)))
    scores[0, 0], scores[1, -1] = 0.9, 0.8  # H1, H7
    out = nlp._scores_to_frame(df, scores)
    assert out["zs_text_hazard"].tolist() == ["H1", "H7"]


def test_duration_edges_parameter():
    assert severity.duration_score(75) == 3
    assert severity.duration_score(75, edges=(10, 90)) == 2
