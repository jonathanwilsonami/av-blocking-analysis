"""Compare narrative-classification approaches against the manual audit labels.

Three families are evaluated:

1. **Rule-based** (:mod:`av_blocking.taxonomy`): needs no training data.
2. **Supervised learned models**: TF-IDF (word + character n-grams) with
   multinomial logistic regression, and with XGBoost, scored by repeated
   stratified cross-validation on the manual labels. Each fold re-applies the
   H6 count rule (2+ AVs) on top of the text prediction, as the taxonomy does,
   because AV counts are a structured field, not text.
3. **Pretrained models with no training on this data** (optional,
   ``uv sync --extra nlp``): two zero-shot NLI transformers (BART-large-MNLI and
   DeBERTa-v3-large NLI) that score each narrative against natural-language
   hazard descriptions, and an embedding-similarity baseline. Predictions are
   cached per model because the models are large and CPU-slow.

The point of the comparison is method selection: with ~120 short, often
truncated narratives and some classes of 4-9 examples, a learned model cannot
estimate the rare classes that matter most for risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from . import config
from .taxonomy import HAZARD_BY_CODE, HAZARD_CODES

# Pretrained models compared without any training on this data. Every model sees
# the same narratives and the same hazard descriptions (ZEROSHOT_LABELS).
#   kind="nli":       zero-shot NLI -- each description becomes the hypothesis
#                     "This incident report describes {description}." and the
#                     model scores entailment against the narrative.
#   kind="embedding": cosine similarity between the narrative embedding and each
#                     description embedding; the most similar description wins.
PRETRAINED_MODELS = {
    "bart": {
        "name": "Zero-shot BART-large-MNLI",
        "kind": "nli",
        "model": "facebook/bart-large-mnli",
    },
    "deberta": {
        "name": "Zero-shot DeBERTa-v3-large NLI",
        "kind": "nli",
        "model": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
    },
    "embedding": {
        "name": "Embedding similarity (mxbai-embed-large)",
        "kind": "embedding",
        "model": "mixedbread-ai/mxbai-embed-large-v1",
        # mxbai's recommended prefix for the query side of a retrieval pair
        "query_prompt": "Represent this sentence for searching relevant passages: ",
    },
}
HYPOTHESIS_TEMPLATE = "This incident report describes {}."

# Natural-language description of each hazard, shared by every pretrained model.
# H6 is not textual (AV count), so the models choose among H1-H5 and "general
# traffic" (H7), and the count rule then promotes H7 -> H6 exactly as the rule
# classifier does.
ZEROSHOT_LABELS = {
    "H1": "a collision or crash involving the autonomous vehicle",
    "H2": "blocking an ambulance, fire truck, police, or an emergency scene",
    "H3": "blocking a bus, Muni, cable car, train, or rail track",
    "H4": "blocking a crosswalk, wheelchair ramp, or bike lane",
    "H5": "a passenger problem such as a sleeping or unconscious rider or an open door",
    "H7": "a stalled vehicle blocking traffic",
}


def _vectorizer() -> FeatureUnion:
    return FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True
                ),
            ),
        ]
    )


def make_models() -> dict:
    return {
        "TF-IDF + logistic regression": make_pipeline(
            _vectorizer(),
            LogisticRegression(max_iter=5000, C=10, class_weight="balanced"),
        ),
        "TF-IDF + XGBoost": make_pipeline(
            _vectorizer(),
            XGBClassifier(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.5,
                eval_metric="mlogloss",
                random_state=config.RANDOM_SEED,
                n_jobs=1,
            ),
        ),
    }


def _apply_count_rule(pred: np.ndarray, n_avs: np.ndarray) -> np.ndarray:
    pred = pred.copy()
    pred[(pred == "H7") & (n_avs >= 2)] = "H6"
    return pred


def _scores(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_F1": f1_score(y_true, y_pred, average="macro", labels=HAZARD_CODES, zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }


def cross_validate_models(
    df: pd.DataFrame, label_col: str = "manual_hazard", n_splits: int = 4, n_repeats: int = 5
) -> tuple[pd.DataFrame, dict]:
    """Repeated stratified CV of learned models; returns per-fold scores and pooled predictions."""
    text = df["narrative"].fillna("").str.lower().values
    y = df[label_col].values
    n_avs = df["n_avs_filled"].values
    # Learned models predict the textual class; H6 is restored by the count rule.
    y_text = np.where(y == "H6", "H7", y)
    enc = LabelEncoder().fit(y_text)
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=config.RANDOM_SEED
    )
    rows, pooled = [], {}
    for name, model in make_models().items():
        preds = np.empty((n_repeats, len(y)), dtype=object)
        for k, (tr, te) in enumerate(cv.split(text, y)):
            model.fit(text[tr], enc.transform(y_text[tr]))
            p = enc.inverse_transform(model.predict(text[te]))
            p = _apply_count_rule(p, n_avs[te])
            preds[k // n_splits, te] = p
            rows.append({"model": name, "fold": k, **_scores(y[te], p)})
        pooled[name] = preds[0]
    return pd.DataFrame(rows), pooled


def majority_baseline(df: pd.DataFrame, label_col: str = "manual_hazard") -> dict:
    y = df[label_col].values
    pred = _apply_count_rule(np.full(len(y), "H7", dtype=object), df["n_avs_filled"].values)
    return _scores(y, pred)


def per_class_f1(y_true, y_pred) -> pd.Series:
    f1 = f1_score(y_true, y_pred, average=None, labels=HAZARD_CODES, zero_division=0)
    return pd.Series(f1, index=HAZARD_CODES)


# --- optional pretrained models ----------------------------------------------------
def zeroshot_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def _scores_to_frame(df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """(n_incidents x n_labels) score matrix -> cache frame with argmax label."""
    codes = list(ZEROSHOT_LABELS)
    best = scores.argmax(axis=1)
    out = pd.DataFrame(
        {
            "incident_id": df["incident_id"].values,
            "zs_text_hazard": [codes[i] for i in best],
            "zs_score": scores[np.arange(len(best)), best],
        }
    )
    for j, code in enumerate(codes):
        out[f"zs_p_{code}"] = scores[:, j]
    return out


def _run_nli(texts: list[str], model: str, batch_size: int) -> np.ndarray:
    from transformers import pipeline

    clf = pipeline("zero-shot-classification", model=model, device=-1)
    labels = list(ZEROSHOT_LABELS.values())
    results = clf(
        texts,
        candidate_labels=labels,
        hypothesis_template=HYPOTHESIS_TEMPLATE,
        batch_size=batch_size,
    )
    return np.array([[r["scores"][r["labels"].index(lab)] for lab in labels] for r in results])


def _run_embedding(texts: list[str], model: str, query_prompt: str = "") -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(model, device="cpu")
    q = enc.encode([query_prompt + t for t in texts], normalize_embeddings=True)
    d = enc.encode(list(ZEROSHOT_LABELS.values()), normalize_embeddings=True)
    return q @ d.T  # cosine similarity (embeddings are unit-normalized)


def run_zeroshot(df: pd.DataFrame, key: str = "bart", batch_size: int = 8) -> pd.DataFrame:
    """Score every narrative with one pretrained model; writes and returns its cache."""
    spec = PRETRAINED_MODELS[key]
    texts = df["narrative"].fillna("(no narrative)").tolist()
    if spec["kind"] == "nli":
        scores = _run_nli(texts, spec["model"], batch_size)
    else:
        scores = _run_embedding(texts, spec["model"], spec.get("query_prompt", ""))
    out = _scores_to_frame(df, scores)
    path = config.zeroshot_cache(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def load_zeroshot(
    df: pd.DataFrame, key: str = "bart", run_if_missing: bool = False
) -> pd.DataFrame | None:
    """Cached predictions for one model with the H6 count rule applied, or ``None``."""
    path = config.zeroshot_cache(key)
    if path.exists():
        zs = pd.read_csv(path)
    elif run_if_missing and zeroshot_available():
        zs = run_zeroshot(df, key)
    else:
        return None
    zs = zs.merge(df[["incident_id", "n_avs_filled"]], on="incident_id")
    zs["zs_hazard"] = _apply_count_rule(
        zs["zs_text_hazard"].values.astype(object), zs["n_avs_filled"].values
    )
    return zs


def load_all_zeroshot(df: pd.DataFrame, run_if_missing: bool = False) -> dict[str, pd.Series]:
    """Display name -> predicted hazard (aligned to ``df``) for every available model."""
    out = {}
    for key, spec in PRETRAINED_MODELS.items():
        zs = load_zeroshot(df, key, run_if_missing)
        if zs is not None:
            pred = df[["incident_id"]].merge(zs, on="incident_id", how="left")["zs_hazard"]
            out[spec["name"]] = pred.values
    return out


# --- paired comparison of two classifiers on the same incidents ---------------------
def mcnemar_exact(y_true, pred_a, pred_b) -> dict:
    """Exact McNemar test on per-incident correctness of two classifiers."""
    from scipy import stats

    a_ok = np.asarray(pred_a) == np.asarray(y_true)
    b_ok = np.asarray(pred_b) == np.asarray(y_true)
    only_a, only_b = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    n = only_a + only_b
    p = 1.0 if n == 0 else float(stats.binomtest(only_a, n, 0.5).pvalue)
    return {"only_a_correct": only_a, "only_b_correct": only_b, "p_value": p}


def paired_bootstrap_f1(
    y_true, pred_a, pred_b, n_boot: int = 5000, seed: int = config.RANDOM_SEED
) -> dict:
    """Macro-F1 difference (a - b) with a paired percentile-bootstrap 95% CI."""
    y, a, b = (np.asarray(v, dtype=object) for v in (y_true, pred_a, pred_b))

    def f1(t, p):
        return f1_score(t, p, average="macro", labels=HAZARD_CODES, zero_division=0)

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        diffs[i] = f1(y[idx], a[idx]) - f1(y[idx], b[idx])
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"delta_macro_F1": f1(y, a) - f1(y, b), "ci_lo": float(lo), "ci_hi": float(hi)}


def hazard_label(code: str) -> str:
    return f"{code} {HAZARD_BY_CODE[code].short}"
