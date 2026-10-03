"""Compare narrative-classification approaches against the manual audit labels.

Three families are evaluated:

1. **Rule-based** (:mod:`av_blocking.taxonomy`): needs no training data.
2. **Supervised learned models**: TF-IDF (word + character n-grams) with
   multinomial logistic regression, and with XGBoost, scored by repeated
   stratified cross-validation on the manual labels. Each fold re-applies the
   H6 count rule (2+ AVs) on top of the text prediction, as the taxonomy does,
   because AV counts are a structured field, not text.
3. **Zero-shot transformer** (optional, ``uv sync --extra nlp``): an NLI model
   scores each narrative against natural-language hazard descriptions without
   any training. Results are cached because the model is large and CPU-slow.

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

ZEROSHOT_MODEL = "facebook/bart-large-mnli"

# Hypothesis text per hazard for zero-shot NLI. H6 is not textual (AV count),
# so the zero-shot model chooses among H1-H5 and "general traffic" (H7), and
# the count rule then promotes H7 -> H6 exactly as the rule classifier does.
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


# --- optional zero-shot ------------------------------------------------------------
def zeroshot_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def run_zeroshot(
    df: pd.DataFrame, model: str = ZEROSHOT_MODEL, batch_size: int = 8
) -> pd.DataFrame:
    """Zero-shot NLI classification; writes the cache CSV and returns it."""
    from transformers import pipeline

    clf = pipeline("zero-shot-classification", model=model, device=-1)
    texts = df["narrative"].fillna("(no narrative)").tolist()
    labels = list(ZEROSHOT_LABELS.values())
    inv = {v: k for k, v in ZEROSHOT_LABELS.items()}
    results = clf(
        texts,
        candidate_labels=labels,
        hypothesis_template="This incident report describes {}.",
        batch_size=batch_size,
    )
    out = pd.DataFrame(
        {
            "incident_id": df["incident_id"].values,
            "zs_text_hazard": [inv[r["labels"][0]] for r in results],
            "zs_score": [r["scores"][0] for r in results],
        }
    )
    for code, lab in ZEROSHOT_LABELS.items():
        out[f"zs_p_{code}"] = [r["scores"][r["labels"].index(lab)] for r in results]
    config.ZEROSHOT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.ZEROSHOT_CACHE, index=False)
    return out


def load_zeroshot(df: pd.DataFrame, run_if_missing: bool = False) -> pd.DataFrame | None:
    """Cached zero-shot predictions with the H6 count rule applied, or ``None``."""
    if config.ZEROSHOT_CACHE.exists():
        zs = pd.read_csv(config.ZEROSHOT_CACHE)
    elif run_if_missing and zeroshot_available():
        zs = run_zeroshot(df)
    else:
        return None
    zs = zs.merge(df[["incident_id", "n_avs_filled"]], on="incident_id")
    zs["zs_hazard"] = _apply_count_rule(
        zs["zs_text_hazard"].values.astype(object), zs["n_avs_filled"].values
    )
    return zs


def hazard_label(code: str) -> str:
    return f"{code} {HAZARD_BY_CODE[code].short}"
