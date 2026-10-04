| method                                   | family                       |   macro_F1 | macro_F1_sd   |   accuracy |   kappa |
|:-----------------------------------------|:-----------------------------|-----------:|:--------------|-----------:|--------:|
| Majority class + count rule              | Baseline                     |      0.249 | —             |      0.724 |   0.433 |
| TF-IDF + XGBoost (CV)                    | Supervised (cross-validated) |      0.32  | 0.074         |      0.735 |   0.477 |
| TF-IDF + logistic regression (CV)        | Supervised (cross-validated) |      0.47  | 0.085         |      0.774 |   0.582 |
| Zero-shot BART-large-MNLI                | Pretrained, no training      |      0.396 | —             |      0.585 |   0.378 |
| Zero-shot DeBERTa-v3-large NLI           | Pretrained, no training      |      0.323 | —             |      0.504 |   0.279 |
| Embedding similarity (mxbai-embed-large) | Pretrained, no training      |      0.62  | —             |      0.691 |   0.563 |
| LLM + coding guide (Codex, GPT-6-based)  | LLM with coding guide        |      0.946 | —             |      0.967 |   0.947 |
| Rule-based (selected; in-sample)         | Rules                        |      0.978 | —             |      0.992 |   0.987 |

: Narrative classifiers scored against the manual audit labels (supervised models: mean and SD over 4-fold x 5 repeated stratified CV; pretrained models and the LLM use no training data; the LLM was given the written coding guide). {#tbl-classifiers}
