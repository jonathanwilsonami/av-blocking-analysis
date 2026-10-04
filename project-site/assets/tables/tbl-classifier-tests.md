| A                                        | B                                 |   Only A correct |   Only B correct | McNemar p   | Macro-F1 A - B (95% CI)   | CI excludes 0?   |
|:-----------------------------------------|:----------------------------------|-----------------:|-----------------:|:------------|:--------------------------|:-----------------|
| Zero-shot DeBERTa-v3-large NLI           | Zero-shot BART-large-MNLI         |                7 |               17 | 0.064       | -0.07 (-0.20 to +0.05)    | No               |
| Embedding similarity (mxbai-embed-large) | Zero-shot BART-large-MNLI         |               28 |               15 | 0.066       | +0.22 (+0.08 to +0.36)    | Yes              |
| Zero-shot BART-large-MNLI                | TF-IDF + logistic regression (CV) |               10 |               35 | < 0.001     | -0.14 (-0.29 to +0.02)    | No               |
| Zero-shot DeBERTa-v3-large NLI           | TF-IDF + logistic regression (CV) |                7 |               42 | < 0.001     | -0.21 (-0.39 to -0.03)    | Yes              |
| Embedding similarity (mxbai-embed-large) | TF-IDF + logistic regression (CV) |               19 |               31 | 0.119       | +0.09 (-0.06 to +0.23)    | No               |

: Paired comparisons of classifiers on the same incidents: exact McNemar test on correctness and paired-bootstrap 95% CI for the macro-F1 difference. {#tbl-classifier-tests}
