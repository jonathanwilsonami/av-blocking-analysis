| A                                        | B                                 |   Only A correct |   Only B correct | McNemar p   | Macro-F1 A - B (95% CI)   | CI excludes 0?   |
|:-----------------------------------------|:----------------------------------|-----------------:|-----------------:|:------------|:--------------------------|:-----------------|
| Zero-shot DeBERTa-v3-large NLI           | Zero-shot BART-large-MNLI         |                7 |               16 | 0.093       | -0.07 (-0.20 to +0.05)    | No               |
| Embedding similarity (mxbai-embed-large) | Zero-shot BART-large-MNLI         |               27 |               15 | 0.088       | +0.23 (+0.09 to +0.37)    | Yes              |
| Zero-shot BART-large-MNLI                | TF-IDF + logistic regression (CV) |               11 |               31 | 0.003       | -0.06 (-0.20 to +0.07)    | No               |
| Zero-shot DeBERTa-v3-large NLI           | TF-IDF + logistic regression (CV) |                9 |               38 | < 0.001     | -0.14 (-0.28 to +0.01)    | No               |
| Embedding similarity (mxbai-embed-large) | TF-IDF + logistic regression (CV) |               21 |               29 | 0.322       | +0.16 (+0.01 to +0.31)    | Yes              |

: Paired comparisons of classifiers on the same incidents: exact McNemar test on correctness and paired-bootstrap 95% CI for the macro-F1 difference. {#tbl-classifier-tests}
