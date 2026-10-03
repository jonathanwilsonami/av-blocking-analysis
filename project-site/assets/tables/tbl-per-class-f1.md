| Hazard                   |   n (manual) |   TF-IDF + logistic regression |   TF-IDF + XGBoost |   Zero-shot BART-MNLI |   Rule-based |
|:-------------------------|-------------:|-------------------------------:|-------------------:|----------------------:|-------------:|
| H1 Collision             |            7 |                           0.67 |               0.4  |                  0.31 |         0.92 |
| H2 Emergency obstruction |            6 |                           0    |               0    |                  0.33 |         0.92 |
| H3 Transit obstruction   |            8 |                           0.46 |               0.22 |                  0.22 |         1    |
| H4 Ped/accessibility     |            4 |                           0.33 |               0    |                  0    |         1    |
| H5 Occupant-related      |            9 |                           0.46 |               0    |                  0.5  |         1    |
| H6 Multi-AV gridlock     |           17 |                           0.94 |               0.94 |                  0.67 |         1    |
| H7 Single-AV obstruction |           72 |                           0.86 |               0.83 |                  0.74 |         1    |

: Per-class F1 by method. {#tbl-per-class-f1}
