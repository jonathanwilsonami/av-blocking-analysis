| Quantity                             | Null hypothesis                                                  | Test                                | Statistic / effect                     | p       | Reject at 0.05?   |
|:-------------------------------------|:-----------------------------------------------------------------|:------------------------------------|:---------------------------------------|:--------|:------------------|
| Monthly incident rate                | Equal rate across months                                         | Poisson homogeneity chi-square      | chi2(5) = 7.17                         | 0.209   | No                |
| Monthly incident rate                | No monotone trend                                                | Mann-Kendall (exact)                | S = 2, tau = 0.13                      | 0.856   | No                |
| Monthly incident rate                | Mar rate = Oct rate                                              | Exact conditional binomial          | RR = 1.14 (0.61-2.11)                  | 0.771   | No                |
| Monthly incident rate                | No log-linear trend                                              | Poisson GLM, log(days) offset       | RR/month = 0.998 (0.922-1.081)         | 0.965   | No                |
| Duration                             | Same distribution in all months                                  | Kruskal-Wallis                      | H(5) = 6.45, eps2 = 0.013              | 0.265   | No                |
| Duration                             | Mar = Oct                                                        | Mann-Whitney U                      | U = 215.0, r_rb = 0.22                 | 0.204   | No                |
| Duration                             | Mar = Oct (class-sheet sample: full log, 603-min record dropped) | Mann-Whitney U                      | W = 377.5                              | 0.064   | No                |
| Duration                             | No ordered shift across months                                   | Jonckheere-Terpstra (permutation)   | JT = 3032, z = 1.05                    | 0.290   | No                |
| Duration                             | No association with date                                         | Spearman; Theil-Sen on log duration | rho = 0.07; +3.9%/30 d (-6.5 to +15.5) | 0.435   | No                |
| Hazard mix                           | Independent of period                                            | Permutation chi-square              | chi2 = 16.06                           | 0.008   | Yes               |
| Narrative truncation                 | Independent of period                                            | Fisher exact                        | 98% vs 41% truncated                   | < 0.001 | Yes               |
| Text-dependent hazards (H1-H5) share | Independent of period                                            | Fisher exact                        | 23% vs 33%                             | 0.306   | No                |
| Multi-AV share                       | No monotone trend                                                | Cochran-Armitage                    | z = 1.23                               | 0.217   | No                |

: Statistical tests for change over time (covered months). {#tbl-tests}
