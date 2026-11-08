# UzPredict - model results

Chronological 80/20 split: the last ~20% of years form the test set.

| Model | Years | Rows | Metric 1 | Metric 2 | Metric 3 | Notes |
|---|---|---|---|---|---|---|
| Simple linear (GDP pc ~ year) | 1987-2025 | 39 | 0.457 | 492.656 | 425.036 | intercept=-78.2, slope per year=81.8 |
| Polynomial degree 2 (GDP pc ~ year) | 1987-2025 | 39 | -8.013 | 2006.273 | 1997.990 | coefficients=[844.2, -1090.9, 636.1] |
| Multiple linear (GDP pc ~ 5 indicators) | 2011-2024 | 14 | -1.127 | 380.381 | 351.606 | standardized coefficients: internet=-263.5, trade=-40.8, life_exp=197.4, inflation=-341.1, unemployment=-65.1 |
| Logistic (inflation > 10%) | 2011-2024 | 14 | 0.333 | 0.333 | 1.000 | accuracy/precision/recall on test set; weights: trade=8.18, unemployment=0.68, internet=-3.27 |

Metric 1-3 meaning: regression models = R2 / RMSE / MAE (test set);
logistic model = accuracy / precision / recall (test set).
