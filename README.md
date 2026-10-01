# AI_A1_G11 - Musanze Cooperative Harvest and Dispatch Decision Lab

SWE 3513 Artificial Intelligence - Assignment 1 - INES-Ruhengeri

| Item | Value |
|---|---|
| Group number / code | AI_A1_G11 / AI-G11 |
| Group leader | NIWENIRINGIYE Christian (25/27889) |
| Repository | https://github.com/cxnisme/AI_A1_G11 |
| Final commit hash | Fill after final evidence commit and push |
| Dataset SHA-256 | `288446a93592327e4c419696e452e0e843882a0e9c08f9dbf6c7531f6eb12e5f` (must equal the lecturer-issued file; re-check with `sha256sum data/AI_A1_G11.csv`) |
| Python tested | 3.12.3 (Windows, isolated virtual environment) |

## Members and roles

| Member | Name | Registration no. | GitHub username | Role |
|---|---|---|---|---|
| 1 | IHIMBAZWE Angelique | Not provided | Angelique-123 | Data and UX lead |
| 2 | RUDASINGWA Theogene | 25/27330 | rudasingwatheogene1 | Regression engineer |
| 3 | IGIRANEZA Alain Providence | 25/28075 | alain143 | Classification engineer |
| 4 | Bullen Ladu Martin | 24/23862 | ladumartinbullen-cmyk | Clustering and QA engineer |
| 5 | NIWENIRINGIYE Christian | 25/27889 | Not provided | Reproducibility and release lead |

GitHub usernames are not necessarily Git commit author names or emails. Record each member's exact `git config user.name` or `git config user.email` value before generating the signed contribution record.

## Setup (clean environment)

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        Linux/macOS:  source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the complete pipeline

```bash
python run_all.py --data data/AI_A1_G11.csv --output artifacts/ --group AI-G11
```

It prints the group code, row count, feature count and the dataset SHA-256, and (over)writes everything in `artifacts/` plus the saved models in `models/`. Run it from the project root.

Options used for live verification (all have defaults, nothing is hard-coded):

| Option | Default | Effect |
|---|---|---|
| `--seed` | 42 | Random seed for splits, logistic regression and k-means; recorded in every metrics file |
| `--lr` | 0.01 | Gradient-descent learning rate |
| `--epochs` | 5000 | Gradient-descent iterations |
| `--threshold` | 0.5 | Classification decision threshold on P(dispatch_attention = 1) |
| `--kmin`, `--kmax` | 2, 5 | Range of k evaluated for clustering |

## Predict one record

```bash
python predict.py --record '{"plot_area_ha":1.2,"rainfall_mm":81,"soil_ph":5.7,"seed_kg":210,"distance_km":14,"arrival_hour":9}'
```

Returns JSON with `regression_prediction`, `classification_prediction`, `classification_probability`, `cluster_label`, `group_code`, `model_version`. Run `run_all.py` first (it creates `models/`).
Missing, extra, non-numeric, non-finite or implausible fields (e.g. `soil_ph` > 14, `arrival_hour` > 24) are rejected with `{"error": "..."}` and exit code 2. On Windows PowerShell, quote the JSON as `'{\"plot_area_ha\":1.2,...}'` or use `--record "{\"plot_area_ha\":1.2,...}"`.

## Tests

```bash
python -m unittest discover -s tests -v
```

Twelve checks run on a temporary copy, so `artifacts/` is not touched: full run, all artifacts present, SHA-256 equals the file, regression MAE recomputed from saved predictions, confusion matrix consistent with accuracy, every record clustered with k=2..5, valid predict, rejected predict inputs, hidden-like data (different rows, shuffled column order, a missing value), the live-change options, a clear error for a diverging learning rate, and gradient descent matching the closed-form least-squares solution.

The isolated Windows virtual-environment run and its 12-test result are recorded in `evidence/CLEAN_ENV_RUN.txt`. The machine-specific command log, including prediction checks, is in `evidence/TEST_LOG.pdf`.

## Expected outputs (`artifacts/`)

`data_report.json`, `regression_metrics.json`, `regression_loss.png`, `classification_metrics.json`, `confusion_matrix.png`, `clustering_metrics.json`, `clusters.csv`, `cluster_plot.png`. `models/` holds `regression_model.npz`, `classification_model.joblib`, `clustering_model.joblib` and two small JSON configs read by `predict.py`.

## Method

**Data (`src/data_pipeline.py`).** Checks the nine required columns (any order), unique `record_id`, numeric types, 0/1 target. Reports missing values, duplicates, descriptive statistics, SHA-256. `record_id` is never a feature. Features: `plot_area_ha, rainfall_mm, soil_ph, seed_kg, distance_km, arrival_hour`. Rows used for supervised models need both targets present.

**Regression (`src/regression.py`).** NumPy only, no estimator. Median imputation and standardisation are fitted on the training split only. Model: y_hat = Xw with a bias column. Loss: MSE = (1/n) * sum((Xw - y)^2). Gradient: (2/n) * X^T (Xw - y). Update: w <- w - lr * gradient, repeated for `--epochs`. Reports MAE, RMSE, R-squared on the held-out 20 %; saves the loss history plot.

**Classification (`src/classification.py`).** Logistic regression in a scikit-learn pipeline (median imputer, scaler, model) fitted on the training split only; stratified 80/20 split when both classes have at least two rows. Reports confusion matrix, accuracy, precision, recall, F1. A false negative (a consignment needing attention is released) is the costlier error; a false positive costs only a review. No tuning on the test set.

**Clustering (`src/clustering.py`).** Only the six input features (no targets, no id). Median imputation, standardisation, k-means for each k in the range, silhouette score per k, highest score selected, label saved for every record. Clusters are data groupings, not verified real-world categories.

## Known limitations

- The data set is small and fictional; with 20 rows the held-out test set has 4 rows, so test metrics are unstable and not strong evidence of real-world performance.
- Clustering and the classifier are fitted on the supplied rows only; predictions outside the training range are extrapolations.
- The dashboard in `AI_A1_G11_UIUX.pdf` is a design concept; it is not implemented.
- Generative AI was used in building this project; see `evidence/AI_USE.md`.
