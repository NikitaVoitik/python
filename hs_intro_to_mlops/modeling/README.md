# Polymarket Resolution — Modeling

A config-driven training pipeline that predicts whether a Polymarket market resolves **YES** or **NO**, using only the trade history available up to a cutoff point in the market's life (`cutoff_frac`, default `0.8`).

It's a binary classification task scored on **ROC AUC** — we rank markets to bet on the most confident ones, so ranking quality matters more than raw accuracy.

- **Model:** LightGBM (handles the categorical topic feature and NaN prices natively)
- **HPO:** Optuna, 25 trials, maximizing validation ROC AUC
- **Tracking:** local MLflow (experiment + model registry with a `@best` alias)
- **Data versioning:** DVC → Google Drive remote
## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/): `curl -LsSf https://astral.sh/uv/install.sh | sh`
## Setup

```bash
make install      # uv sync
```

## Quick start

The compact feature table is versioned with DVC and pulled from Google Drive.

```bash
make all          # dvc pull → HPO search → retrain best → export serving bundle
```

`make all` runs `make train` then `make retrain`. Individually:

```bash
make train        # dvc pull, then Optuna HPO; logs every trial to MLflow, registers @best
make retrain      # retrain best params on train+val, evaluate on test, export models/
make mlflow-ui    # browse runs at http://localhost:5000
```

After `make retrain`, the serving bundle is written to `models/`:

```
models/model.pkl            # trained LightGBM classifier
models/model_info.json      # feature names, params, test metrics, topic config
models/topic_clusterer.pkl  # TF-IDF + KMeans for the slug → topic feature
```

## How the data flows (no leakage)

1. **Feature build** (`src/pipeline/data.py`): for every closed market, aggregate only the trades *before* its cutoff time into one row — trade counts, USD volume, price mean/std/min/max/last/drift, buy fraction, duration, and a `has_pre_trades` flag. Automated `*-updown` crypto markets are dropped. Output → `data/features.parquet`.
2. **Time split:** boundary `D` = the 80th percentile of `end_date`. Train = markets that *ended* before `D`; test = markets whose *cutoff* is at/after `D`; markets in flight at `D` are purged. Splitting on `end_date` (not `created_at`) stops long markets from leaking across the boundary. Validation is the most recent 15% of the training period.
3. **Topic feature:** TF-IDF + KMeans over the market `slug`, fit on the training period only, then applied to all splits and persisted for serving.
## Configuration

Everything lives in `config.yaml` (loaded with OmegaConf). Override any field from the CLI with dot-notation:

```bash
uv run python train.py hpo.n_trials=5 split.test_quantile=0.7
uv run python train.py data.cutoff_frac=0.9
```

## DVC — data versioning

The feature table is tracked with DVC and stored on Google Drive so the exact dataset version is reproducible and shareable. Only the compact `data/features.parquet` is tracked — the raw `quant.parquet`/`markets.parquet` are not.

One-time setup (already done in this repo, shown for reference):

```bash
uv run dvc init
uv run dvc add data/features.parquet      # creates data/features.parquet.dvc (committed to git)
uv run dvc remote add -d gdrive gdrive://<FOLDER_ID>
uv run dvc push                           # authenticate in browser, upload the data
```

On a fresh machine, `make train` runs `dvc pull` first, fetching `data/features.parquet` to match the current git commit.

## Project structure

```
modeling/
├── config.yaml             # data paths, split, topics, features, HPO ranges, MLflow
├── train.py                # entrypoint: Optuna HPO
├── retrain.py              # entrypoint: retrain best on train+val, export bundle
├── src/pipeline/
│   ├── config.py           # OmegaConf loader (+ CLI overrides)
│   ├── data.py             # DatasetManager (feature build, split) + TopicClusterer
│   └── trainer.py          # ModelTrainer (Optuna HPO, MLflow, registry, export)
├── data/                   # features.parquet (DVC-tracked) + its .dvc pointer
├── artifacts/              # gitignored — AUC curves, hyperparam/importance JSON
├── mlflow.db               # gitignored — MLflow tracking + registry (SQLite)
├── mlartifacts/            # gitignored — MLflow-logged models/artifacts
└── models/                 # gitignored — exported serving bundle
```