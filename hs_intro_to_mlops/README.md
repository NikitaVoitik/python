# Polymarket Resolution — MLOps

Deployed training + serving stack for predicting whether a [Polymarket](https://polymarket.com)
market resolves **YES** or **NO**, from the trade history available up to a
cutoff point (default 80%) of the market's life. Binary classification; the
metric is **ROC AUC** (markets are ranked so the most confident ones can be bet on).

Adapted from the [hs_intro_to_mlops](https://github.com/aaaksenova/hs_intro_to_mlops)
template.

## Structure

```
modeling/   — LightGBM training pipeline: Optuna HPO, MLflow tracking, DVC data versioning
serving/    — FastAPI inference service with typed Pydantic schemas and Docker support
```

### `modeling/`

Config-driven, modular training pipeline.

- Builds a leak-free per-market feature table from the raw Polymarket dumps
  (`markets.parquet` + the ~27 GB `quant.parquet` trade log)
- Time-based train/val/test split on `end_date` with purging of in-flight markets
- Optuna HPO; every trial logged to a local MLflow experiment, best run registered `@best`
- Retrain step combines train+val, evaluates on held-out test, exports a serving bundle
- Feature table versioned with DVC on a Google Drive remote

See [`modeling/README.md`](modeling/README.md).

### `serving/`

FastAPI app that loads the exported bundle and serves predictions.

- `GET /health` — liveness check
- `GET /model-info` — model metadata and test metrics
- `POST /predict` — single-market prediction
- `POST /predict-batch` — batch prediction

See [`serving/README.md`](serving/README.md).

## Quick start (Docker Compose)

Docker Compose is the recommended way to run the stack. It brings up the API and
the MLflow UI together from the repo root:

```bash
docker compose up --build
```

- API + Swagger: [http://localhost:8000/docs](http://localhost:8000/docs)
- MLflow experiments: [http://localhost:5050](http://localhost:5050)

The `api` service serves the trained model (mounted read-only from
`modeling/models/`); the `mlflow` service serves `modeling/mlflow.db`.

This needs a model bundle in `modeling/models/` first. If you have DVC access,
pull the versioned feature table and train once:

```bash
cd modeling && make install && make all   # dvc pull → HPO → retrain → export models/
cd ..
docker compose up --build
```

Training itself is not containerised — it needs the raw data / interactive DVC
auth, so run it via `modeling/`'s `make` targets before bringing the stack up.

## Run without Docker

If you'd rather run the pieces directly:

```bash
cd modeling && make install && make all   # dvc pull → HPO → retrain → export models/
cd ../serving && make install && cp -r ../modeling/models ./models && make run
```

## Data & DVC

The raw dumps are large and **not** committed. Only the compact, leak-free
`modeling/data/features.parquet` is tracked with DVC and stored on Google Drive
(shared with **aaaksenova2@gmail.com**), so the pipeline reproduces on any
machine via `dvc pull` without the 27 GB trade log. See
[`modeling/README.md`](modeling/README.md#dvc--data-versioning).
