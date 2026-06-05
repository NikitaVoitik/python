# Polymarket Resolution — Serving

FastAPI application that loads the exported LightGBM bundle and serves YES/NO
resolution predictions over HTTP, with typed Pydantic request/response schemas
and interactive Swagger docs.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Docker (optional)
- The model bundle from `modeling/` (`make retrain` writes `modeling/models/`)

## Setup

```bash
make install
cp -r ../modeling/models ./models
```

`models/` must contain `model.pkl`, `model_info.json`, and `topic_clusterer.pkl`.

## Run locally

```bash
make run
```

- API: [http://localhost:8000](http://localhost:8000)
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs) — every
  endpoint has a pre-filled example you can "Try it out" directly.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Liveness check; reports whether the model loaded |
| `GET`  | `/model-info` | Model metadata (params, test metrics, export timestamp) |
| `POST` | `/predict` | Predict resolution for a single market |
| `POST` | `/predict-batch` | Predict for a list of markets |

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @examples/input_example.json
```

### Example response

```json
{
  "prediction": 1,
  "probability": 0.9134,
  "label": "YES"
}
```

Markets with no early trades are valid input — send `null` for the `price_*`
fields (see `examples/batch_example.json`).

## Run tests

```bash
make test
```

Tests mock all file I/O — no model files required.

## Docker

```bash
cp -r ../modeling/models ./models
make docker-build
make docker-run
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELS_DIR` | `models` | Directory holding `model.pkl`, `model_info.json`, `topic_clusterer.pkl` |

## Project structure

```
serving/
├── app/
│   ├── main.py       # FastAPI app, lifespan, four endpoints
│   ├── model.py      # ModelLoader: load bundle, derive topic, predict
│   └── schemas.py    # Pydantic request/response models
├── tests/
│   └── test_api.py   # Unit tests with mocked file I/O
├── examples/
│   ├── input_example.json
│   └── batch_example.json
├── models/           # model bundle (copy from ../modeling/models)
└── Dockerfile
```
