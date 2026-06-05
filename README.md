# python

Personal repo. The main project lives in [`hs_intro_to_mlops/`](hs_intro_to_mlops/) —
an end-to-end MLOps stack (LightGBM training + FastAPI serving) for predicting
Polymarket market resolutions.

## Quick start

**1. Train the model** (requires DVC access to the feature data). This produces
the model bundle in `hs_intro_to_mlops/modeling/models/`:

```bash
cd hs_intro_to_mlops/modeling && make install && make all
```

**2. Run the serving stack** (API + MLflow UI) with Docker Compose:

```bash
cd hs_intro_to_mlops
docker compose up --build
```

- API + Swagger: http://localhost:8000/docs
- MLflow UI: http://localhost:5050
