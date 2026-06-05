from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.model import ModelLoader
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    MarketFeatures,
    MarketPrediction,
    ModelInfoResponse,
)

model_loader = ModelLoader()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_loader.load()
    except Exception as exc:
        print(f"WARNING: Model failed to load at startup: {exc}")
        print("  /health will report model_loaded=false; /predict and /model-info will return 503.")
    yield


app = FastAPI(
    title="Polymarket Resolution API",
    description=(
        "Predicts whether a Polymarket market resolves YES or NO from its "
        "pre-cutoff trade summary. Serves a LightGBM model trained and "
        "registered via MLflow."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=model_loader.is_loaded)


@app.get("/model-info", response_model=ModelInfoResponse, tags=["ops"])
def model_info() -> ModelInfoResponse:
    if not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfoResponse(**model_loader.model_info)


@app.post("/predict", response_model=MarketPrediction, tags=["inference"])
def predict(features: MarketFeatures) -> MarketPrediction:
    if not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_loader.predict(features)


@app.post("/predict-batch", response_model=BatchPredictResponse, tags=["inference"])
def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    if not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return BatchPredictResponse(predictions=model_loader.predict_batch(request.markets))
