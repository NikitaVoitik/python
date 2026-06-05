from pydantic import BaseModel, Field


class MarketFeatures(BaseModel):
    duration_hours: float = Field(..., ge=0, description="Market lifespan in hours",
                                  examples=[17.0])
    n_trades: int = Field(..., ge=0, description="Number of trades before the cutoff",
                          examples=[41])
    pre_volume: float = Field(..., ge=0, description="Total USD traded before the cutoff",
                              examples=[1306.09])
    price_mean: float | None = Field(None, ge=0, le=1, examples=[0.6009])
    price_std: float | None = Field(None, ge=0, examples=[0.2719])
    price_min: float | None = Field(None, ge=0, le=1, examples=[0.36])
    price_max: float | None = Field(None, ge=0, le=1, examples=[0.999])
    price_last: float | None = Field(
        None, ge=0, le=1, description="Most recent trade price before the cutoff",
        examples=[0.999],
    )
    price_drift: float | None = Field(
        None, description="price_last - price_first over the pre-cutoff window",
        examples=[0.619],
    )
    avg_trade_usd: float | None = Field(None, ge=0, examples=[31.85])
    buy_frac: float | None = Field(
        None, ge=0, le=1, description="Share of pre-cutoff volume that was BUYs",
        examples=[0.0979],
    )
    slug: str | None = Field(
        None, description="Market slug; used to assign the topic-cluster feature",
        examples=["val-bo-rose-2026-03-10"],
    )


class MarketPrediction(BaseModel):
    prediction: int = Field(..., description="0 = NO, 1 = YES")
    probability: float = Field(..., description="P(resolves YES)")
    label: str = Field(..., description='"YES" or "NO"')


class BatchPredictRequest(BaseModel):
    markets: list[MarketFeatures]


class BatchPredictResponse(BaseModel):
    predictions: list[MarketPrediction]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    run_id: str
    trained_on: str
    params: dict[str, float | int]
    metrics: dict[str, float]
    exported_at: str
