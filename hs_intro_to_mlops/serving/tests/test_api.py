import json
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

VALID_PAYLOAD = {
    "duration_hours": 17.0,
    "n_trades": 41,
    "pre_volume": 1306.09,
    "price_mean": 0.6009,
    "price_std": 0.2719,
    "price_min": 0.36,
    "price_max": 0.999,
    "price_last": 0.999,
    "price_drift": 0.619,
    "avg_trade_usd": 31.85,
    "buy_frac": 0.0979,
    "slug": "val-bo-rose-2026-03-10",
}

MOCK_PROBA = [[0.05, 0.95], [0.7, 0.3]]
MOCK_FEATURE_NAMES = [
    "duration_hours", "has_pre_trades", "n_trades", "pre_volume", "price_mean",
    "price_std", "price_min", "price_max", "price_last", "price_drift",
    "avg_trade_usd", "buy_frac", "topic_cluster",
]
MOCK_MODEL_INFO = {
    "run_id": "abc1234567890",
    "trained_on": "train+val",
    "feature_names": MOCK_FEATURE_NAMES,
    "n_topic_clusters": 24,
    "params": {"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 31},
    "metrics": {"test_f1": 0.62, "test_roc_auc": 0.89, "test_loss": 0.41},
    "exported_at": "2026-01-01T00:00:00+00:00",
}


def _make_mock_model():
    m = MagicMock()
    m.predict_proba.side_effect = lambda df: np.array(MOCK_PROBA[: len(df)])
    return m


@pytest.fixture(scope="module")
def client():
    mock_model = _make_mock_model()
    with (
        patch("app.model.joblib.load", return_value=mock_model),
        patch("builtins.open", mock_open(read_data=json.dumps(MOCK_MODEL_INFO))),
        patch("app.model.Path.exists", return_value=False),
    ):
        from app.main import app
        with TestClient(app) as c:
            yield c


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info_schema(client):
    resp = client.get("/model-info")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"run_id", "trained_on", "params", "metrics", "exported_at"}
    assert isinstance(body["metrics"], dict)


def test_predict_valid(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] == 1
    assert body["label"] == "YES"
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_no_trade_market(client):
    payload = {
        "duration_hours": 15.0, "n_trades": 0, "pre_volume": 0.0,
        "price_mean": None, "price_std": None, "price_min": None,
        "price_max": None, "price_last": None, "price_drift": None,
        "avg_trade_usd": None, "buy_frac": None, "slug": "btc-march-price",
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    assert resp.json()["label"] in ("YES", "NO")


def test_predict_batch(client):
    payload = {"markets": [VALID_PAYLOAD, {**VALID_PAYLOAD, "n_trades": 0}]}
    resp = client.post("/predict-batch", json=payload)
    assert resp.status_code == 200
    preds = resp.json()["predictions"]
    assert len(preds) == 2
    assert preds[0]["label"] == "YES"
    assert preds[1]["label"] == "NO"


def test_predict_missing_field(client):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "n_trades"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_out_of_range_price(client):
    payload = {**VALID_PAYLOAD, "price_last": 1.5}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
