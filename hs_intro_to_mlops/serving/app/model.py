import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from app.schemas import MarketFeatures, MarketPrediction

_NUMERIC_FEATURES = [
    "duration_hours", "n_trades", "pre_volume", "price_mean", "price_std",
    "price_min", "price_max", "price_last", "price_drift", "avg_trade_usd",
    "buy_frac",
]


class ModelLoader:
    def __init__(self) -> None:
        self.model: Any = None
        self.feature_names: list[str] = []
        self.n_topic_clusters: int = 0
        self.topic: dict | None = None
        self.model_info: dict = {}

    def load(self) -> None:
        models_dir = Path(os.getenv("MODELS_DIR", "models"))
        model_path = models_dir / "model.pkl"
        info_path = models_dir / "model_info.json"
        topic_path = models_dir / "topic_clusterer.pkl"

        print(f"[model] Loading model from {model_path}…")
        self.model = joblib.load(model_path)

        with open(info_path) as f:
            info = json.load(f)
        self.feature_names = info["feature_names"]
        self.n_topic_clusters = int(info.get("n_topic_clusters", 0))
        self.model_info = {
            "run_id": info.get("run_id", ""),
            "trained_on": info.get("trained_on", ""),
            "params": info.get("params", {}),
            "metrics": info.get("metrics", {}),
            "exported_at": info.get("exported_at", ""),
        }

        if topic_path.exists():
            self.topic = joblib.load(topic_path)
            print("[model] Loaded topic clusterer.")
        else:
            print("[model] No topic clusterer found; topic_cluster defaults to 0.")

        print(f"[model] Loaded. Export timestamp: {self.model_info['exported_at']}")

    def predict(self, features: MarketFeatures) -> MarketPrediction:
        df = self._to_dataframe([features])
        proba = float(self.model.predict_proba(df)[0][1])
        return self._to_prediction(proba)

    def predict_batch(self, items: list[MarketFeatures]) -> list[MarketPrediction]:
        df = self._to_dataframe(items)
        probas = self.model.predict_proba(df)[:, 1]
        return [self._to_prediction(float(p)) for p in probas]

    @staticmethod
    def _to_prediction(proba: float) -> MarketPrediction:
        prediction = int(proba >= 0.5)
        return MarketPrediction(
            prediction=prediction,
            probability=round(proba, 4),
            label="YES" if prediction == 1 else "NO",
        )

    def _to_dataframe(self, items: list[MarketFeatures]) -> pd.DataFrame:
        rows = []
        slugs = []
        for f in items:
            row = {name: getattr(f, name) for name in _NUMERIC_FEATURES}
            for k, v in row.items():
                if v is None:
                    row[k] = np.nan
            row["has_pre_trades"] = f.n_trades > 0
            rows.append(row)
            slugs.append(f.slug or "")

        df = pd.DataFrame(rows)
        df["topic_cluster"] = self._assign_topics(slugs)
        return df[self.feature_names]

    def _assign_topics(self, slugs: list[str]) -> pd.Series:
        n = self.n_topic_clusters or 1
        if self.topic is not None:
            codes = self.topic["kmeans"].predict(
                self.topic["vectorizer"].transform(slugs)
            ).astype("int16")
        else:
            codes = np.zeros(len(slugs), dtype="int16")
        dtype = CategoricalDtype(categories=range(n))
        return pd.Series(codes, dtype=dtype)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None
