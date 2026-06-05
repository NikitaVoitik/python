from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from omegaconf import DictConfig
from pandas.api.types import CategoricalDtype
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")

_QUANT_KEEP = ["timestamp", "market_id", "price", "usd_amount", "side"]
_QUANT_DICT = ["market_id", "side"]

_FEATURE_COLS = [
    "n_trades", "pre_volume", "price_mean", "price_std", "price_min",
    "price_max", "price_last", "price_drift", "avg_trade_usd", "buy_frac",
]
_META_COLS = ["market_id", "event_id", "slug", "label_yes",
              "created_at", "end_date", "duration_hours"]


class TopicClusterer:
    def __init__(self, n_clusters: int, min_df: int, max_features: int,
                 random_state: int) -> None:
        self.n_clusters = n_clusters
        self.vectorizer = TfidfVectorizer(
            stop_words="english", min_df=min_df, max_features=max_features,
            token_pattern=r"[a-z]{3,}",
        )
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=5)

    def fit(self, slugs: pd.Series) -> "TopicClusterer":
        self.kmeans.fit(self.vectorizer.fit_transform(slugs.fillna("")))
        return self

    def transform(self, slugs: pd.Series) -> np.ndarray:
        return self.kmeans.predict(self.vectorizer.transform(slugs.fillna(""))).astype("int16")

    def save(self, path: str | Path) -> None:
        joblib.dump(
            {"vectorizer": self.vectorizer, "kmeans": self.kmeans,
             "n_clusters": self.n_clusters},
            path,
        )


class DatasetManager:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.features_path = Path(cfg.data.features_path)
        self.cutoff_frac = float(cfg.data.cutoff_frac)
        self.topic_clusterer: TopicClusterer | None = None

    def load_or_cache(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        feats = self._load_or_build_features()
        return self._split(feats)

    def _load_or_build_features(self) -> pd.DataFrame:
        if self.features_path.exists():
            print(f"[data] Loading feature table from {self.features_path}")
            return pd.read_parquet(self.features_path)

        markets_path = Path(self.cfg.data.markets_path)
        quant_path = Path(self.cfg.data.quant_path)
        if not (markets_path.exists() and quant_path.exists()):
            raise FileNotFoundError(
                f"'{self.features_path}' is missing and the raw dumps "
                f"('{markets_path}', '{quant_path}') are not present.\n"
                "Run 'dvc pull' to fetch the feature table from the remote, "
                "or point data.markets_path / data.quant_path at the raw files "
                "to rebuild it."
            )
        feats = self._build_features(markets_path, quant_path)
        self.features_path.parent.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(self.features_path, index=False)
        self._write_version(feats)
        print(f"[data] Built and cached feature table → {self.features_path} "
              f"({len(feats):,} markets)")
        return feats

    def _build_features(self, markets_path: Path, quant_path: Path) -> pd.DataFrame:
        print(f"[data] Reading markets from {markets_path}")
        markets = pd.read_parquet(markets_path)

        print(f"[data] Reading trade log from {quant_path} (columnar, dict-encoded)…")
        quant = pq.read_table(
            quant_path, columns=_QUANT_KEEP, read_dictionary=_QUANT_DICT
        ).to_pandas()
        print(f"[data] {len(quant):,} trades loaded "
              f"({quant.memory_usage(deep=True).sum() / 1e9:.1f} GB)")

        m = markets[
            (markets["closed"] == 1)
            & (markets["end_date"] > markets["created_at"])
            & markets["outcome_prices"].notna()
            & ~markets["slug"].str.contains("updown", case=False, na=False)
        ].copy()

        yes_price = m["outcome_prices"].str.extract(r"([-+]?\d*\.?\d+)", expand=False).astype(float)
        m["label_yes"] = yes_price > 0.5

        created_s = (m["created_at"] - _EPOCH) // pd.Timedelta(seconds=1)
        end_s = (m["end_date"] - _EPOCH) // pd.Timedelta(seconds=1)
        m["duration_hours"] = (end_s - created_s) // 3600
        m["cutoff_unix"] = (created_s + (end_s - created_s) * self.cutoff_frac).astype("int64")
        m = m.rename(columns={"id": "market_id"})

        cat = quant["market_id"].cat.categories
        codes = quant["market_id"].cat.codes.to_numpy()
        cutoff_by_cat = (
            pd.Series(m["cutoff_unix"].values, index=m["market_id"].values)
            .reindex(cat).to_numpy(dtype="float64")
        )
        trade_cutoff = cutoff_by_cat[np.where(codes >= 0, codes, 0)]
        trade_cutoff[codes < 0] = np.nan

        ts = quant["timestamp"].to_numpy()
        keep = ts < trade_cutoff
        usd = quant["usd_amount"].to_numpy()

        pre = pd.DataFrame({
            "code": codes[keep],
            "price": quant["price"].to_numpy()[keep],
            "usd": usd[keep],
            "ts": ts[keep],
            "buy_usd": np.where((quant["side"] == "BUY").to_numpy()[keep], usd[keep], 0.0),
        }).sort_values("ts", kind="stable")

        g = pre.groupby("code", sort=False)
        feat = g.agg(
            n_trades=("price", "size"),
            pre_volume=("usd", "sum"),
            price_mean=("price", "mean"),
            price_min=("price", "min"),
            price_max=("price", "max"),
            price_first=("price", "first"),
            price_last=("price", "last"),
            avg_trade_usd=("usd", "mean"),
            buy_usd=("buy_usd", "sum"),
        )
        feat["price_std"] = g["price"].std(ddof=0)
        feat["price_drift"] = feat["price_last"] - feat["price_first"]
        feat["buy_frac"] = feat["buy_usd"] / feat["pre_volume"].replace(0, np.nan)
        feat["market_id"] = np.asarray(cat[feat.index.to_numpy()], dtype=object)
        feat = (feat.drop(columns=["price_first", "buy_usd"])
                    .set_index("market_id")
                    .reindex(columns=_FEATURE_COLS))

        out = (m[["market_id", "event_id", "slug", "label_yes",
                  "created_at", "end_date", "duration_hours"]]
               .merge(feat, on="market_id", how="left"))

        out["has_pre_trades"] = out["n_trades"].notna()
        out[["n_trades", "pre_volume"]] = out[["n_trades", "pre_volume"]].fillna(0)
        return out

    def _split(self, feats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        cfg = self.cfg.split
        df = feats.copy()
        df["cutoff_time"] = df["created_at"] + (df["end_date"] - df["created_at"]) * self.cutoff_frac

        boundary = df["end_date"].quantile(cfg.test_quantile)
        is_train = df["end_date"] < boundary
        is_test = df["cutoff_time"] >= boundary
        purged = ~(is_train | is_test)

        order = df.loc[is_train].sort_values("end_date").index
        n_val = max(1, int(len(order) * cfg.val_frac))
        fit_idx, val_idx = order[:-n_val], order[-n_val:]

        tcfg = self.cfg.topics
        self.topic_clusterer = TopicClusterer(
            n_clusters=tcfg.n_clusters, min_df=tcfg.min_df,
            max_features=tcfg.max_features, random_state=cfg.random_state,
        ).fit(df.loc[is_train, "slug"])
        topic_dtype = CategoricalDtype(categories=range(tcfg.n_clusters))
        df["topic_cluster"] = pd.Series(
            self.topic_clusterer.transform(df["slug"]), index=df.index
        ).astype(topic_dtype)

        train_df = df.loc[fit_idx]
        val_df = df.loc[val_idx]
        test_df = df.loc[is_test]
        print(
            f"[data] Split @ {boundary:%Y-%m-%d %H:%M} (q={cfg.test_quantile}) | "
            f"train={len(train_df):,} val={len(val_df):,} test={len(test_df):,} "
            f"purged={int(purged.sum()):,} | "
            f"train YES rate={train_df['label_yes'].mean():.3f}"
        )
        return train_df, val_df, test_df

    def _write_version(self, feats: pd.DataFrame) -> None:
        with open(self.features_path.parent / "version.json", "w") as f:
            json.dump({
                "created_at": datetime.now(timezone.utc).isoformat(),
                "n_markets": int(len(feats)),
                "yes_rate": float(feats["label_yes"].mean()),
                "cutoff_frac": self.cutoff_frac,
            }, f, indent=2)
