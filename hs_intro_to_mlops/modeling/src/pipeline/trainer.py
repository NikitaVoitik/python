from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
from mlflow import MlflowClient
from omegaconf import DictConfig
from sklearn.metrics import f1_score, log_loss, roc_auc_score

from .data import TopicClusterer


class ModelTrainer:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.features = list(cfg.features)
        self.target = cfg.target
        self.artifacts_dir = Path(cfg.data.artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.mlflow.experiment_name)

    def _xy(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        return df[self.features], df[self.target].astype(int)

    def run(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        X_train, y_train = self._xy(train_df)
        X_val, y_val = self._xy(val_df)

        ranges = self.cfg.hpo.param_ranges
        run_results: list[tuple[str, float]] = []

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", *ranges.n_estimators),
                "learning_rate": trial.suggest_float("learning_rate", *ranges.learning_rate, log=True),
                "num_leaves": trial.suggest_int("num_leaves", *ranges.num_leaves),
                "max_depth": trial.suggest_int("max_depth", *ranges.max_depth),
                "min_child_samples": trial.suggest_int("min_child_samples", *ranges.min_child_samples),
                "subsample": trial.suggest_float("subsample", *ranges.subsample),
                "colsample_bytree": trial.suggest_float("colsample_bytree", *ranges.colsample_bytree),
                "reg_lambda": trial.suggest_float("reg_lambda", *ranges.reg_lambda, log=True),
            }
            run_id, val_auc = self._train_single_run(X_train, y_train, X_val, y_val, params)
            run_results.append((run_id, val_auc))
            return val_auc

        study = optuna.create_study(
            direction=self.cfg.hpo.direction,
            sampler=optuna.samplers.TPESampler(seed=self.cfg.split.random_state),
        )
        study.optimize(objective, n_trials=int(self.cfg.hpo.n_trials))

        print(f"[trainer] HPO done — best val_roc_auc={study.best_value:.4f}")
        self._register_best_model(run_results)

    def _train_single_run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: dict,
    ) -> tuple[str, float]:
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            mlflow.log_params(params)

            model = self._build_model(params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                eval_names=["train", "val"],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            )
            mlflow.log_metric("best_iteration", model.best_iteration_ or params["n_estimators"])

            val_proba = model.predict_proba(X_val)[:, 1]
            val_pred = (val_proba >= 0.5).astype(int)
            val_roc_auc = roc_auc_score(y_val, val_proba)
            mlflow.log_metrics({
                "val_loss": log_loss(y_val, val_proba),
                "val_f1": f1_score(y_val, val_pred),
                "val_roc_auc": val_roc_auc,
            })

            self._log_auc_curve(model, run_id)
            self._log_hyperparams(params, run_id)
            mlflow.sklearn.log_model(model, artifact_path="model")
            mlflow.log_dict({"feature_names": self.features}, "feature_names.json")

        return run_id, val_roc_auc

    def _build_model(self, params: dict) -> lgb.LGBMClassifier:
        return lgb.LGBMClassifier(
            **params,
            objective="binary",
            random_state=self.cfg.split.random_state,
            n_jobs=-1,
            verbosity=-1,
        )

    def _log_auc_curve(self, model: lgb.LGBMClassifier, run_id: str) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            history = model.evals_result_
            fig, ax = plt.subplots(figsize=(8, 5))
            for split_name in ("train", "val"):
                if split_name in history and "auc" in history[split_name]:
                    ax.plot(history[split_name]["auc"], label=f"{split_name} AUC", linewidth=2)
            ax.set(xlabel="Boosting iteration", ylabel="ROC AUC",
                   title="LightGBM training curve")
            ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
            path = self.artifacts_dir / f"auc_curve_{run_id[:8]}.png"
            fig.savefig(path, dpi=100); plt.close(fig)
            mlflow.log_artifact(str(path))
        except Exception as exc:
            print(f"[trainer] (skipped AUC curve: {exc})")

    def _log_hyperparams(self, params: dict, run_id: str) -> None:
        path = self.artifacts_dir / f"hyperparams_{run_id[:8]}.json"
        with open(path, "w") as f:
            json.dump(params, f, indent=2)
        mlflow.log_artifact(str(path))

    def _log_importance(self, model: lgb.LGBMClassifier, run_id: str) -> None:
        imp = (pd.Series(model.booster_.feature_importance(importance_type="gain"),
                         index=self.features)
               .sort_values(ascending=False))
        path = self.artifacts_dir / f"feature_importance_{run_id[:8]}.json"
        with open(path, "w") as f:
            json.dump({k: float(v) for k, v in imp.items()}, f, indent=2)
        mlflow.log_artifact(str(path))
        print("[trainer] Feature importance (gain):")
        print(imp.round(0).to_string())

    def _register_best_model(self, run_results: list[tuple[str, float]]) -> None:
        best_run_id, best_roc_auc = max(run_results, key=lambda x: x[1])
        model_name = self.cfg.mlflow.model_registry_name
        print(f"[trainer] Best run {best_run_id[:8]}… val_roc_auc={best_roc_auc:.4f}; "
              f"registering '{model_name}'")
        mv = mlflow.register_model(model_uri=f"runs:/{best_run_id}/model", name=model_name)
        MlflowClient().set_registered_model_alias(model_name, "best", mv.version)
        print(f"[trainer] Registered '{model_name}' v{mv.version} @best")

    def retrain_best(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        topic_clusterer: TopicClusterer | None = None,
    ) -> None:
        params = self._fetch_best_params()
        print(f"[trainer] Retraining best params on train+val: {params}")

        trainval = pd.concat([train_df, val_df], ignore_index=True)
        X_tv, y_tv = self._xy(trainval)
        X_test, y_test = self._xy(test_df)
        model_name = self.cfg.mlflow.model_registry_name

        with mlflow.start_run(run_name="retrain_best") as run:
            run_id = run.info.run_id
            mlflow.log_params(params)
            mlflow.log_param("trained_on", "train+val")

            model = self._build_model(params)
            model.fit(X_tv, y_tv)

            test_proba = model.predict_proba(X_test)[:, 1]
            test_pred = (test_proba >= 0.5).astype(int)
            metrics = {
                "test_loss": log_loss(y_test, test_proba),
                "test_f1": f1_score(y_test, test_pred),
                "test_roc_auc": roc_auc_score(y_test, test_proba),
            }
            mlflow.log_metrics(metrics)
            print(f"[trainer] Test — roc_auc={metrics['test_roc_auc']:.4f}  "
                  f"f1={metrics['test_f1']:.4f}  loss={metrics['test_loss']:.4f}")

            self._log_hyperparams(params, run_id)
            self._log_importance(model, run_id)
            mlflow.sklearn.log_model(model, artifact_path="model")
            mlflow.log_dict({"feature_names": self.features}, "feature_names.json")

        mv = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=model_name)
        MlflowClient().set_registered_model_alias(model_name, "best", mv.version)
        print(f"[trainer] Registered '{model_name}' v{mv.version} @best")

        self._export_model(model, params, metrics, run_id, topic_clusterer)

    def _export_model(
        self,
        model: lgb.LGBMClassifier,
        params: dict,
        metrics: dict,
        run_id: str,
        topic_clusterer: TopicClusterer | None,
    ) -> None:
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        joblib.dump(model, models_dir / "model.pkl")

        n_topic_clusters = int(self.cfg.topics.n_clusters)
        if topic_clusterer is not None:
            topic_clusterer.save(models_dir / "topic_clusterer.pkl")
            n_topic_clusters = topic_clusterer.n_clusters

        info = {
            "run_id": run_id,
            "trained_on": "train+val",
            "feature_names": self.features,
            "n_topic_clusters": n_topic_clusters,
            "cutoff_frac": float(self.cfg.data.cutoff_frac),
            "params": params,
            "metrics": metrics,
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(models_dir / "model_info.json", "w") as f:
            json.dump(info, f, indent=2)
        print(f"[trainer] Exported serving bundle → {models_dir}/ "
              "(model.pkl, model_info.json, topic_clusterer.pkl)")

    def _fetch_best_params(self) -> dict:
        client = MlflowClient()
        experiment = client.get_experiment_by_name(self.cfg.mlflow.experiment_name)
        if experiment is None:
            raise RuntimeError(
                f"Experiment '{self.cfg.mlflow.experiment_name}' not found. "
                "Run 'make train' first."
            )
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.run_name != 'retrain_best'",
            order_by=["metrics.val_roc_auc DESC"],
            max_results=1,
        )
        if not runs:
            raise RuntimeError("No HPO runs found. Run 'make train' first.")
        return {k: self._coerce_param(v) for k, v in runs[0].data.params.items()}

    @staticmethod
    def _coerce_param(v: str) -> int | float | str:
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return v
