from src.pipeline.config import load_config
from src.pipeline.data import DatasetManager
from src.pipeline.trainer import ModelTrainer


def main() -> None:
    cfg = load_config("config.yaml")
    dm = DatasetManager(cfg)
    train_df, val_df, test_df = dm.load_or_cache()
    trainer = ModelTrainer(cfg)
    trainer.retrain_best(train_df, val_df, test_df, topic_clusterer=dm.topic_clusterer)


if __name__ == "__main__":
    main()
