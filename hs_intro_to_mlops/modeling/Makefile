.PHONY: install pull train retrain all mlflow-ui clean

install:
	uv sync

# Fetch the DVC-tracked feature table (data/features.parquet) from the GDrive remote.
pull:
	uv run dvc pull

train: pull
	uv run python train.py

retrain: pull
	uv run python retrain.py

# Full pipeline: HPO search, then retrain the best on train+val and export the bundle.
all: train retrain

mlflow-ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

clean:
	rm -rf artifacts/ mlruns/ mlartifacts/ mlflow.db models/model.pkl models/model_info.json models/topic_clusterer.pkl
