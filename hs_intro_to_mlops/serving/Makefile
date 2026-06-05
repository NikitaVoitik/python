.PHONY: install run test docker-build docker-run

install:
	uv sync --group dev

run:
	uv run uvicorn app.main:app --reload --port 8000

test:
	uv run pytest tests/ -v

docker-build:
	docker build -t polymarket-serving .

docker-run:
	docker run --rm -p 8000:8000 polymarket-serving
