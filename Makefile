.PHONY: install test lint run docker

install:
	pip install -e ".[dev]"

test:
	pytest --cov=app --cov-report=term-missing

lint:
	ruff check .

run:
	uvicorn app.main:app --reload

docker:
	docker compose up --build
