-include .env
export

DATA_DIR ?= $(PWD)/data
DEV_PORT ?= 8001

.PHONY: dev prod prod-build prod-logs test cli-help

dev:
	DATA_DIR='$(DATA_DIR)' uvicorn 'web.app:create_app' --factory --reload --port $(DEV_PORT)

prod:
	docker compose up -d

prod-build:
	docker compose up --build -d

prod-logs:
	docker compose logs -f app

test:
	PYTHONPATH=. .venv/bin/python -m pytest -q

cli-help:
	python -m cli --help
