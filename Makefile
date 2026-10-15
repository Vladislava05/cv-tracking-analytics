.PHONY: install test lint run docker

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

lint:
	ruff check src tests
	mypy src --ignore-missing-imports

run:
	python -m src.run --source data/sample.mp4 --output out.mp4 --show

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run src/ui/dashboard.py

docker:
	docker compose -f docker/docker-compose.yml up --build