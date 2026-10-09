.PHONY: help install test lint format clean train predict docs

help:
	@echo "Available commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make test        - Run tests"
	@echo "  make lint        - Run linting"
	@echo "  make format      - Format code"
	@echo "  make clean       - Clean build artifacts"
	@echo "  make train       - Train the model"
	@echo "  make predict     - Make predictions"
	@echo "  make docs        - Build documentation"

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v --cov=. --cov-report=html

lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=100 --statistics

format:
	black .
	isort .

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info

train:
	python main.py train --epochs 30

predict:
	python main.py predict --image data/sample_train/forest/forest_0000.png --visualize

docs:
	cd docs && make html
