FEATURE_NAME := council
TAG ?= latest

chatbot:
	docker run --rm \
  	-p 7861:7861 \
	-v $(PWD)/prompts/:/app/prompts/ \
	-v $(PWD)/.env/:/app/.env/ \
	-v $(HOME)/.cache/huggingface/:/root/.cache/huggingface/ \
	${FEATURE_NAME}:${TAG} gradio src/app.py

build:
	docker build -t ${FEATURE_NAME}:${TAG} .

lint:
	isort ./src
	black ./src
	flake8 ./src
	mypy --ignore-missing-imports ./src