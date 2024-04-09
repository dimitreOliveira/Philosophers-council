FROM python:3.10

EXPOSE 7861

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY configs.yaml .
COPY src/ src/