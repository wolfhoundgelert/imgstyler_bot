# syntax=docker/dockerfile:1
FROM python:3.10.9-slim-bullseye
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
COPY token.txt .
WORKDIR /app/source
COPY source .
CMD python -m imgstyler_bot
