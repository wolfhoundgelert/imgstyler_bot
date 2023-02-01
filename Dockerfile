# syntax=docker/dockerfile:1
FROM python:3.10.8
COPY requirements.txt /
RUN pip install -r /requirements.txt
COPY source/ /source/
COPY token.txt /
WORKDIR /source/
CMD python imgstyler_bot.py
