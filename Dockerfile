# This base image allows setting the port with environment variable PORT,
# which is required for deployment to Google Cloud Run
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
#FROM python:3.7-slim-buster
RUN apt-get update
# required for numpy
RUN apt-get --assume-yes install build-essential
COPY requirements.txt /
RUN pip install -r /requirements.txt
COPY . /app/
WORKDIR /app