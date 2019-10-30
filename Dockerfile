# This base image allows setting the port with environment variable PORT,
# which is required for deployment to Google Cloud Run
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
RUN apt-get update
# required for numpy
RUN apt-get --assume-yes install build-essential
COPY requirements.txt /
RUN pip install -r /requirements.txt
COPY . /app/
WORKDIR /app/foi_semantic_search/
ENV APP_MODULE="foi_semantic_search.main:app"
