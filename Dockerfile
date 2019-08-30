FROM python:3.7-slim-buster
RUN apt-get update
# required for numpy
RUN apt-get --assume-yes install build-essential
COPY requirements.txt /
RUN pip install -r /requirements.txt
COPY . /app
WORKDIR /app
# default port for the dash development server
EXPOSE 8050
#CMD ["gunicorn", "-w 4", "main:server"]
CMD ["python", "main.py"]