FROM python:3.8-buster
MAINTAINER Khubbatulin Mark 'Khubbatulin.mark@clover.global'

RUN apt-get update -y && apt-get install -y libsndfile1 libsndfile1-dev ffmpeg

COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip3 install -r requirements.txt
COPY ./app /app
WORKDIR /app

ENTRYPOINT [ "python" ]
CMD ["app.py"]