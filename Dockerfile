FROM ubuntu:18.04
MAINTAINER Khubbatulin Mark 'Khubbatulin.mark@clover.global'

RUN apt-get update -y && \
    apt-get install -y python3.6 python3-pip

RUN apt-get install -y libsndfile1
RUN apt-get install -y libsndfile1-dev
RUN python3 -m pip install --upgrade pip

COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip3 install -r requirements.txt
COPY . /app
WORKDIR /app/app

# ENTRYPOINT [ "python3" ] # чтобы зайти по ssh

# CMD ["app.py"]