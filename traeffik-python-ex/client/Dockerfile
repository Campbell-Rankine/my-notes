FROM ubuntu:22.04 AS base

ARG WD=/usr/src/app/service1
ARG PORT=8000

FROM base AS environment
WORKDIR $WD
RUN echo $PORT
RUN apt-get update && apt-get install -y build-essential cmake git

FROM environment AS python

RUN apt-get install -y python3.11 python3-pip

FROM python AS requirements
COPY ./requirements.txt $WD
RUN pip install -r requirements.txt

FROM requirements AS local
COPY . $WD