FROM ubuntu:22.04 AS base

ARG WD=/usr/src/app/service1
ARG PORT=8000
ARG DEBIAN_FRONTEND=noninteractive
USER 0

FROM base AS environment
WORKDIR $WD

# python environment vars
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN echo $PORT
RUN apt-get update && apt-get install -y build-essential cmake git
# RUN apt install -y nginx
RUN apt-get install -y python3.11 python3-pip

FROM environment AS nginx
# COPY ./nginx/service1.conf /etc/nginx/sites-available/service1.conf
# RUN ln -s /etc/nginx/sites-available/service1.conf  /etc/nginx/sites-enabled/
# RUN echo "COPY NGINX config"
# RUN nginx -t

FROM nginx AS requirements
COPY ./requirements.txt $WD
RUN pip install -r requirements.txt

COPY . $WD
EXPOSE $PORT

# Deprecated as gunicorn only exists as a way to manage processes on uvicorn
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-c", "gunicorn.conf.py", "app:app"]