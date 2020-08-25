# Dockerfile
FROM python:3.8.0

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip

WORKDIR /usr/app

COPY requirements.txt ${PWD}
RUN pip install --no-cache-dir -r requirements.txt