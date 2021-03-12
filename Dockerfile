FROM python:3
ENV PYTHONUNBUFFERED=1

WORKDIR /code

RUN apt install gcc

COPY requirements.txt /code/

RUN pip install -r requirements.txt
RUN wget -P /lib/
COPY raincloud /code/