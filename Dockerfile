FROM python:3
ENV PYTHONUNBUFFERED=1

WORKDIR /code

RUN mkdir /code/geometry
RUN mkdir /code/simulations

RUN apt-get update
RUN apt-get --assume-yes install awscli
RUN apt install gcc git


COPY requirements.txt /code/

RUN pip install -r requirements.txt
RUN git clone https://github.com/gprMax/gprMax.git
RUN cd gprMax && python3 setup.py build
RUN cd gprMax && python3 setup.py install

COPY gprutil /code/gprutil
COPY build_sim.py /code/
COPY geometry_spec.csv /code/
CMD python3 build_sim.py
