FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04
LABEL maintainer="Uladzimir Kazakevich"

RUN apt-get update \
    && apt-get -y upgrade \
    && apt-get install -y \
    git \
    wget \
    unzip \
    libopencv-dev \
    python-pip

RUN git clone --branch master --depth 1 https://github.com/AlexeyAB/darknet.git

WORKDIR /darknet
RUN sed -i "s/OPENCV=0/OPENCV=1/g" Makefile \
    && sed -i 's/GPU=0/GPU=1/g' Makefile \
    && sed -i 's/CUDNN=0/CUDNN=1/g' Makefile \
    && sed -i 's/LIBSO=0/LIBSO=1/g' Makefile
RUN make

RUN pip install 'matplotlib==2.2.2'
RUN pip install opencv-python numpy scikit-image paho-mqtt pyyaml requests urllib3 pytz pygtail

