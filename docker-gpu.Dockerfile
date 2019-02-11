FROM tensorflow/tensorflow:latest-gpu-py3
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN add-apt-repository -y ppa:git-core/ppa
RUN add-apt-repository -y ppa:jonathonf/python-3.6

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    byobu \
    ca-certificates \
    git-core git \
    htop \
    libglib2.0-0 \
    libjpeg-dev \
    libpng-dev \
    libxext6 \
    libsm6 \
    libxrender1 \
    libcupti-dev \
    openssh-server \
    python3.6 \
    python3.6-dev \
    software-properties-common \
    vim \
    unzip \
    && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*

RUN apt-get -y update

#  Setup Python 3.6 (Need for other dependencies)
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.5 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 2
RUN apt-get install -y python3-setuptools
RUN easy_install pip
RUN pip install --upgrade pip

# Pin TF Version on v1.12.0
RUN pip --no-cache-dir install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.12.0-cp36-cp36m-linux_x86_64.whl

# Other python packages
RUN pip --no-cache-dir install --upgrade \
    altair \
    docopt \
    dpu_utils \
    ipdb \
    jsonpath_rw_ext \
    jupyter \
    more_itertools \
    numpy \
    pandas \
    parso \
    pygments \
    requests \
    scipy \
    SetSimilaritySearch \
    tqdm \
    typed_ast \
    wandb \
    wget

ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Open Ports for TensorBoard, Jupyter, and SSH
EXPOSE 6006
EXPOSE 7654
EXPOSE 22

# Copy all code into the container
COPY . /
WORKDIR /src
CMD bash