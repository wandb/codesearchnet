# changes to this image can be made here: https://github.com/ml-msr-github/docker-CodeSearchNet/
FROM github/csnet:cpu

RUN pip --no-cache-dir install --upgrade \
    ipdb

COPY . /
WORKDIR /src
CMD bash