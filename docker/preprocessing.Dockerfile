FROM python:3.7.3
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN pip --no-cache-dir install --upgrade \
    pip \
    docopt \
    pandas

CMD ["/home/dev/script/download_and_preprocess"]
