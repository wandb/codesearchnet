# Original container defined at: https://github.com/ml-msr-github/docker-codenet
FROM github/codenet

# Copy all code into the container
COPY . /
WORKDIR /src
CMD bash
