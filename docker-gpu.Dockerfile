FROM github/codenet

# Copy all code into the container
COPY . /
WORKDIR /src
CMD bash
