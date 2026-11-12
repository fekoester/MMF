FROM ubuntu:24.04 AS build
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential libarmadillo-dev libboost-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY makefile MMF.cpp ./
RUN make

FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libarmadillo12 && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/MMF /MMF
WORKDIR /work
ENTRYPOINT ["/MMF"]
