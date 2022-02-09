FROM ubuntu:latest as build
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential liblapack-dev libblas-dev libboost-dev libarmadillo-dev
ADD makefile /
ADD MMF.cpp /
RUN make

FROM ubuntu:latest
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y libarmadillo9
COPY --from=build /MMF /

ENTRYPOINT ["/MMF"]