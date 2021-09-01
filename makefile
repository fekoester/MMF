CC = g++
CFLAGS = -Wall
CXXFLAGS:=-std=c++17 -O2 -DARMA_DONT_USE_WRAPPER

.PHONY: all MMF clean

MMF: MMF.cpp
	$(CC) $(CFLAGS) $(CXXFLAGS) MMF.cpp -o MMF

clean:
	rm -f MMF
