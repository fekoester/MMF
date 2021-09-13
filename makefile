CC = g++
CFLAGS = -Wall
CXXFLAGS:=-std=c++17 -O2 

.PHONY: all MMF clean

MMF: MMF.cpp
	$(CC) $(CFLAGS) $(CXXFLAGS) MMF.cpp -o MMF -larmadillo -llapack -lblas

clean:
	rm -f MMF
