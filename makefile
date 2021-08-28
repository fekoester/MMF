CC = g++
CFLAGS = -Wall
CXXFLAGS:=-std=c++17 -g -I armadillo-9.500.2/include -I Include -I -DARMA_DONT_USE_WRAPPER -lopenblas -llapack -lm 

.PHONY: all MMF clean

MMF: MMF.cpp
	$(CC) $(CFLAGS) $(CXXFLAGS) MMF.cpp -o MMF

clean:
	rm -f MMF
