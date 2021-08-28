CC = g++
CFLAGS = -Wall
CXXFLAGS:=-std=c++17 -g -I armadillo-9.500.2/include -I Include -I -DARMA_DONT_USE_WRAPPER -lopenblas -llapack -lm 

.PHONY: all solver clean

solver: main.cpp
	$(CC) $(CFLAGS) $(CXXFLAGS) main.cpp -o solver

clean:
	rm -f solver
