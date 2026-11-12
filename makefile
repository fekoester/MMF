CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall
LDLIBS = -larmadillo -llapack -lblas

.PHONY: all clean

all: MMF

MMF: MMF.cpp
	$(CXX) $(CXXFLAGS) MMF.cpp -o MMF $(LDLIBS)

clean:
	rm -f MMF
