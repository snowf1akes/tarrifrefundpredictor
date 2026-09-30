CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O3 -march=native -fopenmp -Wall -Wextra

all: tariff-pricer

tariff-pricer: main.cpp pricer.hpp
	$(CXX) $(CXXFLAGS) main.cpp -o $@

test: test.cpp pricer.hpp
	$(CXX) $(CXXFLAGS) test.cpp -o run-test && ./run-test

clean:
	rm -f tariff-pricer run-test *.exe *.csv *.png
