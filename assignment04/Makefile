CXX       = g++
CXXFLAGS  = -std=c++11 -Wall -O2

all: poisson2d multigrid mg_convergence

poisson2d: poisson2d.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

multigrid: multigrid.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

mg_convergence: mg_convergence.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f poisson2d multigrid *.o

.PHONY: all clean