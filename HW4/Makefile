CXXFLAGS = -O3 -I./include/

EXECUTABLES = spmv-cuda-opt

all: $(EXECUTABLES)

spmv-cuda-opt: spmv-cuda.cu mmio.o config.h include/*.h
	# Compile with nvcc
	nvcc $(CXXFLAGS) -o $@ spmv-cuda-opt.cu mmio.o

mmio.o: mmio.cpp include/mmio.h
	# Compile with g++, the GNU C++ compiler
	g++ $(CXXFLAGS) -o $@ -c $<

.PHONY: clean
clean: 
	rm -f *.o $(EXECUTABLES) test*