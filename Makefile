export NVCC := $(shell which nvcc)
CUDAFLAGS = --std=c++17 -I./ -O0 -g --expt-extended-lambda

all: hash
hash: main.cu kernels.cuh
	@$(NVCC) $(CUDAFLAGS) -o hash main.cu

clean:
	@rm -f *.o hash