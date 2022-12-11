export NVCC := $(shell which nvcc)
CUDAFLAGS = --std=c++17 -I./ -O3 -g -Xptxas -O3 -extended-lambda --expt-extended-lambda -lineinfo --gpu-architecture=sm_61

all: hash
hash: main.cu kernels.cuh gen.cuh
	@$(NVCC) $(CUDAFLAGS) -o hash main.cu

test: tests.cu kernels.cuh gen.cuh
	@$(NVCC) $(CUDAFLAGS) -o tests tests.cu

clean:
	@rm -f *.o hash