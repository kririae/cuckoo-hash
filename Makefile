export NVCC := $(shell which nvcc)
CUDAFLAGS = --std=c++17 -I./ -O3 -g -Xptxas -O3,-v --expt-extended-lambda -lineinfo --gpu-architecture=sm_61

all: hash
hash: main.cu kernels.cuh
	@$(NVCC) $(CUDAFLAGS) -o hash main.cu

clean:
	@rm -f *.o hash