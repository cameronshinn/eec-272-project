NVCC = nvcc
CXX = g++
CUDA_INCLUDE = /usr/local/cuda-11.0/include
CUDA_LIB = /usr/local/cuda-11.0/lib64

spmv_csr_example :
	$(NVCC) -c -I$(CUDA_INCLUDE) $@.cpp
	$(CXX) -o $@ $@.o -L$(CUDA_LIB) -lcusparse -lcudart

clean:
	rm -f spmv_csr_example
