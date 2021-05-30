NVCC = nvcc
CXX = g++
CUDA_INCLUDE = /usr/local/cuda-11.0/include
CUDA_LIB = /usr/local/cuda-11.0/lib64

prunedense2csrbyP :
	$(NVCC) -c -I$(CUDA_INCLUDE) $@.cpp
	$(CXX) -o $@ $@.o -L$(CUDA_LIB) -lcusparse -lcudart

clean:
	rm -f prunedense2csrbyP prunedense2csrbyP.o
