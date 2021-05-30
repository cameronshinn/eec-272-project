NVCC = nvcc
CXX = g++
STD = c++11
CUDA_INCLUDE = /usr/local/cuda/include
CUDA_LIB = /usr/local/cuda/lib64

prunedense2csrbyP :
	$(NVCC) -c -I$(CUDA_INCLUDE) --std=$(STD) $@.cpp
	$(CXX) -o $@.cpp $@.o -L$(CUDA_LIB) -lcusparse -lcudart
