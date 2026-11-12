# Project 1: Jacobi sweeps with CUDA
#
#   make            -> ./jacobi      (CPU + GPU, needs nvcc)
#   make cpu        -> ./jacobi_cpu  (CPU only, no CUDA needed)
#   make test       -> build and run the correctness tests (double and float)
#
# Set the GPU architecture if your nvcc does not support -arch=native (CUDA < 11.5), e.g.
#   make CUDA_ARCH=-arch=sm_80      (A100)   sm_70 (V100), sm_86 (RTX 30xx), sm_90 (H100)

CXX       ?= g++
NVCC      ?= nvcc
CUDA_ARCH ?= -arch=native

CXXFLAGS  ?= -O3 -march=native -std=c++17 -fopenmp -Wall -Wextra
NVCCFLAGS ?= -O3 -std=c++17 $(CUDA_ARCH) -lineinfo -Xcompiler "-O3 -march=native -fopenmp -Wall"

SRC  = src
OBJ  = build
HDRS = $(wildcard $(SRC)/*.hpp)

all: jacobi

jacobi: $(OBJ)/main.o $(OBJ)/stencil.o $(OBJ)/cpu_jacobi.o $(OBJ)/gpu_jacobi.o
	$(NVCC) $(CUDA_ARCH) $^ -o $@ -Xcompiler -fopenmp -lgomp

jacobi_cpu: $(OBJ)/main.o $(OBJ)/stencil.o $(OBJ)/cpu_jacobi.o $(OBJ)/gpu_stub.o
	$(CXX) $(CXXFLAGS) $^ -o $@

cpu: jacobi_cpu

$(OBJ)/%.o: $(SRC)/%.cpp $(HDRS) | $(OBJ)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ)/gpu_jacobi.o: $(SRC)/gpu_jacobi.cu $(HDRS) | $(OBJ)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ):
	mkdir -p $(OBJ)

test: jacobi
	./jacobi --test --prec double
	./jacobi --test --prec float

test-cpu: jacobi_cpu
	./jacobi_cpu --test --prec double
	./jacobi_cpu --test --prec float

clean:
	rm -rf $(OBJ) jacobi jacobi_cpu

.PHONY: all cpu test test-cpu clean
