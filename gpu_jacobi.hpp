// gpu_jacobi.hpp - CUDA implementation (gpu_jacobi.cu) or stubs (gpu_stub.cpp, -DNO_CUDA).
#pragma once
#include "common.hpp"
#include <string>

enum class KernelKind { Generic = 0, Smem = 1 };

struct GpuConfig {
  int bx = 32, by = 8;  // thread block shape (bx * by must be a multiple of 32, <= 1024)
  KernelKind kernel = KernelKind::Generic;
};

bool gpu_available();
std::string gpu_name();

// Host memory; pinned = page-locked (cudaMallocHost) for faster transfers.
void* host_alloc(size_t bytes, bool pinned);
void host_free(void* p, bool pinned);

// Copies hU to the GPU, runs the Jacobi iteration there and copies the final iterate back
// into hU. Times: t_h2d (H2D + init of V), t_compute (all sweeps + residuals), t_d2h.
template <typename T>
SolveResult gpu_solve(const Grid& g, const Stencil& s, const SolveParams& P, const GpuConfig& cfg, T* hU);

void gpu_bandwidth_benchmark(size_t bytes);
