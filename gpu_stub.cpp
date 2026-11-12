// gpu_stub.cpp - used instead of gpu_jacobi.cu for the CPU-only build (make cpu).
#include "gpu_jacobi.hpp"
#include <cstdio>
#include <cstdlib>

bool gpu_available() { return false; }
std::string gpu_name() { return "none"; }

void* host_alloc(size_t bytes, bool) {
  void* p = std::aligned_alloc(64, ((bytes + 63) / 64) * 64);
  if (!p) { std::fprintf(stderr, "out of host memory\n"); std::exit(EXIT_FAILURE); }
  return p;
}
void host_free(void* p, bool) { std::free(p); }

template <typename T>
SolveResult gpu_solve(const Grid&, const Stencil&, const SolveParams&, const GpuConfig&, T*) {
  std::fprintf(stderr, "this binary was built without CUDA\n");
  std::exit(EXIT_FAILURE);
}
template SolveResult gpu_solve<float>(const Grid&, const Stencil&, const SolveParams&, const GpuConfig&, float*);
template SolveResult gpu_solve<double>(const Grid&, const Stencil&, const SolveParams&, const GpuConfig&, double*);

void gpu_bandwidth_benchmark(size_t) { std::printf("GPU: not available (CPU-only build)\n"); }
