// gpu_jacobi.cu - CUDA Jacobi sweeps for general stencils, with a residual reduction.
#include "gpu_jacobi.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#define CUDA_CHECK(call)                                                              \
  do {                                                                                \
    cudaError_t err_ = (call);                                                        \
    if (err_ != cudaSuccess) {                                                        \
      std::fprintf(stderr, "CUDA error '%s' at %s:%d: %s\n", #call, __FILE__, __LINE__, \
                   cudaGetErrorString(err_));                                         \
      std::exit(EXIT_FAILURE);                                                        \
    }                                                                                 \
  } while (0)

// ---------------------------------------------------------------------------------
// Stencil in constant memory. All threads of a warp read the same entry p at the same
// time, so each access is a broadcast from the constant cache.
// ---------------------------------------------------------------------------------
__constant__ int c_goff[MAX_POINTS];      // offsets in the global array: di * ld + dj
__constant__ int c_toff[MAX_POINTS];      // offsets in the shared-memory tile: di * tw + dj
__constant__ double c_coef_d[MAX_POINTS];
__constant__ float c_coef_f[MAX_POINTS];

template <typename T> __device__ __forceinline__ T coef(int p);
template <> __device__ __forceinline__ double coef<double>(int p) { return c_coef_d[p]; }
template <> __device__ __forceinline__ float coef<float>(int p) { return c_coef_f[p]; }

// ---------------------------------------------------------------------------------
// Reduction building blocks. norm == 0: max (values are already |SU| >= 0, so 0 is the
// neutral element), norm == 1: sum (of (SU)^2).
// ---------------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ T combine(T a, T b, int norm) {
  return norm == 0 ? (a > b ? a : b) : a + b;
}

template <typename T>
__device__ __forceinline__ T warp_reduce(T v, int norm) {
  for (int o = 16; o > 0; o >>= 1) v = combine(v, __shfl_down_sync(0xffffffffu, v, o), norm);
  return v;
}

// Reduces one value per thread over the whole (1D or 2D) block and writes the result to
// out[linear block index]. Requires blockDim.x * blockDim.y to be a multiple of 32.
// Must be called by all threads of the block.
template <typename T>
__device__ void block_reduce_store(T v, T* out, int norm) {
  __shared__ T warp_vals[32];
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int nthreads = blockDim.x * blockDim.y;
  const int lane = tid & 31, wid = tid >> 5;
  v = warp_reduce(v, norm);
  if (lane == 0) warp_vals[wid] = v;
  __syncthreads();
  if (wid == 0) {
    v = (lane < (nthreads >> 5)) ? warp_vals[lane] : T(0);
    v = warp_reduce(v, norm);
    if (lane == 0) out[blockIdx.y * gridDim.x + blockIdx.x] = v;
  }
}

// Second stage: one block reduces the per-block partial results.
template <typename T>
__global__ void reduce_partials(const T* __restrict__ in, int m, T* __restrict__ out, int norm) {
  T v = T(0);
  for (int k = threadIdx.x; k < m; k += blockDim.x) v = combine(v, in[k], norm);
  block_reduce_store<T>(v, out, norm);
}

// ---------------------------------------------------------------------------------
// Sweep kernels: one thread per interior point, 2D grid of 2D blocks, arbitrary n.
// V = U - (S U) / sc.  If RES, the block also reduces |SU| (or (SU)^2) of the input U
// into partial[block], i.e. the residual comes for free with the sweep.
// ---------------------------------------------------------------------------------
template <typename T, bool RES>
__global__ void sweep_generic(const T* __restrict__ U, T* __restrict__ V, int n, int R, int ld,
                              int np, T sc, T inv_sc, T* __restrict__ partial, int norm) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;  // x: contiguous -> coalesced
  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  T r = T(0);
  if (i < n && j < n) {
    const long long idx = static_cast<long long>(i + R) * ld + (j + R);
    const T u = U[idx];
    T su = sc * u;
#pragma unroll 4
    for (int p = 0; p < np; ++p) su += coef<T>(p) * U[idx + c_goff[p]];
    V[idx] = u - su * inv_sc;
    if (RES) r = (norm == 0) ? (su < T(0) ? -su : su) : su * su;
  }
  if (RES) block_reduce_store<T>(r, partial, norm);
}

// Same, but the block first loads its tile of U including a halo of width R into shared
// memory, so each value of U is read from global memory (about) once per block.
template <typename T, bool RES>
__global__ void sweep_smem(const T* __restrict__ U, T* __restrict__ V, int n, int R, int ld,
                           int np, T sc, T inv_sc, T* __restrict__ partial, int norm) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  T* tile = reinterpret_cast<T*>(smem_raw);
  const int bx = blockDim.x, by = blockDim.y;
  const int tw = bx + 2 * R, th = by + 2 * R;
  // Tile entry (ti, tj) is array entry (i0 + ti, j0 + tj); interior point (i, j) is array
  // entry (i + R, j + R), i.e. tile entry (threadIdx.y + R, threadIdx.x + R).
  const int i0 = blockIdx.y * by, j0 = blockIdx.x * bx;
  for (int ti = threadIdx.y; ti < th; ti += by) {
    const int ai = i0 + ti;
    for (int tj = threadIdx.x; tj < tw; tj += bx) {
      const int aj = j0 + tj;
      tile[ti * tw + tj] = (ai < ld && aj < ld) ? U[static_cast<long long>(ai) * ld + aj] : T(0);
    }
  }
  __syncthreads();

  const int i = i0 + threadIdx.y, j = j0 + threadIdx.x;
  T r = T(0);
  if (i < n && j < n) {
    const int c = (threadIdx.y + R) * tw + threadIdx.x + R;
    const T u = tile[c];
    T su = sc * u;
#pragma unroll 4
    for (int p = 0; p < np; ++p) su += coef<T>(p) * tile[c + c_toff[p]];
    V[static_cast<long long>(i + R) * ld + (j + R)] = u - su * inv_sc;
    if (RES) r = (norm == 0) ? (su < T(0) ? -su : su) : su * su;
  }
  if (RES) block_reduce_store<T>(r, partial, norm);
}

template <typename T>
static void launch_sweep(bool smem, dim3 grid, dim3 block, size_t smem_bytes, const T* U, T* V,
                         const Grid& g, int np, T sc, T inv, bool res, T* part, int norm) {
  if (!smem) {
    if (res) sweep_generic<T, true><<<grid, block>>>(U, V, g.n, g.R, g.ld, np, sc, inv, part, norm);
    else     sweep_generic<T, false><<<grid, block>>>(U, V, g.n, g.R, g.ld, np, sc, inv, part, norm);
  } else {
    if (res) sweep_smem<T, true><<<grid, block, smem_bytes>>>(U, V, g.n, g.R, g.ld, np, sc, inv, part, norm);
    else     sweep_smem<T, false><<<grid, block, smem_bytes>>>(U, V, g.n, g.R, g.ld, np, sc, inv, part, norm);
  }
  CUDA_CHECK(cudaGetLastError());
}

static void upload_stencil(const Grid& g, const Stencil& s, int tw) {
  const int np = s.np();
  if (np == 0) return;
  std::vector<int> goff(np), toff(np);
  std::vector<double> cd(np);
  std::vector<float> cf(np);
  for (int p = 0; p < np; ++p) {
    goff[p] = s.di[p] * g.ld + s.dj[p];
    toff[p] = s.di[p] * tw + s.dj[p];
    cd[p] = s.c[p];
    cf[p] = static_cast<float>(s.c[p]);
  }
  CUDA_CHECK(cudaMemcpyToSymbol(c_goff, goff.data(), np * sizeof(int)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_toff, toff.data(), np * sizeof(int)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_coef_d, cd.data(), np * sizeof(double)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_coef_f, cf.data(), np * sizeof(float)));
}

template <typename T>
SolveResult gpu_solve(const Grid& g, const Stencil& s, const SolveParams& P, const GpuConfig& cfg, T* hU) {
  SolveResult out;
  const int bx = cfg.bx, by = cfg.by;
  if (bx <= 0 || by <= 0 || (bx * by) % 32 != 0 || bx * by > 1024) {
    std::fprintf(stderr, "block %dx%d: bx*by must be a multiple of 32 and <= 1024\n", bx, by);
    std::exit(EXIT_FAILURE);
  }
  const size_t N = g.size(), bytes = N * sizeof(T);
  const dim3 block(bx, by);
  const dim3 grid((g.n + bx - 1) / bx, (g.n + by - 1) / by);
  const int nblocks = grid.x * grid.y;
  const bool smem = cfg.kernel == KernelKind::Smem;
  const size_t smem_bytes = smem ? static_cast<size_t>(bx + 2 * g.R) * (by + 2 * g.R) * sizeof(T) : 0;
  if (smem_bytes > 48 * 1024) {
    std::fprintf(stderr, "shared-memory tile of %zu bytes too large\n", smem_bytes);
    std::exit(EXIT_FAILURE);
  }
  upload_stencil(g, s, bx + 2 * g.R);

  T *dU, *dV, *dPart, *dRes;
  CUDA_CHECK(cudaMalloc(&dU, bytes));
  CUDA_CHECK(cudaMalloc(&dV, bytes));
  CUDA_CHECK(cudaMalloc(&dPart, nblocks * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&dRes, sizeof(T)));
  cudaEvent_t ev[4];
  for (auto& e : ev) CUDA_CHECK(cudaEventCreate(&e));

  const T sc = static_cast<T>(s.sc), inv = static_cast<T>(1.0 / s.sc);
  const int np = s.np(), norm = static_cast<int>(P.norm);

  // --- transfer to the GPU; V gets the boundary values by a device-to-device copy
  CUDA_CHECK(cudaEventRecord(ev[0]));
  CUDA_CHECK(cudaMemcpy(dU, hU, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dV, dU, bytes, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaEventRecord(ev[1]));

  // --- sweeps; only one scalar (the residual) goes back to the host per check
  double r_first = -1.0;
  int k = 0;
  for (; k < P.kmax; ++k) {
    const bool check = P.tol > 0 && (k % P.check_every == 0);
    launch_sweep<T>(smem, grid, block, smem_bytes, dU, dV, g, np, sc, inv, check, dPart, norm);
    std::swap(dU, dV);
    if (check) {
      reduce_partials<T><<<1, 1024>>>(dPart, nblocks, dRes, norm);
      CUDA_CHECK(cudaGetLastError());
      T hr;
      CUDA_CHECK(cudaMemcpy(&hr, dRes, sizeof(T), cudaMemcpyDeviceToHost));  // synchronises
      const double r = finalize_residual(static_cast<double>(hr), P.norm, g, P);
      if (check_stop(r, r_first, P, out)) { ++k; break; }
    }
  }
  CUDA_CHECK(cudaEventRecord(ev[2]));

  // --- transfer the result back (after the last swap, dU holds the newest iterate)
  CUDA_CHECK(cudaMemcpy(hU, dU, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(ev[3]));
  CUDA_CHECK(cudaEventSynchronize(ev[3]));

  float ms[3];
  for (int q = 0; q < 3; ++q) CUDA_CHECK(cudaEventElapsedTime(&ms[q], ev[q], ev[q + 1]));
  out.sweeps = k;
  out.t_h2d = ms[0] * 1e-3;
  out.t_compute = ms[1] * 1e-3;
  out.t_d2h = ms[2] * 1e-3;
  out.t_total = out.t_h2d + out.t_compute + out.t_d2h;

  for (auto& e : ev) CUDA_CHECK(cudaEventDestroy(e));
  CUDA_CHECK(cudaFree(dU));
  CUDA_CHECK(cudaFree(dV));
  CUDA_CHECK(cudaFree(dPart));
  CUDA_CHECK(cudaFree(dRes));
  return out;
}

template SolveResult gpu_solve<float>(const Grid&, const Stencil&, const SolveParams&, const GpuConfig&, float*);
template SolveResult gpu_solve<double>(const Grid&, const Stencil&, const SolveParams&, const GpuConfig&, double*);

// ---------------------------------------------------------------------------------
bool gpu_available() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

std::string gpu_name() {
  int dev = 0;
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  return prop.name;
}

void* host_alloc(size_t bytes, bool pinned) {
  void* p = nullptr;
  if (pinned) {
    CUDA_CHECK(cudaMallocHost(&p, bytes));
  } else {
    p = std::aligned_alloc(64, ((bytes + 63) / 64) * 64);
    if (!p) { std::fprintf(stderr, "out of host memory\n"); std::exit(EXIT_FAILURE); }
  }
  return p;
}

void host_free(void* p, bool pinned) {
  if (!p) return;
  if (pinned) CUDA_CHECK(cudaFreeHost(p));
  else std::free(p);
}

// ---------------------------------------------------------------------------------
__global__ void copy_kernel(const double* __restrict__ a, double* __restrict__ b, size_t n) {
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<size_t>(gridDim.x) * blockDim.x)
    b[i] = a[i];
}

void gpu_bandwidth_benchmark(size_t bytes) {
  int dev = 0;
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  std::printf("GPU %d: %s, %d SMs, %.1f GB global memory\n", dev, prop.name,
              prop.multiProcessorCount, prop.totalGlobalMem / 1e9);

  const size_t n = bytes / sizeof(double);
  double *da, *db;
  CUDA_CHECK(cudaMalloc(&da, n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&db, n * sizeof(double)));
  CUDA_CHECK(cudaMemset(da, 0, n * sizeof(double)));
  CUDA_CHECK(cudaMemset(db, 0, n * sizeof(double)));
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  float ms;
  const int reps = 20;
  const int blocks = static_cast<int>(std::min<size_t>((n + 255) / 256, prop.multiProcessorCount * 32));

  copy_kernel<<<blocks, 256>>>(da, db, n);  // warm-up
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(e0));
  for (int r = 0; r < reps; ++r) copy_kernel<<<blocks, 256>>>(da, db, n);
  CUDA_CHECK(cudaEventRecord(e1));
  CUDA_CHECK(cudaEventSynchronize(e1));
  CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
  std::printf("  copy kernel      : %8.1f GB/s  (read + write, %.0f MB)\n",
              2.0 * n * 8 * reps / (ms * 1e-3) / 1e9, n * 8 / 1e6);

  CUDA_CHECK(cudaEventRecord(e0));
  for (int r = 0; r < reps; ++r) CUDA_CHECK(cudaMemcpy(db, da, n * 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaEventRecord(e1));
  CUDA_CHECK(cudaEventSynchronize(e1));
  CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
  std::printf("  cudaMemcpy D2D   : %8.1f GB/s  (read + write)\n", 2.0 * n * 8 * reps / (ms * 1e-3) / 1e9);

  for (int pinned = 0; pinned < 2; ++pinned) {
    double* h = static_cast<double*>(host_alloc(n * sizeof(double), pinned));
    for (size_t i = 0; i < n; ++i) h[i] = 1.0;  // touch pages
    CUDA_CHECK(cudaMemcpy(da, h, n * 8, cudaMemcpyHostToDevice));  // warm-up
    float th = 1e30f, td = 1e30f;
    for (int r = 0; r < 5; ++r) {
      CUDA_CHECK(cudaEventRecord(e0));
      CUDA_CHECK(cudaMemcpy(da, h, n * 8, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaEventRecord(e1));
      CUDA_CHECK(cudaEventSynchronize(e1));
      CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
      th = std::min(th, ms);
      CUDA_CHECK(cudaEventRecord(e0));
      CUDA_CHECK(cudaMemcpy(h, da, n * 8, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaEventRecord(e1));
      CUDA_CHECK(cudaEventSynchronize(e1));
      CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
      td = std::min(td, ms);
    }
    std::printf("  H2D %-8s     : %8.1f GB/s\n", pinned ? "pinned" : "pageable", n * 8 / (th * 1e-3) / 1e9);
    std::printf("  D2H %-8s     : %8.1f GB/s\n", pinned ? "pinned" : "pageable", n * 8 / (td * 1e-3) / 1e9);
    host_free(h, pinned);
  }
  CUDA_CHECK(cudaEventDestroy(e0));
  CUDA_CHECK(cudaEventDestroy(e1));
  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(db));
}
