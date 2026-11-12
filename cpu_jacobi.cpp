// cpu_jacobi.cpp - OpenMP reference implementation of the Jacobi sweep.
#include "cpu_jacobi.hpp"
#include <omp.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

template <typename T>
void cpu_sweep(const Grid& g, const Stencil& s, const T* __restrict U, T* __restrict V, double* res) {
  const int n = g.n, R = g.R, ld = g.ld, np = s.np();
  const T sc = static_cast<T>(s.sc), inv = static_cast<T>(1.0 / s.sc);
  std::vector<long> off(np);
  std::vector<T> c(np);
  for (int p = 0; p < np; ++p) {
    off[p] = static_cast<long>(s.di[p]) * ld + s.dj[p];
    c[p] = static_cast<T>(s.c[p]);
  }
  double rmax = 0.0, rsum = 0.0;

#pragma omp parallel reduction(max : rmax) reduction(+ : rsum)
  {
    // Row buffer for SU: the loop over the stencil points is outside the loop over j,
    // so every inner loop is a contiguous, vectorisable AXPY for any stencil.
    std::vector<T> buf(n);
    T* __restrict su = buf.data();
#pragma omp for schedule(static)
    for (int i = R; i < R + n; ++i) {
      const T* __restrict Ur = U + static_cast<size_t>(i) * ld + R;
      T* __restrict Vr = V + static_cast<size_t>(i) * ld + R;
      for (int j = 0; j < n; ++j) su[j] = sc * Ur[j];
      for (int p = 0; p < np; ++p) {
        const T cp = c[p];
        const T* __restrict Up = Ur + off[p];
        for (int j = 0; j < n; ++j) su[j] += cp * Up[j];
      }
      for (int j = 0; j < n; ++j) Vr[j] = Ur[j] - su[j] * inv;
      if (res) {
        for (int j = 0; j < n; ++j) {
          const double a = std::abs(static_cast<double>(su[j]));
          rmax = std::max(rmax, a);
          rsum += a * a;
        }
      }
    }
  }
  if (res) { res[0] = rmax; res[1] = rsum; }
}

template <typename T>
const T* cpu_solve(const Grid& g, const Stencil& s, const SolveParams& P, T* U, T* V, SolveResult& out) {
  out = SolveResult{};
  // V needs the same boundary values as U (the halo is never written by a sweep).
#pragma omp parallel for schedule(static)
  for (long k = 0; k < static_cast<long>(g.size()); ++k) V[k] = U[k];

  const double t0 = omp_get_wtime();
  double r_first = -1.0;
  int k = 0;
  for (; k < P.kmax; ++k) {
    const bool check = P.tol > 0 && (k % P.check_every == 0);
    double raw[2];
    cpu_sweep(g, s, U, V, check ? raw : nullptr);  // residual of U is computed during the sweep
    std::swap(U, V);
    if (check) {
      const double r = finalize_residual(P.norm == Norm::Inf ? raw[0] : raw[1], P.norm, g, P);
      if (check_stop(r, r_first, P, out)) { ++k; break; }
    }
  }
  const double t1 = omp_get_wtime();
  out.sweeps = k;
  out.t_compute = out.t_total = t1 - t0;
  return U;
}

// STREAM-like copy and triad with OpenMP (for the performance model).
void cpu_bandwidth_benchmark(size_t bytes) {
  const size_t n = bytes / sizeof(double);
  double* a = static_cast<double*>(std::aligned_alloc(64, ((n * sizeof(double) + 63) / 64) * 64));
  double* b = static_cast<double*>(std::aligned_alloc(64, ((n * sizeof(double) + 63) / 64) * 64));
  double* c = static_cast<double*>(std::aligned_alloc(64, ((n * sizeof(double) + 63) / 64) * 64));
#pragma omp parallel for schedule(static)
  for (long i = 0; i < static_cast<long>(n); ++i) { a[i] = 1.0; b[i] = 2.0; c[i] = 0.5; }

  const int reps = 10;
  double tcopy = 1e30, ttriad = 1e30;
  for (int r = 0; r < reps; ++r) {
    double t0 = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (long i = 0; i < static_cast<long>(n); ++i) b[i] = a[i];
    double t1 = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (long i = 0; i < static_cast<long>(n); ++i) a[i] = b[i] + 0.5 * c[i];
    double t2 = omp_get_wtime();
    tcopy = std::min(tcopy, t1 - t0);
    ttriad = std::min(ttriad, t2 - t1);
  }
  const double GB = 1e9;
  std::printf("CPU (%d OpenMP threads), arrays of %.0f MB:\n", omp_get_max_threads(), n * 8.0 / 1e6);
  std::printf("  copy : %8.1f GB/s  (2 x 8 B / elem; %.1f GB/s incl. write-allocate)\n",
              2.0 * 8 * n / tcopy / GB, 3.0 * 8 * n / tcopy / GB);
  std::printf("  triad: %8.1f GB/s  (3 x 8 B / elem; %.1f GB/s incl. write-allocate)\n",
              3.0 * 8 * n / ttriad / GB, 4.0 * 8 * n / ttriad / GB);
  std::printf("  (checksum %g)\n", a[n / 2]);
  std::free(a); std::free(b); std::free(c);
}

template void cpu_sweep<float>(const Grid&, const Stencil&, const float*, float*, double*);
template void cpu_sweep<double>(const Grid&, const Stencil&, const double*, double*, double*);
template const float* cpu_solve<float>(const Grid&, const Stencil&, const SolveParams&, float*, float*, SolveResult&);
template const double* cpu_solve<double>(const Grid&, const Stencil&, const SolveParams&, double*, double*, SolveResult&);
