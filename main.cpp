// main.cpp - driver: correctness tests, timing harness, performance model.
#include "common.hpp"
#include "cpu_jacobi.hpp"
#include "gpu_jacobi.hpp"
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------------
struct Options {
  int n = 1024;
  std::string stencil = "5pt";
  int k = 100;
  double tol = 0.0;
  int check_every = 1;
  Norm norm = Norm::Inf;
  bool rscale_op = true;  // residual of A U with A = S / h^2 (true) or of S U (false)
  std::string init = "random";
  uint64_t seed = 42;
  std::string prec = "double";
  GpuConfig gpu;
  bool pinned = false;
  std::string mode = "both";  // both | cpu | gpu
  int reps = 3;
  bool csv = false, csv_header = false;
  bool test = false, bench_bw = false;
  double bw_cpu = 0, bw_gpu = 0, bw_pcie = 0;  // GB/s, for the model
};

static void usage() {
  std::printf(
      "usage: jacobi [options]\n"
      "  --n N              interior points per direction (default 1024)\n"
      "  --stencil S        5pt | 9pt | 9pt-cross | 13pt-cross (default 5pt)\n"
      "  --k K              number of sweeps, or max. number of sweeps with --tol (default 100)\n"
      "  --tol T            stop when residual < T (default 0 = fixed number of sweeps)\n"
      "  --check-every M    compute the residual every M sweeps (default 1)\n"
      "  --norm inf|l2      residual norm (default inf)\n"
      "  --rscale op|none   residual of S/h^2 U (op, default) or of S U (none)\n"
      "  --init random|quad initial guess (default random, fixed seed)\n"
      "  --seed S           seed for the random initial guess (default 42)\n"
      "  --prec double|float\n"
      "  --kernel generic|smem   GPU kernel (default generic)\n"
      "  --block BXxBY      thread block shape (default 32x8)\n"
      "  --pinned           use pinned host memory for the GPU transfers\n"
      "  --mode both|cpu|gpu\n"
      "  --reps R           timing repetitions, the fastest is reported (default 3)\n"
      "  --csv, --csv-header  machine-readable output\n"
      "  --bw-cpu B --bw-gpu B --bw-pcie B   bandwidths in GB/s for the performance model\n"
      "  --test             run the correctness tests\n"
      "  --bench-bw         measure CPU and GPU memory/transfer bandwidths\n");
}

static Options parse(int argc, char** argv) {
  Options o;
  for (int a = 1; a < argc; ++a) {
    std::string s = argv[a];
    auto next = [&]() -> std::string {
      if (a + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", s.c_str()); std::exit(1); }
      return argv[++a];
    };
    if (s == "--n") o.n = std::stoi(next());
    else if (s == "--stencil") o.stencil = next();
    else if (s == "--k") o.k = std::stoi(next());
    else if (s == "--tol") o.tol = std::stod(next());
    else if (s == "--check-every") o.check_every = std::max(1, std::stoi(next()));
    else if (s == "--norm") { std::string v = next(); o.norm = (v == "l2") ? Norm::L2 : Norm::Inf; }
    else if (s == "--rscale") o.rscale_op = (next() != "none");
    else if (s == "--init") o.init = next();
    else if (s == "--seed") o.seed = std::stoull(next());
    else if (s == "--prec") o.prec = next();
    else if (s == "--kernel") o.gpu.kernel = (next() == "smem") ? KernelKind::Smem : KernelKind::Generic;
    else if (s == "--block") {
      std::string v = next();
      if (std::sscanf(v.c_str(), "%dx%d", &o.gpu.bx, &o.gpu.by) != 2) { std::fprintf(stderr, "bad --block\n"); std::exit(1); }
    }
    else if (s == "--pinned") o.pinned = true;
    else if (s == "--mode") o.mode = next();
    else if (s == "--reps") o.reps = std::max(1, std::stoi(next()));
    else if (s == "--csv") o.csv = true;
    else if (s == "--csv-header") o.csv = o.csv_header = true;
    else if (s == "--bw-cpu") o.bw_cpu = std::stod(next());
    else if (s == "--bw-gpu") o.bw_gpu = std::stod(next());
    else if (s == "--bw-pcie") o.bw_pcie = std::stod(next());
    else if (s == "--test") o.test = true;
    else if (s == "--bench-bw") o.bench_bw = true;
    else if (s == "-h" || s == "--help") { usage(); std::exit(0); }
    else { std::fprintf(stderr, "unknown option %s\n", s.c_str()); usage(); std::exit(1); }
  }
  return o;
}

// ---------------------------------------------------------------------------------
// Host buffers (aligned or pinned) and initialisation
// ---------------------------------------------------------------------------------
template <typename T>
struct Buf {
  T* p = nullptr;
  size_t n = 0;
  bool pinned = false;
  explicit Buf(size_t n_, bool pinned_ = false) : n(n_), pinned(pinned_) {
    p = static_cast<T*>(host_alloc(n * sizeof(T), pinned));
  }
  ~Buf() { host_free(p, pinned); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
};

template <typename T>
static void par_copy(const T* a, T* b, size_t n) {
#pragma omp parallel for schedule(static)
  for (long i = 0; i < static_cast<long>(n); ++i) b[i] = a[i];
}

// u = x^2 + y^2 everywhere (interior and halo): one sweep must give U + 4 h^2 / sc.
template <typename T>
static void init_quadratic(const Grid& g, T* U) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < g.ld; ++i)
    for (int j = 0; j < g.ld; ++j) {
      const double x = g.x(j), y = g.y(i);
      U[g.idx(i, j)] = static_cast<T>(x * x + y * y);
    }
}

static inline uint64_t splitmix64(uint64_t z) {
  z += 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

// Boundary (halo): g(x,y) = x^2 - y^2. It is harmonic and quadratic, so every consistent
// stencil reproduces it exactly: the discrete solution of S U = 0 is U = x^2 - y^2.
// Interior: uniform random numbers in [0,1). The value of point (i,j) only depends on the
// seed and the index (counter-based generator), so the initial guess is identical for CPU
// and GPU and independent of the number of threads.
template <typename T>
static void init_random(const Grid& g, T* U, uint64_t seed) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < g.ld; ++i)
    for (int j = 0; j < g.ld; ++j) {
      const bool interior = i >= g.R && i < g.R + g.n && j >= g.R && j < g.R + g.n;
      const double x = g.x(j), y = g.y(i);
      const double v = interior ? (splitmix64(seed * 0x100000001b3ULL + g.idx(i, j)) >> 11) * 0x1.0p-53
                                : x * x - y * y;
      U[g.idx(i, j)] = static_cast<T>(v);
    }
}

template <typename T>
static double max_abs_diff(const T* a, const T* b, size_t n) {
  double m = 0;
#pragma omp parallel for reduction(max : m)
  for (long i = 0; i < static_cast<long>(n); ++i)
    m = std::max(m, std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
  return m;
}

template <typename T>
static double max_abs(const T* a, size_t n) {
  double m = 0;
#pragma omp parallel for reduction(max : m)
  for (long i = 0; i < static_cast<long>(n); ++i) m = std::max(m, std::abs(static_cast<double>(a[i])));
  return m;
}

// error of a single sweep applied to u = x^2 + y^2: interior must be U0 + 4h^2/sc, halo unchanged
template <typename T>
static double quad_sweep_error(const Grid& g, const Stencil& s, const T* U0, const T* V) {
  const double add = 4.0 * g.h * g.h / s.sc;
  double e = 0;
  for (int i = 0; i < g.ld; ++i)
    for (int j = 0; j < g.ld; ++j) {
      const bool interior = i >= g.R && i < g.R + g.n && j >= g.R && j < g.R + g.n;
      const double expect = static_cast<double>(U0[g.idx(i, j)]) + (interior ? add : 0.0);
      e = std::max(e, std::abs(static_cast<double>(V[g.idx(i, j)]) - expect));
    }
  return e;
}

template <typename T>
static double error_vs_exact(const Grid& g, const T* U) {
  double e = 0;
#pragma omp parallel for reduction(max : e)
  for (int i = g.R; i < g.R + g.n; ++i)
    for (int j = g.R; j < g.R + g.n; ++j) {
      const double x = g.x(j), y = g.y(i);
      e = std::max(e, std::abs(static_cast<double>(U[g.idx(i, j)]) - (x * x - y * y)));
    }
  return e;
}

static const char* kname(KernelKind k) { return k == KernelKind::Smem ? "smem" : "generic"; }

// ---------------------------------------------------------------------------------
// Correctness tests
// ---------------------------------------------------------------------------------
template <typename T>
static int run_tests(const Options& o) {
  const bool gpu = gpu_available();
  const double eps = std::numeric_limits<T>::epsilon();
  const std::vector<int> sizes = {1, 2, 3, 7, 31, 32, 33, 100, 257, 1000};
  const std::vector<GpuConfig> cfgs = {{32, 8, KernelKind::Generic}, {16, 16, KernelKind::Generic},
                                       {32, 8, KernelKind::Smem},    {16, 16, KernelKind::Smem},
                                       {64, 2, KernelKind::Smem}};
  int total = 0, fails = 0;
  auto check = [&](bool ok, const char* what, const std::string& st, int n, const char* cfg, double val, double tol) {
    ++total;
    if (!ok) {
      ++fails;
      std::printf("  FAIL %-28s stencil %-10s n=%-5d %-14s value %.3e  tol %.3e\n", what, st.c_str(), n, cfg, val, tol);
    }
  };
  std::printf("Running tests in %s precision (%s)\n", o.prec.c_str(), gpu ? ("GPU: " + gpu_name()).c_str() : "no GPU, CPU only");

  for (const auto& name : stencil_names()) {
    const Stencil s = make_stencil(name);
    double sumc = std::abs(s.sc);
    for (double c : s.c) sumc += std::abs(c);
    for (int n : sizes) {
      const Grid g(n, s.radius);
      const size_t N = g.size();
      Buf<T> U0(N), A(N), B(N), G(N);
      char cfgname[64];

      // (1) one sweep on u = x^2 + y^2
      init_quadratic(g, U0.p);
      const double umax = max_abs(U0.p, N);
      const double tolq = 16 * eps * sumc * umax;
      SolveParams P1; P1.kmax = 1; P1.tol = 0;
      SolveResult rc;
      par_copy(U0.p, A.p, N);
      const T* cres = cpu_solve(g, s, P1, A.p, B.p, rc);
      double e = quad_sweep_error(g, s, U0.p, cres);
      check(e <= tolq, "CPU sweep (x^2+y^2)", name, n, "", e, tolq);

      // (2) residual of u = x^2 + y^2: S U = -4 h^2  ->  ||A U||_inf = 4, ||A U||_2 = 4 n h
      double cpu_res[2] = {0, 0};
      const double tolr_abs = 16 * eps * sumc * umax / (g.h * g.h);  // rounding error of SU / h^2
      for (int nm = 0; nm < 2; ++nm) {
        SolveParams PR; PR.kmax = 1; PR.tol = 1e-300; PR.norm = nm ? Norm::L2 : Norm::Inf;
        PR.rscale = 1.0 / (g.h * g.h);
        par_copy(U0.p, A.p, N);
        cpu_solve(g, s, PR, A.p, B.p, rc);
        cpu_res[nm] = rc.residual;
        const double expect = nm ? 4.0 * n * g.h : 4.0;
        if (tolr_abs < 0.1)  // only meaningful if rounding is small compared to the residual
          check(std::abs(rc.residual - expect) <= tolr_abs, nm ? "CPU residual l2" : "CPU residual inf",
                name, n, "", std::abs(rc.residual - expect), tolr_abs);
      }

      if (gpu) {
        for (const auto& cfg : cfgs) {
          std::snprintf(cfgname, sizeof cfgname, "%s %dx%d", kname(cfg.kernel), cfg.bx, cfg.by);
          par_copy(U0.p, G.p, N);
          gpu_solve(g, s, P1, cfg, G.p);
          e = quad_sweep_error(g, s, U0.p, G.p);
          check(e <= tolq, "GPU sweep (x^2+y^2)", name, n, cfgname, e, tolq);
          for (int nm = 0; nm < 2; ++nm) {
            SolveParams PR; PR.kmax = 1; PR.tol = 1e-300; PR.norm = nm ? Norm::L2 : Norm::Inf;
            PR.rscale = 1.0 / (g.h * g.h);
            par_copy(U0.p, G.p, N);
            const SolveResult rg = gpu_solve(g, s, PR, cfg, G.p);
            const double d = std::abs(rg.residual - cpu_res[nm]);
            const double tol = std::max(tolr_abs, 1e-3 * cpu_res[nm] * (sizeof(T) == 4 ? 1.0 : 1e-6));
            check(d <= tol, nm ? "GPU vs CPU residual l2" : "GPU vs CPU residual inf", name, n, cfgname, d, tol);
          }
        }
        // (3) random initial guess, 10 sweeps: GPU must match CPU
        init_random(g, U0.p, o.seed);
        const int k = 10;
        SolveParams PK; PK.kmax = k; PK.tol = 0;
        par_copy(U0.p, A.p, N);
        cres = cpu_solve(g, s, PK, A.p, B.p, rc);
        const double umax2 = max_abs(U0.p, N);
        const double tolk = 64 * eps * sumc * umax2 * k * std::pow(2.0, k / 5.0);  // allows mild growth (diverging stencils)
        for (const auto& cfg : cfgs) {
          std::snprintf(cfgname, sizeof cfgname, "%s %dx%d", kname(cfg.kernel), cfg.bx, cfg.by);
          par_copy(U0.p, G.p, N);
          gpu_solve(g, s, PK, cfg, G.p);
          const double d = max_abs_diff(cres, G.p, N);
          check(d <= tolk, "GPU vs CPU, 10 sweeps", name, n, cfgname, d, tolk);
        }
      }
    }
  }
  std::printf("%d of %d checks passed%s\n", total - fails, total, fails ? "  <-- FAILURES" : "");
  return fails ? 1 : 0;
}

// ---------------------------------------------------------------------------------
// Timing run
// ---------------------------------------------------------------------------------
template <typename T>
static int run(const Options& o) {
  const Stencil s = make_stencil(o.stencil);
  const Grid g(o.n, s.radius);
  const size_t N = g.size();
  SolveParams P;
  P.kmax = o.k;
  P.tol = o.tol;
  P.check_every = o.check_every;
  P.norm = o.norm;
  P.rscale = o.rscale_op ? 1.0 / (g.h * g.h) : 1.0;

  const bool do_cpu = o.mode != "gpu", do_gpu = o.mode != "cpu";
  if (do_gpu && !gpu_available()) { std::fprintf(stderr, "no GPU available (use --mode cpu)\n"); return 1; }

  Buf<T> U0(N);
  if (o.init == "quad") init_quadratic(g, U0.p);
  else init_random(g, U0.p, o.seed);

  Buf<T> Uc(do_cpu ? N : 1), Vc(do_cpu ? N : 1);
  Buf<T> Ug(do_gpu ? N : 1, do_gpu && o.pinned);
  const T* cres = nullptr;
  SolveResult rc, rg;

  if (do_gpu) {  // untimed warm-up: CUDA context, module loading, first-touch of device memory
    SolveParams W = P;
    W.kmax = std::min(2, P.kmax);
    par_copy(U0.p, Ug.p, N);
    gpu_solve(g, s, W, o.gpu, Ug.p);
  }
  for (int r = 0; r < o.reps; ++r) {
    if (do_cpu) {
      par_copy(U0.p, Uc.p, N);
      SolveResult t;
      cres = cpu_solve(g, s, P, Uc.p, Vc.p, t);
      if (r == 0 || t.t_total < rc.t_total) rc = t;
    }
    if (do_gpu) {
      par_copy(U0.p, Ug.p, N);
      SolveResult t = gpu_solve(g, s, P, o.gpu, Ug.p);
      if (r == 0 || t.t_total < rg.t_total) rg = t;
    }
  }

  // compare CPU and GPU after the same number of sweeps
  double maxdiff = -1, rel = -1;
  if (do_cpu && do_gpu) {
    if (rc.sweeps != rg.sweeps) {
      SolveParams F = P;
      F.tol = 0;
      F.kmax = rg.sweeps;
      SolveResult t;
      par_copy(U0.p, Uc.p, N);
      cres = cpu_solve(g, s, F, Uc.p, Vc.p, t);
    }
    maxdiff = max_abs_diff(cres, Ug.p, N);
    rel = maxdiff / std::max(1e-300, max_abs(cres, N));
  }

  // effective bandwidth: each sweep reads U and writes V once (interior points only)
  const double bytes_sweep = 2.0 * g.n * static_cast<double>(g.n) * sizeof(T);
  const double bw_c = do_cpu ? rc.sweeps * bytes_sweep / rc.t_compute / 1e9 : 0;
  const double bw_g = do_gpu ? rg.sweeps * bytes_sweep / rg.t_compute / 1e9 : 0;

  // simple performance model (if bandwidths are given)
  const int ks = do_gpu ? rg.sweeps : rc.sweeps;
  const double bytes_arr = static_cast<double>(N) * sizeof(T);
  const double m_cpu = o.bw_cpu > 0 ? ks * bytes_sweep / (o.bw_cpu * 1e9) : -1;
  const double m_gpu = o.bw_gpu > 0 ? ks * bytes_sweep / (o.bw_gpu * 1e9) : -1;
  const double m_tr = (o.bw_pcie > 0 && o.bw_gpu > 0)
                          ? 2.0 * bytes_arr / (o.bw_pcie * 1e9) + 2.0 * bytes_arr / (o.bw_gpu * 1e9)
                          : -1;  // H2D + D2H + device copy U -> V
  double m_kstar = -1;
  if (m_cpu > 0 && m_gpu > 0 && m_tr > 0 && ks > 0) {
    const double per_cpu = m_cpu / ks, per_gpu = m_gpu / ks;
    if (per_cpu > per_gpu) m_kstar = m_tr / (per_cpu - per_gpu);
  }

  if (o.csv) {
    if (o.csv_header)
      std::printf("prec,stencil,n,kernel,bx,by,pinned,threads,k,tol,sweeps_cpu,sweeps_gpu,res_cpu,res_gpu,"
                  "t_cpu,t_gpu_h2d,t_gpu_sweeps,t_gpu_d2h,t_gpu_total,bw_cpu_eff,bw_gpu_eff,maxdiff,"
                  "model_t_cpu,model_t_gpu_sweeps,model_t_transfer,model_kstar\n");
    std::printf("%s,%s,%d,%s,%d,%d,%d,%d,%d,%g,%d,%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.3f,%.3f,%.3e,"
                "%.6e,%.6e,%.6e,%.2f\n",
                o.prec.c_str(), s.name.c_str(), g.n, kname(o.gpu.kernel), o.gpu.bx, o.gpu.by, o.pinned ? 1 : 0,
                omp_get_max_threads(), o.k, o.tol, rc.sweeps, rg.sweeps, rc.residual, rg.residual,
                rc.t_total, rg.t_h2d, rg.t_compute, rg.t_d2h, rg.t_total, bw_c, bw_g, maxdiff, m_cpu, m_gpu,
                m_tr, m_kstar);
    return 0;
  }

  std::printf("Jacobi | stencil %s (radius %d, %d points) | n = %d (array %d^2, %.1f MB) | %s\n",
              s.name.c_str(), s.radius, s.np() + 1, g.n, g.ld, bytes_arr / 1e6, o.prec.c_str());
  if (o.tol > 0)
    std::printf("stopping: residual (%s norm, %s) < %g, check every %d, at most %d sweeps\n",
                o.norm == Norm::Inf ? "inf" : "l2", o.rscale_op ? "S/h^2 U" : "S U", o.tol, o.check_every, o.k);
  else
    std::printf("fixed number of sweeps: %d\n", o.k);
  auto status = [](const SolveResult& r) { return r.converged ? "converged" : (r.diverged ? "DIVERGED" : "max sweeps"); };
  if (do_cpu) {
    std::printf("CPU  (%d threads)          : %7d sweeps  %10.4f s  (%8.3f ms/sweep, %7.1f GB/s eff.)",
                omp_get_max_threads(), rc.sweeps, rc.t_total, 1e3 * rc.t_total / std::max(1, rc.sweeps), bw_c);
    if (o.tol > 0) std::printf("  res %.3e %s", rc.residual, status(rc));
    std::printf("\n");
  }
  if (do_gpu) {
    std::printf("GPU  (%s %dx%d%s) : %7d sweeps  %10.4f s  (%8.3f ms/sweep, %7.1f GB/s eff.)",
                kname(o.gpu.kernel), o.gpu.bx, o.gpu.by, o.pinned ? ", pinned" : "", rg.sweeps, rg.t_total,
                1e3 * rg.t_compute / std::max(1, rg.sweeps), bw_g);
    if (o.tol > 0) std::printf("  res %.3e %s", rg.residual, status(rg));
    std::printf("\n     = H2D %.4f s + sweeps %.4f s + D2H %.4f s   (%s)\n", rg.t_h2d, rg.t_compute, rg.t_d2h,
                gpu_name().c_str());
  }
  if (do_cpu && do_gpu) {
    std::printf("speed-up CPU / GPU incl. transfers: %.2f   (sweeps only: %.2f)\n", rc.t_total / rg.t_total,
                rc.t_compute / rg.t_compute);
    // measured crossover: transfers / (CPU time per sweep - GPU time per sweep)
    const double pc = rc.t_compute / rc.sweeps, pg = rg.t_compute / rg.sweeps;
    if (pc > pg)
      std::printf("measured crossover: GPU faster from k > %.1f sweeps\n", (rg.t_h2d + rg.t_d2h) / (pc - pg));
    std::printf("max |CPU - GPU| = %.3e  (relative %.3e)%s\n", maxdiff, rel,
                rc.sweeps != rg.sweeps ? "  [CPU re-run with the GPU's sweep count]" : "");
  }
  if (o.init != "quad") {
    const T* fin = do_gpu ? Ug.p : cres;
    std::printf("max |U - (x^2 - y^2)| on the interior: %.3e\n", error_vs_exact(g, fin));
  }
  if (m_cpu > 0 || m_gpu > 0) {
    std::printf("model (%.0f B/point/sweep):", 2.0 * sizeof(T));
    if (m_cpu > 0) std::printf("  CPU %.4f s", m_cpu);
    if (m_gpu > 0) std::printf("  GPU sweeps %.4f s", m_gpu);
    if (m_tr > 0) std::printf("  transfers %.4f s", m_tr);
    if (m_kstar > 0) std::printf("  crossover k* = %.1f", m_kstar);
    std::printf("\n");
  }
  return 0;
}

int main(int argc, char** argv) {
  const Options o = parse(argc, argv);
  try {
    if (o.bench_bw) {
      cpu_bandwidth_benchmark(size_t(512) << 20);
      if (gpu_available()) gpu_bandwidth_benchmark(size_t(512) << 20);
      return 0;
    }
    if (o.prec == "float") return o.test ? run_tests<float>(o) : run<float>(o);
    if (o.prec == "double") return o.test ? run_tests<double>(o) : run<double>(o);
    std::fprintf(stderr, "--prec must be float or double\n");
    return 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
