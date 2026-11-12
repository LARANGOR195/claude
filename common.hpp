// common.hpp - shared data structures for the CPU and GPU Jacobi codes.
#pragma once
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

constexpr int MAX_RADIUS = 3;
constexpr int MAX_POINTS = (2 * MAX_RADIUS + 1) * (2 * MAX_RADIUS + 1) - 1;  // off-centre points

enum class Norm { Inf = 0, L2 = 1 };

// A stencil S, already multiplied by h^2 (as in Exercise 1):
//   (S U)_{i,j} = sc * U_{i,j} + sum_p c[p] * U_{i+di[p], j+dj[p]}
// Only the off-centre points are stored in di/dj/c.
struct Stencil {
  std::string name;
  int radius = 0;
  double sc = 0.0;
  std::vector<int> di, dj;
  std::vector<double> c;
  int np() const { return static_cast<int>(c.size()); }
};
Stencil make_stencil(const std::string& name);
std::vector<std::string> stencil_names();

// Grid with n x n interior points on (0,1)^2, h = 1/(n+1), plus a halo of
// width R (= stencil radius) on each side that holds the Dirichlet values.
// Storage is row-major: index (i, j) = i * ld + j, i <-> y, j <-> x.
struct Grid {
  int n, R, ld;
  double h;
  Grid(int n_, int R_) : n(n_), R(R_), ld(n_ + 2 * R_), h(1.0 / (n_ + 1)) {}
  size_t size() const { return static_cast<size_t>(ld) * ld; }
  size_t idx(int i, int j) const { return static_cast<size_t>(i) * ld + j; }
  double x(int j) const { return (j - R + 1) * h; }
  double y(int i) const { return (i - R + 1) * h; }
};

struct SolveParams {
  int kmax = 100;        // fixed number of sweeps (tol <= 0) or maximum number of sweeps
  double tol = 0.0;      // stopping tolerance on the residual; <= 0 means "no stopping criterion"
  int check_every = 1;   // compute the residual every m-th sweep
  Norm norm = Norm::Inf;
  double rscale = 1.0;   // residual = rscale * ||S U||   (1/h^2 -> residual of the operator A = S/h^2)
  double diverge_factor = 1e3;  // stop if r > diverge_factor * r_first (or r is not finite)
};

struct SolveResult {
  int sweeps = 0;
  double residual = -1.0;   // last computed residual (of the iterate *before* the last checked sweep)
  bool converged = false;
  bool diverged = false;
  double t_h2d = 0, t_compute = 0, t_d2h = 0, t_total = 0;  // seconds
};

// raw = max |SU| (Inf) or sum (SU)^2 (L2). Discrete L2 norm: sqrt(h^2 * sum).
inline double finalize_residual(double raw, Norm norm, const Grid& g, const SolveParams& P) {
  const double r = (norm == Norm::Inf) ? raw : std::sqrt(raw * g.h * g.h);
  return P.rscale * r;
}

// Shared stopping logic so that CPU and GPU make identical decisions.
// Returns true if the iteration should stop.
inline bool check_stop(double r, double& r_first, const SolveParams& P, SolveResult& out) {
  out.residual = r;
  if (r_first < 0) r_first = r;
  if (r < P.tol) { out.converged = true; return true; }
  if (!(r <= P.diverge_factor * r_first)) { out.diverged = true; return true; }  // also catches NaN/inf
  return false;
}
