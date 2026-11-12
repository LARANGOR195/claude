// cpu_jacobi.hpp - OpenMP reference implementation.
#pragma once
#include "common.hpp"

// One sweep V = U - (S U) / sc on the interior. The halo of V is not touched.
// If res != nullptr: res[0] = max |SU|, res[1] = sum (SU)^2 over the interior.
template <typename T>
void cpu_sweep(const Grid& g, const Stencil& s, const T* U, T* V, double* res);

// Jacobi iteration (U and V swap roles after every sweep, no copies).
// U holds the initial guess (including the boundary values). Returns a pointer to
// the final iterate (either U or V).
template <typename T>
const T* cpu_solve(const Grid& g, const Stencil& s, const SolveParams& P, T* U, T* V, SolveResult& out);

void cpu_bandwidth_benchmark(size_t bytes);
