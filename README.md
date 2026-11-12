# Project 1 – Jacobi sweeps with CUDA

Group: *(names)*

Jacobi iteration `V = U − (S U)/s_c` for general stencils `S` (multiplied by `h²`) on the
unit square, on the CPU (OpenMP) and on the GPU (CUDA), with a residual-based stopping
criterion and a timing harness.

## Build

```bash
make                         # ./jacobi  (needs nvcc; uses -arch=native)
make CUDA_ARCH=-arch=sm_80   # if your nvcc is older than 11.5 (sm_70 V100, sm_80 A100, sm_90 H100)
make cpu                     # ./jacobi_cpu, CPU-only build without CUDA
make test                    # correctness tests in double and float
```

## Run

```bash
./jacobi --test                                   # all tests (stencils x sizes x kernels x block shapes)
./jacobi --bench-bw                               # CPU stream copy/triad, GPU copy, H2D/D2H bandwidths
./jacobi --n 4096 --stencil 9pt --k 100           # k fixed sweeps, CPU vs GPU incl. transfers
./jacobi --n 256 --tol 1e-6 --k 1000000           # with stopping criterion (k = max. sweeps)
./jacobi --n 8192 --mode gpu --kernel smem --block 64x4 --prec float
./jacobi --n 4096 --bw-cpu 180 --bw-gpu 1500 --bw-pcie 22   # also prints the model prediction
scripts/bench.sh 180 1500 22 && python3 scripts/plot.py     # all measurements + plots for the report
```

Set `OMP_NUM_THREADS`, `OMP_PROC_BIND=close` and `OMP_PLACES=cores` for reproducible CPU timings.
`./jacobi --help` lists all options.

## Files

| file | content |
|---|---|
| `src/common.hpp` | grid, stencil, solver parameters, shared stopping logic |
| `src/stencil.cpp` | the stencils |
| `src/cpu_jacobi.cpp` | OpenMP sweep, Jacobi loop, CPU bandwidth benchmark |
| `src/gpu_jacobi.cu` | CUDA kernels (generic and shared memory), residual reduction, GPU driver, GPU bandwidth benchmark |
| `src/gpu_stub.cpp` | replaces the CUDA part in the CPU-only build |
| `src/main.cpp` | command line, initial values, tests, timing, model |
| `scripts/bench.sh`, `scripts/plot.py` | measurement series and plots |

## Design

**Grid.** `n × n` interior points, `h = 1/(n+1)`, plus a halo of width `R` (the stencil radius)
that holds the Dirichlet values. Row-major storage with leading dimension `ld = n + 2R`, `x`
is the contiguous direction. The halo is never written by a sweep, so it is set once in both `U` and `V`.
After every sweep only the pointers are swapped.

**Stencils** (`--stencil`). Each stencil is stored as a list of offsets `(di, dj)` and coefficients
plus the centre coefficient `s_c`, so one kernel handles every stencil:

| name | radius | points | `h²·S` |
|---|---|---|---|
| `5pt` | 1 | 5 | `[0 −1 0; −1 4 −1; 0 −1 0]` |
| `9pt` | 1 | 9 | `1/6 [−1 −4 −1; −4 20 −4; −1 −4 −1]` (compact) |
| `9pt-cross` | 2 | 9 | 4th order, `1/12 [1 −16 30 −16 1]` per direction |
| `13pt-cross` | 3 | 13 | 6th order, `1/180 [−2 27 −270 490 −270 27 −2]` per direction |

All of them are exact for quadratic polynomials, which the tests rely on. A new stencil is a few lines in `stencil.cpp`.

**Tests** (`--test`). For every stencil, `n ∈ {1,2,3,7,31,32,33,100,257,1000}` (including sizes
that are not multiples of the block size), both kernels and several block shapes:
1. one sweep on `u = x² + y²` gives exactly `U + 4h²/s_c` on the interior, with the halo unchanged (CPU and GPU);
2. the residual of `u = x² + y²` is `‖S U/h²‖_∞ = 4` and `‖S U/h²‖₂ = 4nh` (CPU), and the GPU reduction gives the same value;
3. starting from a random guess, the GPU agrees with the CPU after 10 sweeps.

**Initial guess** (`--init random`). The interior is uniform random in [0,1). Each value is computed
from the seed and the index (counter-based generator), so CPU and GPU start from bit-identical
values whatever the thread count. The boundary is `g = x² − y²`, which is harmonic and quadratic,
so the discrete solution is exactly `x² − y²`. The program also prints the error against it.

**Residual.** `r = ‖S U‖ / h²`, i.e. the residual of the linear system `A U = 0` with `A = S/h²`
(`--rscale none` for `‖S U‖`), in the ∞-norm or the discrete 2-norm `sqrt(h² Σ (SU)²)` (`--norm`).
`S U` is computed anyway during the sweep, so the residual of `U` is reduced in the same kernel
(template parameter `RES`) at almost no extra memory traffic. The reduction is written by hand:
- stage 1: inside each block, warp shuffles (`__shfl_down_sync`) and then one shared-memory value per warp, giving one partial result per block;
- stage 2: one block of 1024 threads reduces the partials;
- only one scalar per check goes back to the host. `--check-every m` checks only every m-th sweep.

The iteration stops when `r < tol`, after `kmax` sweeps, or when `r > 10³ · r_first` (divergence).
CPU and GPU share the same stopping logic (`check_stop` in `common.hpp`). If the two stop after
a different number of sweeps, the CPU is re-run with the GPU's count before the results are compared.

**CPU/GPU split.** The CPU does the initialisation (once, cheap, and the identical start values
are needed for the comparison anyway), the stopping decision (one scalar per check) and the
verification. The GPU does everything that runs `k` times, i.e. the sweeps and the residual, so
the data crosses PCIe only twice: `U` H2D at the start and the result D2H at the end. `V` gets its
boundary values by a device-to-device copy, not by a second H2D transfer.

**GPU kernels** (`--kernel`, `--block BXxBY`). One thread per interior point on a 2D grid of 2D blocks,
with bounds checks for arbitrary `n`. The stencil (offsets and coefficients) is in `__constant__`
memory: all threads of a warp read the same entry at the same time, which is a broadcast.
- `generic`: neighbours are read directly from global memory, relying on L1/L2 for the reuse between neighbouring threads.
- `smem`: each block first loads its tile plus a halo of width `R` into shared memory, then computes from the tile.

`--prec float` switches everything to single precision and `--pinned` uses page-locked host memory for the transfers.

**CPU kernel.** OpenMP over rows. For each row, `SU` is accumulated in a row buffer with the
loop over stencil points *outside* the loop over `j`. Every inner loop is then a contiguous
AXPY that the compiler vectorises, for any stencil. Arrays are first-touched in parallel with the
same static schedule (this matters for NUMA).

## Performance model

Each sweep must read `U` and write `V` at least once, i.e. `2·n²·s` bytes (`s` = 8 or 4), whatever the stencil.
The neighbours come from cache if reuse is perfect. With the measured stream bandwidths `B_cpu` and `B_gpu` and the PCIe bandwidth `B_pcie`:

```
T_cpu(k)  = k · 2 n² s / B_cpu
T_gpu(k)  = 2 ld² s / B_pcie  +  2 ld² s / B_gpu  +  k · 2 n² s / B_gpu
            (H2D + D2H)          (copy U -> V)       (sweeps)
k*        = T_transfer / (2 n² s / B_cpu − 2 n² s / B_gpu)      (GPU faster for k > k*)
```

Since `B_pcie ≪ B_cpu`, one transfer costs several CPU sweeps. `k*` is therefore at least
`≈ 2 B_cpu / B_pcie` and does not depend on `n` (as long as the arrays do not fit in the CPU cache).
The program prints the measured crossover `(t_H2D + t_D2H)/(t_sweep,CPU − t_sweep,GPU)` and, with
`--bw-*`, the model values. Where the measurements deviate from the model, look for:
- on the CPU: write-allocate (3 instead of 2 transfers per point, unless streaming stores are used), and small grids that fit in L2/L3 (faster than the model);
- on the GPU: small grids that do not fill the device or are dominated by launch latency (≈ 5 µs per kernel), and large-radius stencils whose rows no longer fit in L1/L2 so that `U` is read more than once;
- pageable vs. pinned host memory for the transfers.

## Convergence of the stencils

For `5pt` and `9pt` the iteration matrix `I − S/s_c` has spectral radius `1 − O(h²) < 1`, so Jacobi
converges (slowly: `O(n²)` sweeps). For the higher-order cross stencils the largest eigenvalue of
`S` (at the highest frequency, θ = π in both directions) is larger than `2 s_c`:
- `9pt-cross`: `2·64/12 = 10.7 > 2·5`, giving the eigenvalue `1 − 10.7/5 ≈ −1.13`;
- `13pt-cross`: `≈ 12.1 > 2·5.44`, giving `≈ −1.22`.

So Jacobi **diverges** for these, and the program reports `DIVERGED`. For them, compare CPU and
GPU after a fixed number of sweeps (`--k 100`, no `--tol`), as the assignment says.
