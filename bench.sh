#!/usr/bin/env bash
# Benchmark sweep for the report. Writes CSV files into results/.
#
# Usage:  scripts/bench.sh BW_CPU BW_GPU BW_PCIE      (GB/s, from ./jacobi --bench-bw)
# e.g.    OMP_NUM_THREADS=32 OMP_PROC_BIND=close OMP_PLACES=cores scripts/bench.sh 180 1500 22
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=./jacobi
BWC=${1:-0}; BWG=${2:-0}; BWP=${3:-0}
MODEL="--bw-cpu $BWC --bw-gpu $BWG --bw-pcie $BWP"
mkdir -p results

echo "== bandwidths" ; $BIN --bench-bw | tee results/bandwidth.txt

# 1) k = 100 sweeps, all stencils, grid sizes, both precisions
out=results/sizes.csv; first=1
for prec in double float; do
  for st in 5pt 9pt 9pt-cross 13pt-cross; do
    for n in 128 256 512 1024 2048 4096 8192; do
      hdr=""; [[ $first == 1 ]] && hdr="--csv-header" && first=0
      $BIN --prec $prec --stencil $st --n $n --k 100 --csv $hdr $MODEL
    done
  done
done > $out; echo "wrote $out"

# 2) crossover: time vs. number of sweeps k (transfers included)
out=results/crossover.csv; first=1
for n in 512 2048 8192; do
  for k in 1 2 5 10 20 50 100 200 500; do
    hdr=""; [[ $first == 1 ]] && hdr="--csv-header" && first=0
    $BIN --n $n --k $k --csv $hdr $MODEL
    $BIN --n $n --k $k --csv --pinned $MODEL
  done
done > $out; echo "wrote $out"

# 3) optimisations: kernel, block shape, precision (GPU only)
out=results/optim.csv; first=1
for prec in double float; do
  for st in 5pt 13pt-cross; do
    for kern in generic smem; do
      for blk in 32x1 32x4 32x8 32x16 64x4 128x2 256x1 16x16 8x32; do
        hdr=""; [[ $first == 1 ]] && hdr="--csv-header" && first=0
        $BIN --mode gpu --prec $prec --stencil $st --n 8192 --k 100 --kernel $kern --block $blk --csv $hdr $MODEL
      done
    done
  done
done > $out; echo "wrote $out"

# 4) stopping criterion: tolerances, residual every sweep vs. every 10th
out=results/tol.csv; first=1
for st in 5pt 9pt; do
  for tol in 1e-4 1e-6 1e-8; do
    for m in 1 10; do
      hdr=""; [[ $first == 1 ]] && hdr="--csv-header" && first=0
      $BIN --n 256 --stencil $st --tol $tol --k 1000000 --check-every $m --reps 1 --csv $hdr $MODEL
    done
  done
done > $out; echo "wrote $out"
