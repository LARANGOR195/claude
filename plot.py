#!/usr/bin/env python3
"""Plots for the report from the CSV files written by scripts/bench.sh (needs matplotlib)."""
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = sys.argv[1] if len(sys.argv) > 1 else "results"


def load(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def f(r, k):
    return float(r[k])


# --- 1) time per sweep vs n, measured and model ------------------------------------
rows = load("sizes.csv")
if rows:
    for prec in ("double", "float"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, dev in zip(axes, ("cpu", "gpu")):
            by_st = defaultdict(list)
            for r in rows:
                if r["prec"] == prec:
                    by_st[r["stencil"]].append(r)
            for st, rs in by_st.items():
                rs.sort(key=lambda r: int(r["n"]))
                n = [int(r["n"]) for r in rs]
                if dev == "cpu":
                    t = [f(r, "t_cpu") / int(r["sweeps_cpu"]) for r in rs]
                else:
                    t = [f(r, "t_gpu_sweeps") / int(r["sweeps_gpu"]) for r in rs]
                ax.loglog(n, t, "o-", label=st)
            # model (same for all stencils: 2 values per point and sweep)
            rs = sorted(by_st["5pt"], key=lambda r: int(r["n"]))
            key = "model_t_cpu" if dev == "cpu" else "model_t_gpu_sweeps"
            sw = "sweeps_cpu" if dev == "cpu" else "sweeps_gpu"
            if rs and f(rs[0], key) > 0:
                ax.loglog([int(r["n"]) for r in rs], [f(r, key) / int(r[sw]) for r in rs], "k--", label="model")
            ax.set_title(f"{dev.upper()}, {prec}")
            ax.set_xlabel("n")
            ax.grid(True, which="both", alpha=0.3)
        axes[0].set_ylabel("time per sweep [s]")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(RES, f"time_per_sweep_{prec}.pdf"))
        print("wrote", os.path.join(RES, f"time_per_sweep_{prec}.pdf"))

# --- 2) crossover: total time vs k -------------------------------------------------
rows = load("crossover.csv")
if rows:
    ns = sorted({int(r["n"]) for r in rows})
    fig, axes = plt.subplots(1, len(ns), figsize=(4 * len(ns), 3.6))
    if len(ns) == 1:
        axes = [axes]
    for ax, n in zip(axes, ns):
        rs = [r for r in rows if int(r["n"]) == n]
        for pinned, style in (("0", "-"), ("1", ":")):
            sel = sorted([r for r in rs if r["pinned"] == pinned], key=lambda r: int(r["k"]))
            k = [int(r["k"]) for r in sel]
            if pinned == "0":
                ax.loglog(k, [f(r, "t_cpu") for r in sel], "o-", label="CPU")
            ax.loglog(k, [f(r, "t_gpu_total") for r in sel], "s" + style,
                      label="GPU incl. transfers" + (" (pinned)" if pinned == "1" else ""))
        ax.set_title(f"n = {n}")
        ax.set_xlabel("sweeps k")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("time [s]")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "crossover.pdf"))
    print("wrote", os.path.join(RES, "crossover.pdf"))

# --- 3) optimisations: effective bandwidth per configuration ---------------------
rows = load("optim.csv")
if rows:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["prec"], r["stencil"])].append(r)
    fig, axes = plt.subplots(1, len(groups), figsize=(4.5 * len(groups), 3.8), sharey=True)
    if len(groups) == 1:
        axes = [axes]
    for ax, ((prec, st), rs) in zip(axes, sorted(groups.items())):
        blocks = []
        for r in rs:
            b = f'{r["bx"]}x{r["by"]}'
            if b not in blocks:
                blocks.append(b)
        x = range(len(blocks))
        for i, kern in enumerate(("generic", "smem")):
            vals = {f'{r["bx"]}x{r["by"]}': f(r, "bw_gpu_eff") for r in rs if r["kernel"] == kern}
            ax.bar([xi + 0.4 * i for xi in x], [vals.get(b, 0) for b in blocks], 0.4, label=kern)
        if f(rs[0], "model_t_gpu_sweeps") > 0:
            # the model bandwidth is the measured stream bandwidth passed to the program
            bw = f(rs[0], "bw_gpu_eff") * f(rs[0], "t_gpu_sweeps") / f(rs[0], "model_t_gpu_sweeps")
            ax.axhline(bw, color="k", ls="--", lw=1, label="model BW")
        ax.set_xticks([xi + 0.2 for xi in x])
        ax.set_xticklabels(blocks, rotation=45)
        ax.set_title(f"{st}, {prec}")
    axes[0].set_ylabel("effective bandwidth [GB/s]")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "optimisations.pdf"))
    print("wrote", os.path.join(RES, "optimisations.pdf"))
