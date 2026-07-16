#!/usr/bin/env python
"""Generate only fig_diversity.pdf and fig_language2.pdf (self-contained).

Reads the per-run metric cache (<run>/figure_cache.pkl) that accumulates
alongside each simulation, so no snapshot recompute / jax / physis is needed.
    .venv/bin/python paper/make_figures_paper.py
"""
import os, glob, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SEEDS = [62, 63, 64, 65, 66]
OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]  # colorblind-safe, one/seed
FIGDIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9,
    "axes.labelsize": 9, "legend.fontsize": 7.5, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.formatter.use_mathtext": True,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})


def find_run(seed):
    ds = glob.glob(f"output/run_200000_cycles_seed_{seed}_*")
    return max(ds, key=lambda d: os.path.basename(d).split(f"seed_{seed}_")[-1]) if ds else None


def collect():
    """{seed: {'cycles':[...], <metric>:[...]}} read from each run's figure_cache.pkl."""
    data = {}
    for s in SEEDS:
        d = find_run(s)
        if not d:
            continue
        p = os.path.join(d, "figure_cache.pkl")
        if not os.path.exists(p):
            print(f"seed {s}: no {p}, skipping"); continue
        with open(p, "rb") as f:
            snaps = pickle.load(f)["snaps"]           # {cycle: {metric: value}}
        cycles = sorted(snaps)
        rec = {"cycles": cycles}
        for k in ("sr_unique_raw", "sr_unique_eff", "sr_gest_nunique",
                  "instructions_defined", "instructions_used",
                  "instruction_reuse", "micro_ops_per_instruction"):
            rec[k] = [snaps[c][k] for c in cycles]
        data[s] = rec
    return data


def _plot(ax, data, ykey, ylabel):
    for i, s in enumerate(SEEDS):
        if s not in data:
            continue
        ax.plot(np.asarray(data[s]["cycles"], float),
                np.asarray(data[s][ykey], float), color=OKABE[i], lw=1.5, label=f"seed {s}")
    ax.set_xlabel("cycle"); ax.set_ylabel(ylabel)


def savefig(fig, name):
    for ax in fig.axes:
        if ax.get_xlabel() == "cycle":
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    p = os.path.join(FIGDIR, name)
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


def main():
    data = collect()
    if not data:
        print("no run data found"); return

    # ---- diversity: genomic | executable phenotype | behavioural phenotype ----
    fig, axs = plt.subplots(1, 3, figsize=(9.5, 2.8))
    _plot(axs[0], data, "sr_unique_raw", "distinct genomes")
    _plot(axs[1], data, "sr_unique_eff", "distinct programs")
    _plot(axs[2], data, "sr_gest_nunique", "distinct replication speeds")
    savefig(fig, "fig_diversity.pdf")

    # ---- language: instructions, recurrence, hierarchy ----
    fig, axs = plt.subplots(1, 3, figsize=(9.5, 2.8))
    for i, s in enumerate(SEEDS):
        if s not in data:
            continue
        c = np.asarray(data[s]["cycles"], float)
        axs[0].plot(c, data[s]["instructions_defined"], color=OKABE[i], lw=1.5)
        axs[0].plot(c, data[s]["instructions_used"], color=OKABE[i], lw=1.0, ls="--", alpha=0.7)
    axs[0].set_xlabel("cycle"); axs[0].set_ylabel("# instructions")
    axs[0].legend(handles=[Line2D([], [], color="0.3", lw=1.5, label="total"),
                           Line2D([], [], color="0.3", lw=1.0, ls="--", label="used")],
                  loc="best", frameon=False)
    _plot(axs[1], data, "instruction_reuse", "refs / used instruction")
    _plot(axs[2], data, "micro_ops_per_instruction", "micro-ops / instruction")
    savefig(fig, "fig_language2.pdf")

    print("done.")


if __name__ == "__main__":
    main()
