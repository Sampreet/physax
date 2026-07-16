#!/usr/bin/env python
"""Generate the summary figures for paper/experiment_summary.tex.

Self-contained: reads each run's lineage/ snapshots (genomes, status, executed)
and edge files (births), recomputes population / diversity / genome-structure /
language metrics per snapshot, and writes vector PDFs into paper/figures/.

Run (CPU is fine, keeps GPUs free for the live sims):
    JAX_PLATFORMS=cpu .venv/bin/python paper/make_figures.py
Re-run any time; it just picks up whatever snapshots have accumulated.
"""
import os, glob, pickle, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import jax
from physis.sim.config import make_config, SELF_REPLICATING, FERTILE, UNCLASSIFIED, BLANK
from physis.sim.agent import Agent
from physis.analysis.genome_stats import compute_language_stats, compute_diversity_stats
from physis.analysis.gp_map import (load_snapshots, functional_diversity, gp_map_bias,
                             genotype_network, gp_bipartite)
import networkx as nx

SEEDS = [62, 63, 64, 65, 66]
FIGDIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGDIR, exist_ok=True)

# Okabe-Ito colorblind-safe categorical palette (fixed order, one per seed).
OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9,
    "axes.labelsize": 9, "legend.fontsize": 7.5, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.formatter.use_mathtext": True,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})


def find_run(seed):
    # Pick the most recent run by the timestamp in the directory name, NOT by
    # file mtime: make_figures writes a metrics cache into the run dir, which
    # bumps its mtime and could otherwise make a stale completed run outrank the
    # live one (an mtime race that silently plotted an old 200k run for one seed).
    ds = glob.glob(f"output/run_200000_cycles_seed_{seed}_*")
    return max(ds, key=lambda d: os.path.basename(d).split(f"seed_{seed}_")[-1]) if ds else None


def rebuild_pop(snap, cfg):
    """Reconstruct a parsed Agent population from a snapshot's genomes, then graft
    the recorded alive/executed/status/gestation so language stats are exact."""
    g = np.asarray(snap["genome"]); gl = np.asarray(snap["genome_len"])
    init = jax.vmap(lambda gg, ll: Agent.init_organism(
        gg, ll, np.int32(-1), np.int32(UNCLASSIFIED), np.int32(-1), cfg))(g, gl)
    return init._replace(
        alive=np.asarray(snap["alive"]).astype(bool),
        executed=np.asarray(snap["executed"]).astype(bool),
        status=np.asarray(snap["status"]),
        gestation_time=np.asarray(snap["gestation_time"]),
    )


def births_per_cycle(lineage_dir):
    # sort by NUMERIC cycle, not lexicographically, or the x-axis is non-monotonic
    files = sorted(glob.glob(f"{lineage_dir}/edges_*.npz"),
                   key=lambda f: int(f.rsplit("_", 1)[1].split(".")[0]))
    cyc, cnt = [], []
    for f in files:
        z = np.load(f); c = int(f.rsplit("_", 1)[1].split(".")[0])
        cyc.append(c); cnt.append(int(np.sum(z["child_id"] >= 0)))
    return np.array(cyc), np.array(cnt)


# Per-snapshot metric keys (everything except "cycles", which is the cache key,
# and the births series, which is cheap and recomputed each run). Bump
# CACHE_VERSION whenever the meaning of any of these changes so stale entries
# are discarded instead of silently reused.
METRIC_KEYS = (
    "alive", "self_rep", "fertile",
    "sr_unique_raw", "sr_unique_exec", "sr_unique_eff", "sr_shannon",
    "sr_gest_mean", "sr_gest_std", "sr_gest_nunique",
    "glen", "header", "code", "registers",
    "coding_fraction", "dialect_shannon", "distinct_dialects",
    "opcode_vocab", "dead_instr_frac", "redundancy", "gini",
    "instructions_defined", "instructions_used", "instruction_reuse",
    "micro_ops_per_instruction")
CACHE_VERSION = 1


def _cache_path(run_dir):
    return os.path.join(run_dir, "figure_cache.pkl")


def _load_cache(run_dir):
    """Return {cycle: {metric: value}} for already-computed snapshots, or {}."""
    p = _cache_path(run_dir)
    if os.path.exists(p):
        try:
            with open(p, "rb") as f:
                c = pickle.load(f)
            if c.get("version") == CACHE_VERSION:
                return c["snaps"]
            print(f"[cache] {p}: version {c.get('version')} != {CACHE_VERSION}, ignoring")
        except Exception as e:  # corrupt/partial pickle -> recompute from scratch
            print(f"[cache] failed to load {p}: {e}")
    return {}


def _save_cache(run_dir, snaps_cache):
    p = _cache_path(run_dir)
    tmp = p + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"version": CACHE_VERSION, "snaps": snaps_cache}, f)
    os.replace(tmp, p)  # atomic: a crash mid-write can't corrupt the real cache


def compute_snapshot_metrics(snap, cfg):
    """The expensive per-snapshot work (rebuild + language stats). Returns a flat
    {metric: value} dict keyed by METRIC_KEYS; this is what gets cached."""
    m = {}
    alive = np.asarray(snap["alive"]).astype(bool)
    status = np.asarray(snap["status"])
    m["alive"] = int(alive.sum())
    m["self_rep"] = int((alive & (status == SELF_REPLICATING)).sum())
    m["fertile"] = int((alive & (status == FERTILE)).sum())
    # diversity (self-replicators)
    fd = functional_diversity(snap, status_filter=SELF_REPLICATING)
    m["sr_unique_raw"] = fd["raw"]["unique"]
    m["sr_unique_exec"] = fd["executed"]["unique"]
    m["sr_unique_eff"] = fd["effective"]["unique"]
    m["sr_shannon"] = fd["raw"]["shannon"]
    ds = compute_diversity_stats(snap)
    m["sr_gest_mean"] = ds.get("avg_gest_self_rep", np.nan)
    srmask = alive & (status == SELF_REPLICATING)
    gt = np.asarray(snap["gestation_time"])[srmask]
    gt = gt[gt < 2147483647]
    m["sr_gest_std"] = float(np.std(gt)) if gt.size else np.nan
    m["sr_gest_nunique"] = int(np.unique(gt).size) if gt.size else 0
    bias = gp_map_bias(snap, phenotype="effective", status_filter=SELF_REPLICATING)
    m["redundancy"] = bias["redundancy"]
    m["gini"] = bias["gini"]
    # genome structure + language (rebuild)
    pop = rebuild_pop(snap, cfg)
    ls = compute_language_stats(pop, cfg)
    sepp = np.asarray(pop.separator_pos)[alive]
    glen = np.asarray(pop.genome_len)[alive]
    m["glen"] = float(np.mean(glen)) if glen.size else np.nan
    m["header"] = float(np.mean(sepp)) if sepp.size else np.nan
    m["code"] = float(np.mean(np.maximum(glen - sepp - 1, 0))) if glen.size else np.nan
    sc = ls["scalars"] if ls else {}
    m["registers"] = sc.get("registers_mean", np.nan)
    m["coding_fraction"] = sc.get("coding_fraction", np.nan)
    m["dialect_shannon"] = sc.get("dialect_shannon", np.nan)
    m["distinct_dialects"] = sc.get("distinct_dialects", np.nan)
    m["opcode_vocab"] = sc.get("opcode_vocab_size", np.nan)
    m["dead_instr_frac"] = sc.get("dead_instruction_frac", np.nan)
    m["instructions_defined"] = sc.get("instructions_defined", np.nan)
    m["instructions_used"] = sc.get("effective_instructions", np.nan)
    m["instruction_reuse"] = sc.get("instruction_reuse", np.nan)
    m["micro_ops_per_instruction"] = sc.get("micro_ops_per_instruction", np.nan)
    return m


def collect():
    """Return {seed: {'cycles':[...], <metric>:[...]}} plus births series.

    Per-snapshot metrics are cached in <run>/figure_cache.pkl keyed by cycle and
    saved after each newly computed snapshot, so a rerun only pays for snapshots
    not already in the cache (e.g. new ones that accumulated since last time)."""
    cfg = make_config(pop_size=16384, max_genome_len=256)
    data = {}
    for s in SEEDS:
        d = find_run(s)
        if not d:
            continue
        cache = _load_cache(d)
        snaps = load_snapshots(f"{d}/lineage")
        rec = {k: [] for k in ("cycles",) + METRIC_KEYS}
        n_hit = n_new = 0
        for snap in snaps:
            cyc = int(snap["cycle"])
            if cyc not in cache:
                cache[cyc] = compute_snapshot_metrics(snap, cfg)
                _save_cache(d, cache)  # persist immediately: crash -> no recompute
                n_new += 1
                print(f"[cache] seed {s} computed cycle {cyc}")
            else:
                n_hit += 1
            m = cache[cyc]
            rec["cycles"].append(cyc)
            for k in METRIC_KEYS:
                rec[k].append(m[k])
        print(f"[cache] seed {s}: {n_hit} cached, {n_new} newly computed")
        cyc, cnt = births_per_cycle(f"{d}/lineage")
        rec["birth_cycles"], rec["births"] = cyc, cnt
        data[s] = rec
    return data


def _plot(ax, data, xkey, ykey, ylabel, logy=False):
    for i, s in enumerate(SEEDS):
        if s not in data:
            continue
        x = np.asarray(data[s][xkey], float); y = np.asarray(data[s][ykey], float)
        ax.plot(x, y, color=OKABE[i], lw=1.5, label=f"seed {s}")
    ax.set_xlabel("cycle"); ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")


def savefig(fig, name):
    # Cycle axes run to 2e5; show ticks in scientific notation (shared x10^n
    # offset at the axis end) instead of long strings of trailing zeros.
    for ax in fig.axes:
        if ax.get_xlabel() == "cycle":
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    p = os.path.join(FIGDIR, name)
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


def make_legend():
    """Standalone shared legend (one entry per seed) for side-placement in LaTeX."""
    handles = [Line2D([0], [0], color=OKABE[i], lw=1.5, label=f"seed {s}")
               for i, s in enumerate(SEEDS)]
    fig = plt.figure(figsize=(0.95, 1.5))
    fig.legend(handles=handles, loc="center", frameon=False, handlelength=1.6,
               borderaxespad=0)
    savefig(fig, "fig_legend.pdf")


def make_gpmap(data):
    """Genotype-network node-link (one representative seed) + redundancy trend."""
    seed = next((s for s in SEEDS if s in data), None)
    if seed is None:
        return
    snap = load_snapshots(f"{find_run(seed)}/lineage")[-1]
    cyc = int(snap["cycle"])
    fig, axs = plt.subplots(1, 3, figsize=(10.0, 3.2))

    # Panel 0: bipartite genotype<->phenotype (phenotype = executable program).
    B, gn, pn = gp_bipartite(snap, phenotype="effective", status_filter=SELF_REPLICATING)
    nsets = sorted((B.nodes[p]["nset"] for p in pn), reverse=True)
    print(f"[GP-map] seed {seed} cycle {cyc}: bipartite {len(gn)} genotypes / "
          f"{len(pn)} phenotypes; largest neutral sets {nsets[:6]}")
    if B.number_of_nodes() > 0:
        pos = nx.spring_layout(B, seed=0, k=0.35)
        nx.draw_networkx_edges(B, pos, ax=axs[0], alpha=0.25, width=0.4)
        nx.draw_networkx_nodes(B, pos, nodelist=list(gn), node_size=6, node_color="0.65",
                               linewidths=0, ax=axs[0])
        pdeg = [B.nodes[p]["nset"] for p in pn]
        nx.draw_networkx_nodes(B, pos, nodelist=list(pn), node_size=[25 + 20 * d for d in pdeg],
                               node_color=OKABE[0], edgecolors="white", linewidths=0.3, ax=axs[0])
        axs[0].legend(handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor="0.65", ms=4, label="genotype"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=OKABE[0], ms=8,
                   label="phenotype (size=#geno)")], fontsize=5.5, loc="best", frameon=False)
    axs[0].set_title(f"Bipartite genotype$\\leftrightarrow$phenotype\n(seed {seed}, cycle {cyc})")
    axs[0].axis("off")

    # Panel 1: genotype network (connectivity), coloured by replication speed.
    G, info, comps = genotype_network(snap, phenotype="gestation",
                                      status_filter=SELF_REPLICATING, max_dist=2)
    print(f"[GP-map]   genotype network {G.number_of_nodes()} nodes / "
          f"{G.number_of_edges()} edges; components per speed {comps}")
    if G.number_of_nodes() > 0:
        pos = nx.spring_layout(G, seed=0)
        phenos = sorted({info[n]["pheno"] for n in G.nodes})
        cmap = {p: OKABE[i % len(OKABE)] for i, p in enumerate(phenos)}
        nx.draw_networkx_edges(G, pos, ax=axs[1], alpha=0.3, width=0.5)
        nx.draw_networkx_nodes(G, pos, ax=axs[1],
                               node_color=[cmap[G.nodes[n]["pheno"]] for n in G.nodes],
                               node_size=[10 + 5 * G.nodes[n]["count"] for n in G.nodes],
                               linewidths=0)
        axs[1].legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap[p],
                      ms=6, label=f"{p} cyc") for p in phenos], fontsize=5.5, loc="best",
                      frameon=False, title="gestation", title_fontsize=6)
    axs[1].set_title("Genotype network\n(edges=1--2 mutations, colour=speed)")
    axs[1].axis("off")

    # Panel 2: redundancy over time (all seeds).
    _plot(axs[2], data, "cycles", "redundancy", "genotypes / phenotype")
    savefig(fig, "fig_gpmap.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-gpmap", dest="gpmap", action="store_false",
                    help="Skip the GP-map figure (fig_gpmap.pdf) -- it is the most "
                         "expensive panel (whole-DB bipartite + genotype network).")
    args = ap.parse_args()

    data = collect()
    if not data:
        print("no run data found"); return
    make_legend()
    if args.gpmap:
        make_gpmap(data)
    else:
        print("[GP-map] skipped (--no-gpmap)")

    # ---- Section 1: population ----
    fig, axs = plt.subplots(1, 4, figsize=(12.5, 2.8))
    _plot(axs[0], data, "cycles", "alive", "living organisms")
    _plot(axs[1], data, "cycles", "self_rep", "self-replicators")
    _plot(axs[2], data, "cycles", "fertile", "fertile organisms")
    for i, s in enumerate(SEEDS):
        if s in data:
            axs[3].plot(data[s]["birth_cycles"], data[s]["births"], color=OKABE[i],
                        lw=1.0, alpha=0.9, label=f"seed {s}")
    axs[3].set_xlabel("cycle"); axs[3].set_ylabel("births / cycle")
    savefig(fig, "fig_population.pdf")

    # ---- Section 2: diversity (genomic | executable phenotype | behavioural phenotype) ----
    fig, axs = plt.subplots(1, 3, figsize=(9.5, 2.8))
    _plot(axs[0], data, "cycles", "sr_unique_raw", "distinct genomes")
    # executable panel: canonical opcodes -- junk dropped and synonymous encodings
    # (different tokens that map to the same command via modulo) collapsed
    _plot(axs[1], data, "cycles", "sr_unique_eff", "distinct programs")
    _plot(axs[2], data, "cycles", "sr_gest_nunique", "distinct replication speeds")
    savefig(fig, "fig_diversity.pdf")

    # ---- Section 3: genome structure (instruction-set / code / registers separately) ----
    fig, axs = plt.subplots(1, 3, figsize=(9.5, 2.8))
    _plot(axs[0], data, "cycles", "header", "instruction-set length (tokens)")
    _plot(axs[1], data, "cycles", "code", "code-section length (tokens)")
    _plot(axs[2], data, "cycles", "registers", "registers per organism")
    savefig(fig, "fig_genome.pdf")

    # ---- Section 4: language ----
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.8))
    _plot(axs[0], data, "cycles", "coding_fraction", "coding fraction")
    _plot(axs[1], data, "cycles", "dialect_shannon", "dialect entropy (nats)")
    savefig(fig, "fig_language.pdf")

    # ---- Section 4b: language -- instructions, recurrence, hierarchy ----
    fig, axs = plt.subplots(1, 3, figsize=(9.5, 2.8))
    for i, s in enumerate(SEEDS):
        if s not in data:
            continue
        c = np.asarray(data[s]["cycles"], float)
        axs[0].plot(c, data[s]["instructions_defined"], color=OKABE[i], lw=1.5)
        axs[0].plot(c, data[s]["instructions_used"], color=OKABE[i], lw=1.0, ls="--", alpha=0.7)
    axs[0].set_xlabel("cycle"); axs[0].set_ylabel("# instructions")
    # solid = total defined in the instruction set, dashed = actually referenced by code
    axs[0].legend(handles=[Line2D([], [], color="0.3", lw=1.5, label="total"),
                           Line2D([], [], color="0.3", lw=1.0, ls="--", label="used")],
                  loc="best", frameon=False)
    _plot(axs[1], data, "cycles", "instruction_reuse", "refs / used instruction")
    _plot(axs[2], data, "cycles", "micro_ops_per_instruction", "micro-ops / instruction")
    savefig(fig, "fig_language2.pdf")

    print("done.")


if __name__ == "__main__":
    main()
