"""Offline genome and language statistics over population snapshots (NumPy).

Diversity/gestation summaries, the emergent-"language" metrics (vocabulary,
instruction-set composition, dialects), genome-to-text formatting, and the
top-genomes-over-time plot. These consume saved snapshots/stats after a run and
are not part of the jitted simulation loop (that is
:mod:`physax.sim.classification`).
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from physax.sim.config import (
    Config, UNCLASSIFIED, SELF_REPLICATING, FERTILE, NON_FERTILE,
    OP_NAMES, UP_IS_SIZE, N_OPERANDS,
)
from physax.analysis._util import fold_hash, shannon_entropy


def compute_diversity_stats(snap):
    alive = snap['alive']
    status = snap['status']
    gest_times = snap['gestation_time']
    hashes = snap['hash']
    if hashes.ndim == 2:
        hashes = fold_hash(hashes)

    mask_sr = alive & (status == SELF_REPLICATING)
    mask_f = alive & (status == FERTILE)

    stats_out = {}

    for label, mask in [('self_rep', mask_sr), ('fertile', mask_f)]:
        if np.any(mask):
            gests = gest_times[mask]
            gests = gests[gests < 2000000000]
            avg_gest = np.mean(gests) if len(gests) > 0 else np.nan

            h = hashes[mask]
            uniq, counts = np.unique(h, return_counts=True)
            total = int(np.sum(mask))
            unique = int(len(uniq))
            shannon = shannon_entropy(counts)
        else:
            avg_gest = np.nan
            total = 0
            unique = 0
            shannon = 0.0

        stats_out[f'avg_gest_{label}'] = avg_gest
        stats_out[f'total_{label}'] = total
        stats_out[f'unique_{label}'] = unique
        stats_out[f'shannon_div_{label}'] = shannon

    return stats_out


def format_genome(gen_arr, length):
    genes = [int(x) for x in gen_arr[:length]]
    names = [OP_NAMES.get(g, str(g)) for g in genes]
    return f"[{', '.join(names)}]"


def compute_language_stats(pop, cfg: Config):
    """Metrics describing the emergent *language*, in two families matching how a
    genome is structured:

      A. Genotype -- the raw token sequence itself.
      B. Genotype->phenotype mapping -- the header before SEP, which defines each
         organism's own registers (R/S/Q) and instruction set (I markers), i.e.
         the language its post-SEP program is written in.

    All reductions are over the living population. Returns a dict of scalars plus
    two length-UP_IS_SIZE count vectors (for opcode/symbol bar charts), or None if
    nobody is alive.
    """
    alive = np.asarray(pop.alive)
    if not np.any(alive):
        return None

    genome = np.asarray(pop.genome)[alive]              # (n, L) raw tokens
    glen = np.asarray(pop.genome_len)[alive].astype(np.int64)
    sep = np.asarray(pop.separator_pos)[alive].astype(np.int64)
    n_instr = np.asarray(pop.n_instructions)[alive].astype(np.int64)
    itable = np.asarray(pop.instruction_table)[alive]   # (n, M, K) parsed micro-ops
    ilens = np.asarray(pop.instruction_lengths)[alive]  # (n, M)
    executed = np.asarray(pop.executed)[alive]          # (n, L) bool
    n_ses = np.asarray(pop.n_ses)[alive].astype(np.int64)

    n, L = genome.shape
    M, K = itable.shape[1], itable.shape[2]
    in_genome = np.arange(L)[None, :] < glen[:, None]   # valid-token mask
    n_ops = np.asarray(N_OPERANDS)

    stats = {}

    # ---- A. Genotype: the raw token sequence -------------------------------
    # A2. Coding fraction: share of a genome's tokens the VM ever executed --
    # a "junk DNA" measure. executed is only ever set within genome_len.
    coding = np.where(glen > 0, np.sum(executed & in_genome, axis=1) / np.maximum(glen, 1), 0.0)
    stats['coding_fraction'] = float(np.mean(coding))

    # A3. Header/code split: SEP divides the language definition from the
    # program that uses it.
    stats['header_fraction'] = float(np.mean(np.where(glen > 0, sep / np.maximum(glen, 1), 0.0)))
    stats['code_section_len'] = float(np.mean(np.maximum(glen - sep - 1, 0)))
    stats['registers_mean'] = float(np.mean(n_ses))

    # A1. Vocabulary: pooled distribution of raw symbols (clamped to the opcode
    # range; operands/code-refs outside it are dropped from the alphabet view).
    tokens = genome[in_genome]
    tokens = tokens[(tokens >= 0) & (tokens < UP_IS_SIZE)]
    symbol_counts = np.bincount(tokens, minlength=UP_IS_SIZE)[:UP_IS_SIZE]
    stats['genome_symbol_entropy'] = shannon_entropy(symbol_counts)
    stats['genome_vocab_size'] = int(np.sum(symbol_counts > 0))

    # ---- B. Genotype->phenotype mapping: the parsed instruction set --------
    # Recover which instruction-table positions are opcodes by re-walking the
    # operand-count chain (mirrors Agent._build_instruction_set), vectorised over
    # all (organism, instruction) pairs, looping only over the K micro-op slots.
    next_op = np.zeros((n, M), dtype=np.int64)
    is_op = np.zeros((n, M, K), dtype=bool)
    for j in range(K):
        val = itable[:, :, j]
        at_op = (next_op == j) & (j < ilens)
        is_op[:, :, j] = at_op
        nj = n_ops[np.clip(val, 0, UP_IS_SIZE - 1)]
        next_op = np.where(at_op, j + 1 + nj, next_op)

    # B5. Instruction-set composition: which primitives the language is built from.
    opcodes = itable[is_op]
    opcodes = opcodes[(opcodes >= 0) & (opcodes < UP_IS_SIZE)]
    opcode_counts = np.bincount(opcodes, minlength=UP_IS_SIZE)[:UP_IS_SIZE]
    stats['opcode_entropy'] = shannon_entropy(opcode_counts)
    stats['opcode_vocab_size'] = int(np.sum(opcode_counts > 0))

    # Per-organism metrics needing genome slicing / hashing -- one pass.
    # B6/B7. Referenced instructions: post-SEP tokens index the table via
    #        abs(token) % n_instructions (as in VirtualMachine.execute_one).
    # B8.    Dialect: two organisms sharing a header (genome[:sep+1]) define the
    #        same language even if their programs differ.
    eff_instr = []
    dead_frac = []
    n_defined = []
    reuse = []
    mops_per_instr = []
    dialects = []
    for i in range(n):
        end = int(glen[i])
        s = int(sep[i])
        header = genome[i, :min(s + 1, end)]
        dialects.append(hash(header.tobytes()))

        ni = int(n_instr[i])
        if ni <= 0:
            continue
        n_defined.append(ni)                            # B9. instructions defined
        # B10. Hierarchy proxy: micro-ops bundled per defined instruction.
        li = ilens[i, :ni]; li = li[li > 0]
        if li.size:
            mops_per_instr.append(float(np.mean(li)))
        code = genome[i, s + 1:end]
        if code.size == 0:
            refd = 0
        else:
            refd = int(np.unique(np.abs(code) % ni).size)
            # B11. Recurrence: how many times each *used* instruction is invoked.
            reuse.append(code.size / max(refd, 1))
        eff_instr.append(refd)
        dead_frac.append((ni - refd) / ni)

    stats['effective_instructions'] = float(np.mean(eff_instr)) if eff_instr else 0.0
    stats['dead_instruction_frac'] = float(np.mean(dead_frac)) if dead_frac else 0.0
    stats['instructions_defined'] = float(np.mean(n_defined)) if n_defined else 0.0
    stats['instruction_reuse'] = float(np.mean(reuse)) if reuse else 0.0
    stats['micro_ops_per_instruction'] = float(np.mean(mops_per_instr)) if mops_per_instr else 0.0

    dialects = np.array(dialects)
    uniq, counts = np.unique(dialects, return_counts=True)
    stats['distinct_dialects'] = int(uniq.size)
    stats['dialect_shannon'] = shannon_entropy(counts)

    # A4. Within-genome symbol entropy: internal redundancy / compressibility.
    per_genome_ent = []
    for i in range(n):
        row = genome[i, :int(glen[i])]
        row = row[(row >= 0) & (row < UP_IS_SIZE)]
        per_genome_ent.append(shannon_entropy(np.bincount(row, minlength=UP_IS_SIZE)))
    stats['genome_entropy_mean'] = float(np.mean(per_genome_ent)) if per_genome_ent else 0.0

    return {
        'scalars': stats,
        'symbol_counts': symbol_counts,
        'opcode_counts': opcode_counts,
    }


def analyze_and_plot_top_genomes(all_stats, filename="top_genomes.png", start_cycle=0, only_self_replicators=True):
    """
    Analyze the saved chunk records, extract the prevalence of self-replicating
    genomes over time, and plot the top 10 most frequent ones.
    """
    # Track prevalence of each hash at each recorded cycle
    # hash -> {cycle: count}
    prevalence = defaultdict(lambda: defaultdict(int))
    # hash -> cumulative births/copies
    cumulative_copies = defaultdict(int)
    # hash -> min gestation time observed
    gestation_times = {}

    cycles = []

    for chunk in all_stats:
        cycle = chunk['cycle']
        if cycle < start_cycle:
            continue
        cycles.append(cycle)
        snap = chunk['snapshot']

        alive = snap['alive']
        status = snap['status']
        hashes = snap['hash']
        if hashes.ndim == 2:
            hashes = fold_hash(hashes)
        gest_times = snap['gestation_time']

        if only_self_replicators:
            alive_self_rep_mask = alive & (status == SELF_REPLICATING)
        else:
            alive_self_rep_mask = alive & (status != UNCLASSIFIED) & (status != NON_FERTILE)
        valid_hashes = hashes[alive_self_rep_mask]
        valid_gest = gest_times[alive_self_rep_mask]

        # Count prevalence at this cycle
        unique_hashes, counts = np.unique(valid_hashes, return_counts=True)
        for h, c in zip(unique_hashes, counts):
            prevalence[h][cycle] += c

        # Cumulative copies: approximated by the max prevalence a genome reaches.
        # (Exact cumulative copies would require tracking every birth event.)
        for h in unique_hashes:
            cumulative_copies[h] = max(cumulative_copies[h], prevalence[h][cycle])

        # Update min gestation time
        for h, gt in zip(valid_hashes, valid_gest):
            if gt < 2000000000:
                if h not in gestation_times:
                    gestation_times[h] = gt
                else:
                    gestation_times[h] = min(gestation_times[h], gt)

    if not cumulative_copies:
        print("No self-replicating genomes found to plot.")
        return

    top_n = 10
    # Find top 10 most frequent hashes
    top_10 = sorted(cumulative_copies.items(), key=lambda x: x[1], reverse=True)[:top_n]

    plt.figure(figsize=(12, 8))
    for h, _ in top_10:
        timeseries = [prevalence[h].get(c, 0) for c in cycles]
        gest = gestation_times.get(h, "N/A")
        plt.plot(cycles, timeseries, label=f"Hash: {h} (Gest: {gest})")

    plt.xlabel("Cycle")
    plt.ylabel("Population Count")
    plt.title(f"Top {top_n} {'Self-Replicating' if only_self_replicators else 'All Fertile'} Genomes Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved top {top_n} genomes plot to {filename}")

    return [h for h, _ in top_10]
