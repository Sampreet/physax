import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from physax.config import (
    Config, UNCLASSIFIED, SELF_REPLICATING, FERTILE, NON_FERTILE, NON_STANDARD,
    FAST_TRACK, SLOW_TRACK, PERCENTILES, OP_NAMES
)
from collections import defaultdict

def compute_diversity_stats(snap):
    alive = snap['alive']
    status = snap['status']
    gest_times = snap['gestation_time']
    hashes = snap['hash']
    if hashes.ndim == 2:
        hashes = (hashes[:, 0].astype(np.int64) << 32) | (hashes[:, 1].astype(np.uint32).astype(np.int64))

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
            probs = counts / np.sum(counts)
            shannon = float(-np.sum(probs * np.log(probs)))
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

def classify_genome(pop, cfg: Config):
    """
    Classify organisms into specific genome categories based on their behavior.
    """
    just_divided_unclassified = pop.alive & pop.has_child & (pop.status == UNCLASSIFIED)
    
    matching_mask = jnp.arange(cfg.max_genome_len) < pop.child_len[:, None]
    is_exact_copy = jnp.all(jnp.where(matching_mask, pop.genome == pop.child, True), axis=-1) & (pop.genome_len == pop.child_len)
    
    # Self-replicating: divided, exact copy, no reading from child
    is_self_rep = is_exact_copy & ~pop.read_from_child
    
    # Non-standard: read from child
    is_non_standard = pop.read_from_child
    
    # Fertile: divided, but not exact copy or read from child
    is_fertile = ~is_self_rep & ~is_non_standard
    
    new_status = jnp.where(
        just_divided_unclassified,
        jnp.where(is_self_rep, jnp.int32(SELF_REPLICATING),
                  jnp.where(is_non_standard, jnp.int32(NON_STANDARD), jnp.int32(FERTILE))),
        pop.status
    )
    
    # Non-fertile: taking too long (> 2000 cycles) to reproduce and hasn't done anything weird
    new_status = jnp.where(
        (new_status == UNCLASSIFIED) & (pop.age > 2000),
        jnp.where(pop.read_from_child, jnp.int32(NON_STANDARD), jnp.int32(NON_FERTILE)),
        new_status
    )
    
    # Record gestation time for those that just divided
    new_gestation = jnp.where(
        just_divided_unclassified,
        pop.age,
        pop.gestation_time
    )
    
    return new_status, new_gestation

def get_execution_route(status, cfg):
    """
    Map statuses to execution routes.
    Fast track: SELF_REPLICATING, NON_FERTILE, FERTILE, NON_STANDARD
    Slow track: UNCLASSIFIED
    """
    is_fast = (status != UNCLASSIFIED) & getattr(cfg, 'caching', True)
    return jnp.where(is_fast, jnp.int32(FAST_TRACK), jnp.int32(SLOW_TRACK))

def compute_cycle_stats(pop, n_births, cfg: Config):
    """
    Compute population statistics for logging.
    """
    alive_count = jnp.sum(pop.alive)
    q_genome_len = jnp.nanpercentile(jnp.where(pop.alive, pop.genome_len, jnp.nan), PERCENTILES)
    
    return {
        'pop_size': alive_count,
        'births': n_births,
        'q_genome_len': q_genome_len,
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
            hashes = (hashes[:, 0].astype(np.int64) << 32) | (hashes[:, 1].astype(np.uint32).astype(np.int64))
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
            
        # Cumulative copies: an approximation based on prevalence over time
        # or we could just use the max prevalence reached
        # Wait, exact cumulative copies requires tracking every birth event.
        # Max prevalence is a good proxy for frequency.
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
