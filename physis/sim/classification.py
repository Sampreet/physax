"""Per-cycle genome classification and routing (JAX, runs inside the sim loop).

These are the jittable functions the population loop calls every cycle:
classify each organism by its division behaviour, map a status to a fast/slow
execution route, and compute the lightweight per-cycle logging stats. Offline
genome/language statistics live in :mod:`physis.analysis.genome_stats`.
"""
import jax.numpy as jnp

from physis.sim.config import (
    Config, UNCLASSIFIED, SELF_REPLICATING, FERTILE, NON_FERTILE, NON_STANDARD,
    FAST_TRACK, SLOW_TRACK, PERCENTILES,
)


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
