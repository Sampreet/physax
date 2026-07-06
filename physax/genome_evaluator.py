import jax
import jax.numpy as jnp
from jax import random
from physax.virtual_machine import VirtualMachine
from physax.config import UNCLASSIFIED, UP_IS_SIZE
from physax.agent import Agent

def run_batch_until_division(agents, max_steps, keys, cfg):
    vm = VirtualMachine(cfg)
    
    def cond_fun(val):
        a_batch, step, keys, finished_steps = val
        all_divided = jnp.all(a_batch.has_child)
        within_limits = step < max_steps
        return ~all_divided & within_limits

    def body_fun(val):
        a_batch, step, keys, finished_steps = val
        new_a_batch = jax.vmap(vm.execute_one)(a_batch, keys)
        keys = jax.vmap(jax.random.split)(keys)[:, 1]
        
        just_finished = new_a_batch.has_child & ~a_batch.has_child
        new_finished_steps = jnp.where(just_finished, step + 1, finished_steps)
        
        return new_a_batch, step + 1, keys, new_finished_steps

    init_finished = jnp.full(agents.age.shape[0], -1, dtype=jnp.int32)
    final_agents, final_steps, _, final_finished_steps = jax.lax.while_loop(
        cond_fun, body_fun, (agents, jnp.int32(0), keys, init_finished)
    )
    return final_agents, final_steps, keys, final_finished_steps


def _evaluate_batch(genomes, genome_lens, max_steps, key, cfg):
    """Run a batch of genomes to (attempted) division. Returns, per genome:
    divides (produced any child) and self_replicates (that child is an exact copy
    of the genome and nothing was read from the child -- the classify_genome
    definition of a true self-replicator)."""
    n = genomes.shape[0]
    agents = jax.vmap(
        lambda g, l: Agent.init_organism(
            g, l, jnp.int32(-1), jnp.int32(UNCLASSIFIED), jnp.int32(-1), cfg
        )
    )(genomes, genome_lens)

    keys = random.split(key, n)
    final_agents, _, _, _ = run_batch_until_division(agents, jnp.int32(max_steps), keys, cfg)

    divides = final_agents.has_child
    matching = jnp.arange(cfg.max_genome_len)[None, :] < final_agents.child_len[:, None]
    is_exact = (
        jnp.all(jnp.where(matching, final_agents.genome == final_agents.child, True), axis=-1)
        & (final_agents.genome_len == final_agents.child_len)
    )
    self_replicates = divides & is_exact & ~final_agents.read_from_child
    return divides, self_replicates


def compute_mutational_robustness(genomes, genome_lens, max_steps, key, cfg, n_mutants=32):
    """Monte-Carlo estimate of point-mutational robustness for a set of genomes.

    For each genome we draw n_mutants single-point substitutions (a random
    position < genome_len set to a random opcode in [0, UP_IS_SIZE) -- the same
    alphabet the copy/point mutation operators draw from) and re-run them through
    the VM. Robustness is the fraction of mutants whose replication phenotype
    survives. The unmutated genomes are evaluated too as a baseline, so a budget
    that is too short to let the originals divide is visible rather than silently
    depressing the score.

    Returns a dict of population-mean scalars plus per-genome arrays.
    """
    n, L = genomes.shape
    k_pos, k_val, k_orig, k_mut = random.split(key, 4)

    # Per-(genome, mutant) point substitution.
    pos = random.randint(k_pos, (n, n_mutants), 0, jnp.maximum(genome_lens, 1)[:, None])
    val = random.randint(k_val, (n, n_mutants), 0, UP_IS_SIZE).astype(genomes.dtype)
    base = jnp.broadcast_to(genomes[:, None, :], (n, n_mutants, L))
    row = jnp.arange(n)[:, None]
    col = jnp.arange(n_mutants)[None, :]
    mutants = base.at[row, col, pos].set(val)                      # (n, n_mutants, L)
    mutant_lens = jnp.broadcast_to(genome_lens[:, None], (n, n_mutants))

    # Baseline: do the originals replicate under this budget?
    base_div, base_sr = _evaluate_batch(genomes, genome_lens, max_steps, k_orig, cfg)

    # Mutants, evaluated as one flat batch.
    flat = mutants.reshape(n * n_mutants, L)
    flat_lens = mutant_lens.reshape(n * n_mutants)
    mut_div, mut_sr = _evaluate_batch(flat, flat_lens, max_steps, k_mut, cfg)
    mut_div = mut_div.reshape(n, n_mutants)
    mut_sr = mut_sr.reshape(n, n_mutants)

    # Per-genome robustness = fraction of that genome's mutants surviving.
    div_robustness = jnp.mean(mut_div.astype(jnp.float32), axis=1)
    sr_robustness = jnp.mean(mut_sr.astype(jnp.float32), axis=1)

    return {
        'baseline_divides': float(jnp.mean(base_div.astype(jnp.float32))),
        'baseline_self_replicates': float(jnp.mean(base_sr.astype(jnp.float32))),
        'robustness_divide': float(jnp.mean(div_robustness)),
        'robustness_self_replicate': float(jnp.mean(sr_robustness)),
        'per_genome_robustness_self_replicate': jax.device_get(sr_robustness),
        'n_genomes': int(n),
        'n_mutants': int(n_mutants),
    }
