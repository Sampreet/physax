"""Simulation configuration and run-level constants for Physis.

This module owns the *tunable* parts of a run — everything a caller might want
to change between experiments:

  * ``Config`` / ``make_config`` — fixed-size array bounds, population size,
    mutation rates, and behavioural thresholds (all sizes are static so JAX can
    compile against them).
  * ``GenomeDB`` sizing (``HASH_TABLE_SIZE`` / ``EMPTY_KEY``).
  * Genome-classification statuses (``UNCLASSIFIED`` … ``NON_STANDARD``) and
    execution routes (``FAST_TRACK`` / ``SLOW_TRACK``).
  * Plotting defaults (``LS`` line styles, ``PERCENTILES``).

The virtual machine's *instruction set* (opcodes, operand tables, ``OpState`` /
``OpArgs``, tape helpers, ``get_opcode_functions``) lives in :mod:`physax.sim.isa`.
It is re-exported here via ``from physax.sim.isa import *`` so that existing
``from physax.sim.config import ...`` call sites (opcode constants, ``OP_NAMES``,
``N_OPERANDS``, etc.) continue to work unchanged.
"""
import jax.numpy as jnp

# Re-export the instruction set so `from physax.sim.config import <opcode/...>`
# keeps working. New code should prefer importing these from `physax.sim.isa`.
from physax.sim.isa import *  # noqa: F401,F403

# GenomeDB hash-table sizing (see physax.sim.model.GenomeDB).
HASH_TABLE_SIZE = 1048576 * 2
EMPTY_KEY = -1

# Plotting values for percentile plots.
LS = ['dotted', 'dashdot', 'dashed', 'solid', 'dashed', 'dashdot', 'dotted']
PERCENTILES = jnp.array([5, 10, 25, 50, 75, 90, 95])


class Config:
    """Simulation configuration with fixed sizes for JAX."""
    max_genome_len: int = 256
    max_se_count: int = 16
    max_instructions: int = 64
    max_micro_ops: int = 32
    pop_size: int = 1024
    initial_pop: int = 1

    steps_per_update: int = 34
    copy_mutation_rate: float = 0.009
    divide_mutation_rate: float = 0.0
    divide_insert_rate: float = 0.0013
    divide_delete_rate: float = 0.0013
    min_allocation_ratio: float = 0.5
    max_allocation_ratio: float = 2.0
    min_proliferation_ratio: float = 0.80

    use_species_color: bool = True
    caching: bool = True


def make_config(**kwargs) -> Config:
    """Create config with optional overrides."""
    cfg = Config()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


# Genome Classification Status
UNCLASSIFIED = 0
SELF_REPLICATING = 1
FERTILE = 2
NON_FERTILE = 3
NON_STANDARD = 4

# Execution Routes
FAST_TRACK = 0
SLOW_TRACK = 1
