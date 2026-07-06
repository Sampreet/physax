import os
import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random
from functools import partial
from tqdm import trange
import os
import pickle
import physax.wandb_logger as wandb_logger

from physax.config import *
from physax.agent import Agent
from physax.virtual_machine import VirtualMachine
from physax.genome_analysis import classify_genome, get_execution_route, compute_cycle_stats, compute_language_stats
from physax.genome_evaluator import compute_mutational_robustness
from typing import NamedTuple

# Percentiles logged per chunk for each genome-part distribution (registers, instructions,
# op-codes per instruction). Wider than a lone median so wandb shows the spread and tails.
GENOME_PART_PCTS = jnp.array([5, 25, 50, 75, 95])

class GenomeDB(NamedTuple):
    keys: jnp.ndarray
    statuses: jnp.ndarray
    gestations: jnp.ndarray
    child_genomes: jnp.ndarray
    child_lens: jnp.ndarray

def init_genome_db(cfg: Config):
    return GenomeDB(
        keys=jnp.full((HASH_TABLE_SIZE + 1, 2), EMPTY_KEY, dtype=jnp.int32),
        statuses=jnp.full(HASH_TABLE_SIZE + 1, UNCLASSIFIED, dtype=jnp.int32),
        gestations=jnp.full(HASH_TABLE_SIZE + 1, 2147483647, dtype=jnp.int32),
        child_genomes=jnp.full((HASH_TABLE_SIZE + 1, cfg.max_genome_len), BLANK, dtype=jnp.int32),
        child_lens=jnp.zeros(HASH_TABLE_SIZE + 1, dtype=jnp.int32)
    )

def lookup_db(hashes, db: GenomeDB):
    def lookup_one(h):
        start_hash = jnp.abs(h[0])
        def body_fn(i, state):
            found, idx = state
            probe_idx = (start_hash + i) % HASH_TABLE_SIZE
            key = db.keys[probe_idx]
            is_match = jnp.all(key == h) & (h[0] != EMPTY_KEY)
            new_found = found | is_match
            new_idx = jnp.where(is_match & ~found, probe_idx, idx)
            return new_found, new_idx
        found, idx = lax.fori_loop(0, 32, body_fn, (jnp.bool_(False), jnp.int32(-1)))
        return found, idx
    return jax.vmap(lookup_one)(hashes)

def add_to_db(db: GenomeDB, pop: Agent, mask: jnp.ndarray, cfg: Config):
    # Only add valid, newly classified genomes (Self-Rep, Fertile, Non-Fertile, Non-Standard)
    # Since VM is deterministic, we can cache all classified genomes.
    to_add = mask & (pop.status != UNCLASSIFIED)

    def find_slot(h):
        start_hash = jnp.abs(h[0])
        def body_fn(i, state):
            found, idx = state
            probe_idx = (start_hash + i) % HASH_TABLE_SIZE
            key = db.keys[probe_idx]
            is_valid = jnp.all(key == EMPTY_KEY) | jnp.all(key == h)
            new_found = found | is_valid
            new_idx = jnp.where(is_valid & ~found, probe_idx, idx)
            return new_found, new_idx
        found, idx = lax.fori_loop(0, 32, body_fn, (jnp.bool_(False), jnp.int32(-1)))
        return idx
    
    add_indices = jax.vmap(find_slot)(pop.genome_hash)
    
    # To avoid a race condition where multiple agents map to the same hash index 
    # but some have to_add=False (overwriting the valid update with db.keys[idx]),
    # we remap the invalid items to an out-of-bounds index and use mode='drop'.
    # We also ignore items where find_slot returned -1 (DB full).
    valid_mask = mask & (add_indices != -1)
    safe_indices = jnp.where(valid_mask, add_indices, db.keys.shape[0])
    
    db = db._replace(
        keys=db.keys.at[safe_indices].set(pop.genome_hash, mode='drop'),
        statuses=db.statuses.at[safe_indices].set(pop.status, mode='drop'),
        gestations=db.gestations.at[safe_indices].set(pop.gestation_time, mode='drop'),
        child_genomes=db.child_genomes.at[safe_indices].set(pop.child, mode='drop'),
        child_lens=db.child_lens.at[safe_indices].set(pop.child_len, mode='drop')
    )
    return db

global_self_replicating_genomes = {}
global_fertile_genomes = {}

def collect_self_replicating(hashes, genomes, mask):
    hashes_np = np.array(hashes)
    genomes_np = np.array(genomes)
    mask_np = np.array(mask)
    valid_indices = np.where(mask_np)[0]
    for idx in valid_indices:
        h = (int(hashes_np[idx, 0]), int(hashes_np[idx, 1]))
        if h not in global_self_replicating_genomes:
            global_self_replicating_genomes[h] = np.copy(genomes_np[idx])

def collect_fertile(hashes, genomes, mask):
    hashes_np = np.array(hashes)
    genomes_np = np.array(genomes)
    mask_np = np.array(mask)
    valid_indices = np.where(mask_np)[0]
    for idx in valid_indices:
        h = (int(hashes_np[idx, 0]), int(hashes_np[idx, 1]))
        if h not in global_fertile_genomes:
            global_fertile_genomes[h] = np.copy(genomes_np[idx])

class Model:
    """Simulation manager tying together configuration, VM, and Agent population."""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.vm = VirtualMachine(cfg)

    def init_population(self, key) -> Agent:
        """Initialize population with ancestor genomes at random positions."""
        ancestor_genome, ancestor_len = Agent.create_ancestor_genome(self.cfg)

        k1, k2 = random.split(key)
        perm = random.permutation(k1, self.cfg.pop_size)
        alive_indices = perm[:self.cfg.initial_pop]

        is_alive = jnp.zeros(self.cfg.pop_size, dtype=jnp.bool_)
        is_alive = is_alive.at[alive_indices].set(True)

        def init_one(i, alive_mask_val):
            state = Agent.init_organism(
                ancestor_genome, ancestor_len, 
                jnp.int32(-1), jnp.int32(UNCLASSIFIED), jnp.int32(-1), 
                self.cfg
            )
            return state._replace(alive=alive_mask_val)

        pop = jax.vmap(init_one)(jnp.arange(self.cfg.pop_size), is_alive)

        # Seed lineage: give every slot a unique id in [0, pop_size). Births in
        # later cycles draw ids >= pop_size (see cycle_step), so ids never collide.
        pop = pop._replace(
            id=jnp.arange(self.cfg.pop_size, dtype=jnp.int32),
            parent_id=jnp.full(self.cfg.pop_size, -1, dtype=jnp.int32),
            birth_cycle=jnp.zeros(self.cfg.pop_size, dtype=jnp.int32),
        )
        return pop

    def _place_children_on_grid(self, pop: Agent, has_child: jnp.ndarray, child_states: Agent, key, cycle_idx) -> tuple[Agent, jnp.ndarray, jnp.ndarray]:
        """Calculate spatial placement on a 2D toroidal grid using the OldestNurse algorithm."""
        grid_side = int(np.ceil(np.sqrt(self.cfg.pop_size)))
        W = grid_side
        H = (self.cfg.pop_size + W - 1) // W

        parent_indices = jnp.arange(self.cfg.pop_size)
        y = parent_indices // W
        x = parent_indices % W

        dy = jnp.array([-1, -1, -1, 0, 0, 1, 1, 1])
        dx = jnp.array([-1, 0, 1, -1, 1, -1, 0, 1])

        ny = (y[:, None] + dy[None, :]) % H
        nx = (x[:, None] + dx[None, :]) % W

        neighbor_indices = ny * W + nx
        neighbor_valid = neighbor_indices < self.cfg.pop_size
        neighbor_indices_safe = jnp.where(neighbor_valid, neighbor_indices, parent_indices[:, None])

        neighbor_alive = pop.alive[neighbor_indices_safe]
        neighbor_age = pop.age[neighbor_indices_safe]

        # Score: empty cells get very high score, alive cells get their age
        base_score = jnp.where(neighbor_alive, neighbor_age.astype(jnp.float32), 1e5)
        valid_score = jnp.where(neighbor_valid, base_score, -1.0)

        # Break ties randomly
        noise = random.uniform(key, (self.cfg.pop_size, 8)) * 0.5
        final_score = valid_score + noise

        best_neighbor_local_idx = jnp.argmax(final_score, axis=1)
        target_indices = jnp.take_along_axis(
            neighbor_indices_safe, best_neighbor_local_idx[:, None], axis=1
        ).squeeze(1)

        # Place children with an O(N) scatter instead of an O(N^2) match matrix.
        # Each birthing parent competes for its target cell; when several parents
        # target the same cell the lowest parent index wins (this reproduces the
        # tie-break of the old argmax-over-parents approach).
        NO_PARENT = jnp.int32(self.cfg.pop_size)
        parent_ids = jnp.arange(self.cfg.pop_size, dtype=jnp.int32)
        contrib = jnp.where(has_child, parent_ids, NO_PARENT)   # non-birthing parents don't compete
        winner = jnp.full(self.cfg.pop_size, NO_PARENT, dtype=jnp.int32)
        winner = winner.at[target_indices].min(contrib)        # segmented scatter-min, O(N)

        # any_child_placed[j] = a child was placed at cell j; source_parent[j] = its parent
        any_child_placed = winner < NO_PARENT
        source_parent = jnp.where(any_child_placed, winner, 0)  # clamp unplaced cells to a valid index

        # Gather child states from the winning parents
        gathered_children = jax.tree.map(lambda x: x[source_parent], child_states)

        # Assign lineage to each placed child. The child in cell j gets a globally
        # unique id (cycle_idx * pop_size + j) and records the id of its parent
        # (the organism at source_parent[j]). Cells with no birth keep pop's values.
        new_ids = cycle_idx * self.cfg.pop_size + jnp.arange(self.cfg.pop_size, dtype=jnp.int32)
        parent_ids_for_cell = pop.id[source_parent]
        gathered_children = gathered_children._replace(
            id=new_ids,
            parent_id=parent_ids_for_cell,
            birth_cycle=jnp.full(self.cfg.pop_size, cycle_idx, dtype=jnp.int32),
        )

        # Conditionally replace: cell j gets child state if a child was placed, else keeps current
        new_pop = jax.tree.map(
            lambda c, p: jnp.where(
                any_child_placed.reshape((-1,) + (1,) * (c.ndim - 1)),
                c, p
            ),
            gathered_children, pop
        )
        # Birth edges for genealogy (-1 where no child was placed this cycle)
        child_id_edge = jnp.where(any_child_placed, new_ids, jnp.int32(-1))
        parent_id_edge = jnp.where(any_child_placed, parent_ids_for_cell, jnp.int32(-1))
        return new_pop, child_id_edge, parent_id_edge

    def cycle_step(self, pop: Agent, db: GenomeDB, cycle_idx, key) -> tuple[Agent, GenomeDB, dict]:
        """Execute one cycle using the JAX VM interpreter (reference path).

        cycle_idx is a monotonically increasing counter (starts at 1) used to mint
        unique child ids and to stamp birth cycles for lineage tracking. Composes
        the three phases; _post_vm re-splits `key` so the RNG stream matches the
        kernel path exactly.
        """
        pop, is_slow = self._pre_vm(pop, db)
        k_exec = random.split(key, 3)[0]
        pop = self._vm_jax(pop, is_slow, k_exec)
        return self._post_vm(pop, db, cycle_idx, key, is_slow)

    def _pre_vm(self, pop: Agent, db: GenomeDB) -> tuple[Agent, jnp.ndarray]:
        """Cycle phase before the VM: DB cache lookup and the fast-track step.
        Returns the updated population and the slow-track mask. Split out so the
        VM between _pre_vm and _post_vm can be either the JAX interpreter or the
        CUDA kernel. Uses no randomness, so it needs no key.
        """
        # 0. Check DB for unclassified/new agents
        needs_lookup = pop.alive & (pop.status == UNCLASSIFIED)
        found, db_idx = lookup_db(pop.genome_hash, db)
        
        apply_cache = needs_lookup & found
        pop = pop._replace(
            status=jnp.where(apply_cache, db.statuses[db_idx], pop.status),
            gestation_time=jnp.where(apply_cache, db.gestations[db_idx], pop.gestation_time)
        )

        # Determine execution routes based on genome classification
        routes = get_execution_route(pop.status, self.cfg)
        is_fast = routes == FAST_TRACK
        is_slow = pop.alive & (routes == SLOW_TRACK)

        # 1. Execution Phase
        # Fast Track
        fast_age = jnp.where(pop.alive & is_fast, pop.age + 1, pop.age)
        fast_divide = pop.alive & is_fast & (pop.gestation_time > 0) & ((fast_age % pop.gestation_time) == 0) & (pop.status != NON_FERTILE)
        
        _, div_db_idx = lookup_db(pop.genome_hash, db)
        
        pop = pop._replace(
            age=fast_age,
            has_child=fast_divide,
            child=jnp.where(fast_divide[:, None], db.child_genomes[div_db_idx], pop.child),
            child_len=jnp.where(fast_divide, db.child_lens[div_db_idx], pop.child_len),
        )
        return pop, is_slow

    def _vm_jax(self, pop: Agent, is_slow: jnp.ndarray, k_exec) -> Agent:
        """Slow-track VM via the JAX interpreter: run the VM on the whole
        population and keep the result only for is_slow agents. This is the
        reference path that the CUDA kernel (VMRunner) reproduces."""
        exec_keys = random.split(k_exec, self.cfg.pop_size)
        p_exec = jax.vmap(self.vm.update)(pop, exec_keys)
        return jax.tree.map(
            lambda new, old: jnp.where(is_slow.reshape((-1,) + (1,) * (new.ndim - 1)), new, old),
            p_exec, pop
        )

    def _post_vm(self, pop: Agent, db: GenomeDB, cycle_idx, key, is_slow: jnp.ndarray,
                 do_collect: bool = True) -> tuple[Agent, GenomeDB, dict]:
        """Cycle phases after the VM: slow-track aging, classification, birth,
        spatial placement, cleanup, and stats. Runs identically whether the VM
        was the JAX interpreter or the CUDA kernel. Only k_birth/k_place are used
        here (k_exec drove the VM), but we re-split the same key so the RNG stream
        is identical to the original single-function cycle_step.

        do_collect gates the two genome-collection callbacks. They only populate
        the global self-replicator/fertile dictionaries used for end-of-run
        reports -- they do NOT affect pop/db -- but each `jax.debug.callback`
        transfers the full genome array to host and stalls the pipeline (~100 ms/
        cycle at pop 16384). It is a *static* flag so that when False the callback
        nodes are compiled out entirely (no host transfer), letting the Python-
        driven kernel loop collect only every N cycles. pop/db stay bitwise-
        identical regardless of its value."""
        k_exec, k_birth, k_place = random.split(key, 3)

        # 2. Aging Phase for slow track
        pop = pop._replace(age=jnp.where(pop.alive & is_slow, pop.age + 1, pop.age))

        # 3. Classification Phase
        new_status, new_gestation = classify_genome(pop, self.cfg)
        
        if do_collect:
            # Archiving newly-classified genomes needs a host transfer via
            # jax.debug.callback, whose fixed overhead is ~119 ms/cycle at pop
            # 16384 -- and it dominated runtime once the VM moved to the kernel.
            # A genome is actually newly classified in only ~8% of cycles, so we
            # fire each callback under lax.cond(any(mask)): the collected set is
            # bitwise-identical to firing every cycle, but the transfer is skipped
            # on the ~92% of cycles with nothing new. (do_collect still allows an
            # additional coarser cadence via --collect_interval.)
            newly_self_rep = pop.alive & (new_status == SELF_REPLICATING) & (pop.status != SELF_REPLICATING)
            lax.cond(
                jnp.any(newly_self_rep),
                lambda: jax.debug.callback(collect_self_replicating, pop.genome_hash, pop.genome, newly_self_rep),
                lambda: None,
            )
            newly_fertile = pop.alive & (new_status == FERTILE) & (pop.status != FERTILE)
            lax.cond(
                jnp.any(newly_fertile),
                lambda: jax.debug.callback(collect_fertile, pop.genome_hash, pop.genome, newly_fertile),
                lambda: None,
            )

        just_classified = pop.alive & (new_status != UNCLASSIFIED) & (pop.status == UNCLASSIFIED)
        pop = pop._replace(status=new_status, gestation_time=new_gestation)
        
        db = add_to_db(db, pop, just_classified, self.cfg)

        # 4. Birth Phase (Mutation & Parsing)
        has_child = pop.has_child
        n_births = jnp.sum(has_child)

        # Prepare child tapes for organisms that divided
        # Copy child + child_len into child_tape + child_tape_len
        pop = pop._replace(
            child_tape=jnp.where(has_child[:, None], pop.child, pop.child_tape),
            child_tape_len=jnp.where(has_child, pop.child_len, pop.child_tape_len)
        )

        # Apply divide mutations
        mut_keys = random.split(k_birth, self.cfg.pop_size)

        def mutate_one(mut_key, tape, tape_len, has, status):
            new_tape, new_len = Agent.apply_divide_mutations(mut_key, tape, tape_len, status, self.cfg)
            tape_out = jnp.where(has, new_tape, tape)
            len_out = jnp.where(has, new_len, tape_len)
            return tape_out, len_out

        mutated_tapes, mutated_lens = jax.vmap(mutate_one)(
            mut_keys, pop.child_tape, pop.child_tape_len, has_child, pop.status
        )

        child_states = jax.vmap(Agent.init_organism, in_axes=(0, 0, 0, 0, 0, None))(
            mutated_tapes, mutated_lens, 
            pop.genome_hash, pop.status, pop.gestation_time,
            self.cfg
        )

        # 5. Spatial Placement Phase
        pop, child_id_edge, parent_id_edge = self._place_children_on_grid(
            pop, has_child, child_states, k_place, cycle_idx
        )

        # 5. Cleanup Phase
        blank_child = jnp.full(self.cfg.max_genome_len, BLANK, dtype=jnp.int32)
        
        pop = pop._replace(
            child=jnp.where(has_child[:, None], blank_child, pop.child),
            child_len=jnp.where(has_child, jnp.int32(0), pop.child_len),
            child_copied=jnp.where(
                has_child[:, None],
                jnp.zeros(self.cfg.max_genome_len, dtype=jnp.bool_),
                pop.child_copied
            ),
            already_allocated=jnp.where(has_child, jnp.bool_(False), pop.already_allocated),
            has_child=jnp.zeros(self.cfg.pop_size, dtype=jnp.bool_),
            child_tape=jnp.where(
                has_child[:, None],
                jnp.full(self.cfg.max_genome_len, BLANK, dtype=jnp.int32),
                pop.child_tape
            ),
            child_tape_len=jnp.where(has_child, jnp.int32(0), pop.child_tape_len)
        )

        # Stats
        stats = compute_cycle_stats(pop, n_births, self.cfg)
        stats['has_child'] = has_child
        # Genealogy edge list for this cycle: for each grid cell, the id of the
        # child born there and the id of its parent (-1 where no birth). Stacked
        # over the scan into (log_interval, pop_size) arrays.
        stats['child_id'] = child_id_edge
        stats['parent_id'] = parent_id_edge
        return pop, db, stats

    def _robustness_of_self_replicators(self, pop, cycle_num, n_genomes=16, n_mutants=32):
        """Sample up to n_genomes living self-replicators and estimate their
        point-mutational robustness via genome_evaluator. Returns None if there
        are no living self-replicators. Kept off the per-chunk path -- it re-runs
        n_genomes * n_mutants genomes through the VM -- so callers gate it on the
        infrequent report cadence."""
        sr_mask = np.array(pop.alive) & (np.array(pop.status) == SELF_REPLICATING)
        idx = np.nonzero(sr_mask)[0]
        if idx.size == 0:
            return None

        if idx.size > n_genomes:
            idx = np.random.default_rng(cycle_num).choice(idx, size=n_genomes, replace=False)

        gests = np.array(pop.gestation_time)[idx]
        finite = gests[gests < 2147483647]
        max_gest = int(finite.max()) if finite.size > 0 else 100
        # Step budget with headroom so mutants that replicate a little slower than
        # their parent still finish within it (mirrors log_frequency_reports).
        max_steps = max_gest * self.cfg.steps_per_update * 2 + 1000

        genomes = jnp.array(np.array(pop.genome)[idx])
        genome_lens = jnp.array(np.array(pop.genome_len)[idx])
        return compute_mutational_robustness(
            genomes, genome_lens, max_steps, random.PRNGKey(cycle_num), self.cfg,
            n_mutants=n_mutants,
        )

    def run_simulation(self, key, total_cycles, log_interval=10000, use_wandb=False,
                       output_dir="output", toy_mode=False,
                       track_lineage=False, lineage_dir="lineage",
                       snapshot_interval=0, use_kernel=False, collect_interval=1):
        """Run the simulation for total_cycles.

        use_kernel selects the CUDA VM kernel (physax.vm_kernel.VMRunner) instead
        of the JAX interpreter for the slow-track execution. The kernel cannot run
        inside lax.scan/jit, so that path drives cycles from Python: the pre- and
        post-VM halves stay jitted, with the kernel launched between them.
        """
        print(f"=== JAX PHYSIS SIMULATION ===")
        print(f"Population capacity: {self.cfg.pop_size}, Initial: {self.cfg.initial_pop}")
        print(f"Steps per update: {self.cfg.steps_per_update}")
        print(f"Total cycles: {total_cycles}, Log interval: {log_interval}")
        print()
        
        os.makedirs(output_dir, exist_ok=True)

        if use_wandb:
            if not wandb_logger.init_wandb(self.cfg, total_cycles):
                use_wandb = False

        global_self_replicating_genomes.clear()
        global_fertile_genomes.clear()

        k1, k2 = random.split(key)
        pop = self.init_population(k1)
        db = init_genome_db(self.cfg)

        # cycle_idx starts at 1 so the first cycle's births get ids >= pop_size,
        # never colliding with the seed ids [0, pop_size) set in init_population.
        cycle_idx = jnp.int32(1)

        def scan_cycles(state, keys):
            def step(carry, k):
                p, d, cyc = carry
                new_p, new_d, stats = self.cycle_step(p, d, cyc, k)
                return (new_p, new_d, cyc + 1), stats
            return lax.scan(step, state, keys)

        jit_scan = jax.jit(scan_cycles)

        # Kernel path: the CUDA VM can't run inside jit/lax.scan, so drive cycles
        # from Python with the pre/post halves jitted and the kernel in between.
        # Produces the same per-chunk stacked stats as jit_scan.
        if use_kernel:
            from physax.vm_kernel import VMRunner
            vm_runner = VMRunner(self.cfg)
            pre_jit = jax.jit(self._pre_vm)
            # do_collect (arg index 5) is static so the genome-collection callbacks
            # compile out entirely on cycles where we skip them (no host transfer).
            post_jit = jax.jit(self._post_vm, static_argnums=(5,))

            def run_chunk(pop, db, cycle_idx, chunk_keys, base_cycle):
                stats_list = []
                for c in range(chunk_keys.shape[0]):
                    k = chunk_keys[c]
                    cyc_num = base_cycle + c + 1
                    pop, is_slow = pre_jit(pop, db)
                    pop = vm_runner.run(pop, is_slow)            # mutates pop in place
                    # do_collect=False: keep the genome-collection jax.debug.callbacks
                    # OUT of the jitted graph. They are an ordered effect whose mere
                    # presence forces a host-sync barrier every cycle (~90 ms at pop
                    # 16384), dominating runtime once the VM is on the kernel. We
                    # collect in plain Python below instead (no lax.scan here to stop
                    # us), which costs one host transfer per collect_interval cycles.
                    pop, db, stats = post_jit(pop, db, cycle_idx, k, is_slow, False)
                    cycle_idx = cycle_idx + 1
                    stats_list.append(stats)
                    # Archive the standing self-replicator / fertile genomes on a
                    # coarse cadence (dedup'd by hash in the collectors). Auxiliary
                    # data for end-of-run reports; does not affect pop/db.
                    if (cyc_num % collect_interval == 0) or (cyc_num == total_cycles):
                        collect_self_replicating(
                            pop.genome_hash, pop.genome,
                            pop.alive & (pop.status == SELF_REPLICATING))
                        collect_fertile(
                            pop.genome_hash, pop.genome,
                            pop.alive & (pop.status == FERTILE))
                stats = jax.tree.map(lambda *xs: jnp.stack(xs), *stats_list)
                return pop, db, cycle_idx, stats
            print(f"VM backend: CUDA kernel (physax.vm_kernel.VMRunner); "
                  f"genome collection every {collect_interval} cycles")
        else:
            print("VM backend: JAX interpreter")

        if track_lineage:
            os.makedirs(lineage_dir, exist_ok=True)

        n_chunks = total_cycles // log_interval
        all_stats = []
        cycle_keys = random.split(k2, total_cycles)

        # Set up logging for toy mode
        toy_log_file = None
        if toy_mode:
            log_path = os.path.join(output_dir, "toy_example_logs.txt")
            toy_log_file = open(log_path, "w")
            
            def format_genome(gen, length):
                genes = [int(x) for x in gen[:length]]
                names = [OP_NAMES.get(g, str(g)) for g in genes]
                return f"[{', '.join(names)}]"
            
            # Initial state
            init_ages = np.array(pop.age)
            init_gests = np.array(pop.gestation_time)
            init_statuses = np.array(pop.status)
            init_has_child = np.array(pop.has_child)
            init_genomes = np.array(pop.genome)
            init_genome_lens = np.array(pop.genome_len)
            init_alive = np.array(pop.alive)
            init_exec_insts = np.array(pop.executed_instructions)
            
            init_log = "Initial Population State:\n"
            for i in range(self.cfg.pop_size):
                if init_alive[i]:
                    genome_str = format_genome(init_genomes[i], init_genome_lens[i])
                    init_log += (
                        f"Initial State: Agent {i}, age: {init_ages[i]}, "
                        f"gest: {init_gests[i]}, "
                        f"exec_inst: {init_exec_insts[i]}, "
                        f"status: {init_statuses[i]}, "
                        f"has_child: {init_has_child[i]}, "
                        f"genome: {genome_str}\n"
                    )
            print(init_log, end="")
            toy_log_file.write(init_log)

        try:
            for chunk in trange(n_chunks, desc="Running"):
                start = chunk * log_interval
                end = (chunk + 1) * log_interval
                chunk_keys = cycle_keys[start:end]

                if use_kernel:
                    pop, db, cycle_idx, stats = run_chunk(pop, db, cycle_idx, chunk_keys, start)
                else:
                    (pop, db, cycle_idx), stats = jit_scan((pop, db, cycle_idx), chunk_keys)
                (pop, db) = jax.block_until_ready((pop, db))

                cycle_num = end
                pop_size = int(stats['pop_size'][-1])
                births = int(jnp.sum(stats['births']))
                q_len = stats['q_genome_len'][-1]

                # SS: distribution (not just the median) of each genome part over the living
                # population, so wandb shows spread -- a flat median can still hide a shifting
                # tail as variants sweep. GENOME_PART_PCTS picks the percentiles to log.
                def _alive_pcts(vals):
                    return jnp.nanpercentile(jnp.where(pop.alive, vals, jnp.nan), GENOME_PART_PCTS)

                # avg op-codes (micro-ops) per instruction, per organism
                ops_per_instr = jnp.sum(pop.instruction_lengths, axis=1) / jnp.maximum(pop.n_instructions, 1)
                q_registers = _alive_pcts(pop.n_ses)              # how many registers / state elements
                q_instructions = _alive_pcts(pop.n_instructions)  # how many instructions
                q_ops_per_instr = _alive_pcts(ops_per_instr)      # how many op-codes per instruction

                print(f"Cycle {cycle_num}: Pop={pop_size}, Births={births}, Percentiles={q_len}")

                if use_wandb:
                    wandb_logger.log_cycle_metrics(start, log_interval, stats)
                    # SS: percentile spread of the genome's structural parts, logged once per chunk.
                    wandb_logger.log_genome_part_distribution(
                        cycle_num, GENOME_PART_PCTS, q_registers, q_instructions, q_ops_per_instr
                    )
                    # SS: language properties -- the genotype (raw tokens) and the
                    # genotype->phenotype mapping (each genome's self-defined instruction
                    # set). Scalars every chunk; opcode/symbol bar charts on the heavier
                    # frequency-report cadence.
                    lang_stats = compute_language_stats(pop, self.cfg)
                    if lang_stats is not None:
                        log_hist = (cycle_num % 10000 == 0 or cycle_num == total_cycles)
                        wandb_logger.log_language_properties(cycle_num, lang_stats, log_hist=log_hist)

                hash_vals = pop.genome_hash

                snapshot = {
                    'cycle': cycle_num,
                    'alive': np.array(pop.alive),
                    'genome_len': np.array(pop.genome_len),
                    'gestation_time': np.array(pop.gestation_time),
                    'executed_instructions': np.array(pop.executed_instructions),
                    'age': np.array(pop.age),
                    'hash': np.array(hash_vals),
                    'status': np.array(pop.status),
                    'id': np.array(pop.id),
                    'parent_id': np.array(pop.parent_id),
                    'birth_cycle': np.array(pop.birth_cycle),
                    # Genomes of the current population, so a lineage's genotype can
                    # be diffed against an ancestor sampled at an earlier snapshot.
                    'genome': np.array(pop.genome),
                }

                # Persist the genealogy edge list for this chunk. Only the cells that
                # actually produced a child are kept, so the file is a compact list of
                # (birth cycle, child id, parent id) rows rather than the dense grid.
                if track_lineage:
                    child_ids = np.array(stats['child_id'])      # (log_interval, pop_size)
                    parent_ids = np.array(stats['parent_id'])
                    rows, cols = np.nonzero(child_ids >= 0)
                    born_cycle = (start + 1) + rows               # cycle_idx of each birth
                    np.savez_compressed(
                        f"{lineage_dir}/edges_{cycle_num}.npz",
                        birth_cycle=born_cycle.astype(np.int64),
                        child_id=child_ids[rows, cols].astype(np.int64),
                        parent_id=parent_ids[rows, cols].astype(np.int64),
                    )

                # Persist a population genome snapshot on a coarser cadence than
                # the edge files, so lineage ids can be mapped back to genotypes
                # post-hoc (analysis.index_snapshot_genomes / conservation_map).
                # The genome grid is large, so this is gated on snapshot_interval
                # (a multiple of log_interval) rather than saved every chunk.
                if (track_lineage and snapshot_interval > 0
                        and (cycle_num % snapshot_interval == 0 or cycle_num == total_cycles)):
                    np.savez_compressed(
                        f"{lineage_dir}/snapshot_{cycle_num}.npz",
                        cycle=np.int64(cycle_num),
                        id=snapshot['id'].astype(np.int64),
                        parent_id=snapshot['parent_id'].astype(np.int64),
                        birth_cycle=snapshot['birth_cycle'].astype(np.int64),
                        alive=snapshot['alive'],
                        status=snapshot['status'],
                        genome=snapshot['genome'],
                        genome_len=snapshot['genome_len'],
                        gestation_time=snapshot['gestation_time'],
                        hash=snapshot['hash'],
                        # per-position executed mask (pop_size, max_genome_len): lets
                        # analysis separate functional (executed) tape positions from
                        # junk, to test whether genetic diversity is concentrated in
                        # non-executed / non-functional positions.
                        executed=np.array(pop.executed),
                        executed_instructions=snapshot['executed_instructions'],
                    )

                chunk_rec = {
                    'cycle': cycle_num,
                    'pop_size': pop_size,
                    'births': births,
                    'q_len': q_len,
                    'snapshot': snapshot
                }
                
                if use_wandb:
                    wandb_logger.log_snapshot_and_diversity(cycle_num, snapshot, self.cfg)
                    
                    if cycle_num % 10000 == 0 or cycle_num == total_cycles:
                        wandb_logger.log_frequency_reports(
                            cycle_num, snapshot,
                            global_self_replicating_genomes,
                            global_fertile_genomes,
                            output_dir,
                            self.cfg
                        )

                        # SS: mutational robustness of the living self-replicators.
                        # Heavy (re-runs many mutants through the VM), so kept on this
                        # infrequent cadence alongside the frequency reports.
                        rob_stats = self._robustness_of_self_replicators(pop, cycle_num)
                        if rob_stats is not None:
                            wandb_logger.log_mutational_robustness(cycle_num, rob_stats)
                
                # Stack has_child to stats as well
                if 'has_child' in stats:
                    chunk_rec['has_child'] = stats['has_child']

                all_stats.append(chunk_rec)
                
                if toy_mode and toy_log_file is not None:
                    curr_ages = np.array(pop.age)
                    curr_gests = np.array(pop.gestation_time)
                    curr_statuses = np.array(pop.status)
                    # Use stats['has_child'] if available, else pop.has_child
                    if 'has_child' in stats:
                        curr_has_child = np.array(stats['has_child'][-1])
                    else:
                        curr_has_child = np.array(pop.has_child)
                    curr_genomes = np.array(pop.genome)
                    curr_genome_lens = np.array(pop.genome_len)
                    curr_alive = np.array(pop.alive)
                    curr_exec_insts = np.array(pop.executed_instructions)
                    
                    cycle_log = ""
                    for i in range(self.cfg.pop_size):
                        if curr_alive[i]:
                            genome_str = format_genome(curr_genomes[i], curr_genome_lens[i])
                            cycle_log += (
                                f"Cycle {cycle_num}, Agent {i}, age: {curr_ages[i]}, "
                                f"gest: {curr_gests[i]}, "
                                f"exec_inst: {curr_exec_insts[i]}, "
                                f"status: {curr_statuses[i]}, "
                                f"has_child: {curr_has_child[i]}, "
                                f"genome: {genome_str}\n"
                            )
                    print(cycle_log, end="")
                    toy_log_file.write(cycle_log)
                    
        except KeyboardInterrupt:
            print("Simulation interrupted by user.")
        except Exception as e:
            print(f"Simulation interrupted by error: {e}")
            import traceback
            traceback.print_exc()

        if use_wandb:
            wandb_logger.finish_wandb()
            
        if toy_log_file is not None:
            toy_log_file.close()
            print(f"Toy debug log written to {os.path.join(output_dir, 'toy_example_logs.txt')}")
            
        # Save all stats for later analysis
        stats_path = os.path.join(output_dir, "simulation_stats.pkl")
        with open(stats_path, "wb") as f:
            pickle.dump(all_stats, f)
        print(f"Saved simulation stats to {stats_path}")
            
        return pop, all_stats
