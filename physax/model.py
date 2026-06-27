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
from physax.genome_analysis import classify_genome, get_execution_route, compute_cycle_stats
from typing import NamedTuple

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
        return pop

    def _place_children_on_grid(self, pop: Agent, has_child: jnp.ndarray, child_states: Agent, key) -> Agent:
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

        # Place children using gather approach (avoids scatter conflicts with duplicates)
        # For each cell j, check if any birthing parent targets it
        all_cells = jnp.arange(self.cfg.pop_size)
        # targets_match[j, k] = True if parent k targets cell j AND has_child[k]
        targets_match = (target_indices[None, :] == all_cells[:, None]) & has_child[None, :]
        # any_child_placed[j] = True if any parent places a child at cell j
        any_child_placed = jnp.any(targets_match, axis=1)
        # source_parent[j] = index of (first) parent placing a child at cell j
        source_parent = jnp.argmax(targets_match.astype(jnp.int32), axis=1)

        # Gather child states from source parents
        gathered_children = jax.tree.map(lambda x: x[source_parent], child_states)

        # Conditionally replace: cell j gets child state if a child was placed, else keeps current
        new_pop = jax.tree.map(
            lambda c, p: jnp.where(
                any_child_placed.reshape((-1,) + (1,) * (c.ndim - 1)),
                c, p
            ),
            gathered_children, pop
        )
        return new_pop, any_child_placed

    def cycle_step(self, pop: Agent, db: GenomeDB, key) -> tuple[Agent, GenomeDB, dict]:
        """Execute one cycle: step all organisms, handle births."""

        k_exec, k_birth, k_place = random.split(key, 3)

        # 0. Check DB for unclassified/new agents
        needs_lookup = pop.alive & (pop.status == UNCLASSIFIED)
        found, db_idx = lookup_db(pop.genome_hash, db)
        
        apply_cache = needs_lookup & found
        pop = pop._replace(
            status=jnp.where(apply_cache, db.statuses[db_idx], pop.status),
            gestation_time=jnp.where(apply_cache, db.gestations[db_idx], pop.gestation_time)
        )

        # Determine execution routes based on genome classification
        routes = get_execution_route(pop.status)
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
        
        # Slow Track: UNCLASSIFIED, FERTILE, NON_STANDARD
        exec_keys = random.split(k_exec, self.cfg.pop_size)
        
        # Process entire population natively without max batch limits.
        # We run VM on all agents but only apply updates to `is_slow` ones.
        p_exec = jax.vmap(self.vm.update)(pop, exec_keys)
        
        pop = jax.tree.map(
            lambda new, old: jnp.where(is_slow.reshape((-1,) + (1,) * (new.ndim - 1)), new, old),
            p_exec, pop
        )
        
        # 2. Aging Phase for slow track
        pop = pop._replace(age=jnp.where(pop.alive & is_slow, pop.age + 1, pop.age))

        # 3. Classification Phase
        new_status, new_gestation = classify_genome(pop, self.cfg)
        
        newly_self_rep = pop.alive & (new_status == SELF_REPLICATING) & (pop.status != SELF_REPLICATING)
        jax.debug.callback(collect_self_replicating, pop.genome_hash, pop.genome, newly_self_rep)
        
        newly_fertile = pop.alive & (new_status == FERTILE) & (pop.status != FERTILE)
        jax.debug.callback(collect_fertile, pop.genome_hash, pop.genome, newly_fertile)
        
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
        pop_new, overwritten_mask = self._place_children_on_grid(pop, has_child, child_states, k_place)
        
        pop = pop_new

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
        return pop, db, stats

    def run_simulation(self, key, total_cycles, log_interval=10000, use_wandb=False, output_dir="output", toy_mode=False):
        """Run the simulation for total_cycles."""
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

        def scan_cycles(state, keys):
            def step(carry, k):
                p, d = carry
                new_p, new_d, stats = self.cycle_step(p, d, k)
                return (new_p, new_d), stats
            return lax.scan(step, state, keys)

        jit_scan = jax.jit(scan_cycles)

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

                (pop, db), stats = jit_scan((pop, db), chunk_keys)
                (pop, db) = jax.block_until_ready((pop, db))

                cycle_num = end
                pop_size = int(stats['pop_size'][-1])
                births = int(jnp.sum(stats['births']))
                q_len = stats['q_genome_len'][-1]

                print(f"Cycle {cycle_num}: Pop={pop_size}, Births={births}, Percentiles={q_len}")

                if use_wandb:
                    wandb_logger.log_cycle_metrics(start, log_interval, stats)

                hash_vals = pop.genome_hash

                snapshot = {
                    'cycle': cycle_num,
                    'alive': np.array(pop.alive),
                    'genome_len': np.array(pop.genome_len),
                    'gestation_time': np.array(pop.gestation_time),
                    'executed_instructions': np.array(pop.executed_instructions),
                    'age': np.array(pop.age),
                    'hash': np.array(hash_vals),
                    'status': np.array(pop.status)
                }

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
