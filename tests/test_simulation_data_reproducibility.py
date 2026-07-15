import os
import shutil
import glob
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import pytest
from dotenv import load_dotenv
from pathlib import Path

from physax.sim.config import make_config
from physax.sim.model import Model, init_genome_db
import subprocess
import sys
import pickle
from physax.analysis.genome_evaluator import run_batch_until_division
from physax.sim.agent import Agent
from physax.sim.config import SELF_REPLICATING, FERTILE, UNCLASSIFIED


def test_simulation_data_reproducibility():
    load_dotenv()
    base_path = str(Path(os.getenv("BASE_PATH", "output")))
    test_dir = base_path + '/test_reproducibility_output'
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir, exist_ok=True)

    print(f"\nDevice: {jax.devices()[0].platform}")

    print("\n--- STEP 1: Running Simulation via __main__.py ---")
    
    cmd = [
        sys.executable, "-m", "physax.__main__",
        "--pop_size", "256",
        "--initial_pop", "256",
        "--total_cycles", "1000",
        "--log_interval", "1",
        "--track_lineage",
        "--snapshot_interval", "1",
        "--lineage_dir", test_dir,
        "--seed", "42",
        "--copy_mutation_rate", "0.05",
        "--divide_mutation_rate", "0.01",
        "--divide_insert_rate", "0.01",
        "--divide_delete_rate", "0.01"
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        pytest.fail("Simulation via __main__.py failed!")
        
    print("Simulation completed successfully.")

    print("\n--- STEP 2: Verifying Classifications & Gestation Times via VM ---")
    
    cfg = make_config(
        pop_size=256, 
        max_micro_ops=16,
        copy_mutation_rate=0.05,
        divide_mutation_rate=0.01,
        divide_insert_rate=0.01,
        divide_delete_rate=0.01
    )
    eval_key = random.PRNGKey(42)

    # First, collect all birth genomes so we test genomes from their actual starting state 
    # (this prevents false failures for genomes that self-mutate during gestation).
    edges_data = {}
    for edges_file in glob.glob(f"{test_dir}/edges_*.npz"):
        edges_loaded = np.load(edges_file)
        c_ids = edges_loaded['child_id']
        keys = edges_loaded['mut_key']
        if 'unmut_tape' in edges_loaded:
            unmut_tapes = edges_loaded['unmut_tape']
            unmut_lens = edges_loaded['unmut_len']
            for idx in range(len(c_ids)):
                edges_data[c_ids[idx]] = {
                    'key': keys[idx],
                    'unmut_tape': unmut_tapes[idx],
                    'unmut_len': unmut_lens[idx]
                }
        edges_loaded.close()
    
    unique_genomes = {}
    
    for i in range(1, 1001):
        snapshot_file = f"{test_dir}/snapshot_{i}.npz"
        if not os.path.exists(snapshot_file):
            continue
            
        loaded = np.load(snapshot_file)
        
        alive = loaded['alive']
        status = loaded['status']
        mask = alive & ((status == SELF_REPLICATING) | (status == FERTILE))
        indices = np.where(mask)[0]
        
        ids = loaded['id'][indices]
        genomes = loaded['genome'][indices]
        lens = loaded['genome_len'][indices]
        gests = loaded['gestation_time'][indices]
        
        for id_val, g, l, st, gest in zip(ids, genomes, lens, status[indices], gests):
            # If we have the exact birth sequence, use it! Otherwise fallback to the snapshot sequence.
            if id_val in edges_data:
                mut_key = jnp.array(edges_data[id_val]['key'], dtype=jnp.uint32)
                unmut_tape = jnp.array(edges_data[id_val]['unmut_tape'])
                unmut_len = jnp.int32(edges_data[id_val]['unmut_len'])
                reconstructed_tape, reconstructed_len = Agent.apply_divide_mutations(
                    mut_key, unmut_tape, unmut_len, jnp.int32(UNCLASSIFIED), cfg
                )
                orig_g = np.array(reconstructed_tape)
                orig_l = int(reconstructed_len)
            else:
                orig_g = g
                orig_l = l

            h_tup = tuple(orig_g[:orig_l])
            if h_tup not in unique_genomes:
                unique_genomes[h_tup] = {
                    'genome': orig_g,
                    'len': orig_l,
                    'status': st,
                    'gestation_time': gest
                }
        loaded.close()
        
    print(f"Collected {len(unique_genomes)} unique self-replicating/fertile genomes.")
    
    if len(unique_genomes) > 0:
        genomes_arr = jnp.array([v['genome'] for v in unique_genomes.values()])
        lens_arr = jnp.array([v['len'] for v in unique_genomes.values()])
        status_arr = np.array([v['status'] for v in unique_genomes.values()])
        gest_arr = np.array([v['gestation_time'] for v in unique_genomes.values()])
        
        n = len(genomes_arr)
        agents = jax.vmap(
            lambda g, l: Agent.init_organism(
                g, l, jnp.int32(-1), jnp.int32(UNCLASSIFIED), jnp.int32(-1), cfg
            )
        )(genomes_arr, lens_arr)

        max_steps = 2000 * cfg.steps_per_update
        
        print(f"Running batch execution up to {max_steps} steps...")
        keys = random.split(eval_key, n)
        
        # We need to compile run_batch_until_division with static max_steps
        @jax.jit
        def _run_batch(ag, k):
            return run_batch_until_division(ag, jnp.int32(max_steps), k, cfg)
            
        final_agents, final_steps, _, final_finished_steps = _run_batch(agents, keys)
        
        divides = final_agents.has_child
        matching = jnp.arange(cfg.max_genome_len)[None, :] < final_agents.child_len[:, None]
        is_exact = (
            jnp.all(jnp.where(matching, final_agents.genome == final_agents.child, True), axis=-1)
            & (final_agents.genome_len == final_agents.child_len)
        )
        self_replicates = divides & is_exact & ~final_agents.read_from_child
        
        divides = np.array(divides)
        self_replicates = np.array(self_replicates)
        final_finished_steps = np.array(final_finished_steps)
        
        sr_mask = (status_arr == SELF_REPLICATING)
        fertile_mask = (status_arr == FERTILE)
        
        errors = []

        print("Checking self-replicators...")
        if np.any(sr_mask):
            failed_sr = np.where(sr_mask & ~self_replicates)[0]
            if len(failed_sr) > 0:
                errors.append(f"{len(failed_sr)} SELF_REPLICATING genomes did not perfectly self-replicate!")
        
        print("Checking fertile genomes...")
        if np.any(fertile_mask):
            failed_f = np.where(fertile_mask & ~divides)[0]
            if len(failed_f) > 0:
                errors.append(f"{len(failed_f)} FERTILE genomes did not divide!")
            
        print("Checking gestation times...")
        # final_finished_steps is in instructions, gest_arr is in cycles. 
        # Convert instructions to cycles: ceil(instructions / steps_per_update)
        vm_cycles = np.ceil(final_finished_steps / cfg.steps_per_update).astype(np.int32)
        
        # -1 means it didn't finish, keep it as -1
        vm_cycles = np.where(final_finished_steps == -1, -1, vm_cycles)
        
        mismatched_gest = np.sum((vm_cycles != gest_arr) & (vm_cycles != -1))
        if mismatched_gest > 0:
            errors.append(f"Gestation time mismatch for {mismatched_gest} genomes (excluding those that didn't divide)!")
            for idx in np.where((vm_cycles != gest_arr) & (vm_cycles != -1))[0]:
                errors.append(f"  Genome idx {idx}: expected {gest_arr[idx]} cycles, got {vm_cycles[idx]} cycles ({final_finished_steps[idx]} steps)")

    print("\n--- STEP 3: Verifying Reproduction via Genealogical Tree ---")
    genomes_by_id = {}
    reproduction_events = []

    for i in range(1, 1001):
        snapshot_file = f"{test_dir}/snapshot_{i}.npz"
        if not os.path.exists(snapshot_file):
            continue

        loaded = np.load(snapshot_file)
        alive = loaded['alive']
        ids = loaded['id']
        birth_cycles = loaded['birth_cycle']
        genomes = loaded['genome']
        lens = loaded['genome_len']

        # Find new births this cycle
        for idx in np.where(alive & (birth_cycles == i))[0]:
            reproduction_events.append({
                'child_id': ids[idx],
                'cycle': i,
                'child_genome': genomes[idx],
                'child_len': lens[idx]
            })
        loaded.close()

    print(f"Found {len(reproduction_events)} reproduction events to verify.")

    edges_data = {}
    for edges_file in glob.glob(f"{test_dir}/edges_*.npz"):
        edges_loaded = np.load(edges_file)
        c_ids = edges_loaded['child_id']
        keys = edges_loaded['mut_key']
        unmut_tapes = edges_loaded['unmut_tape']
        unmut_lens = edges_loaded['unmut_len']
        for idx in range(len(c_ids)):
            edges_data[c_ids[idx]] = {
                'key': keys[idx],
                'unmut_tape': unmut_tapes[idx],
                'unmut_len': unmut_lens[idx]
            }
        edges_loaded.close()

    mismatch_count = 0
    accounted_mutations = 0

    for ev in reproduction_events:
        cid = ev['child_id']
        c_gen = ev['child_genome']
        c_len = ev['child_len']

        if cid in edges_data:
            ed = edges_data[cid]
            unmut_tape = ed['unmut_tape']
            unmut_len = ed['unmut_len']
            mut_key = jnp.array(ed['key'], dtype=jnp.uint32)

            if unmut_len != c_len or not np.array_equal(unmut_tape[:unmut_len], c_gen[:c_len]):
                mismatch_count += 1
                
                # Re-run the exact JAX mutation function with the exact same key to reconstruct it
                reconstructed_tape, reconstructed_len = Agent.apply_divide_mutations(
                    mut_key, jnp.array(unmut_tape), jnp.int32(unmut_len), jnp.int32(UNCLASSIFIED), cfg
                )
                
                # Check if the mutated reconstructed child perfectly matches the simulation child
                if reconstructed_len != c_len or not np.array_equal(np.array(reconstructed_tape)[:reconstructed_len], c_gen[:c_len]):
                    errors.append(f"Mutation reproduction failed for child {cid}! The reconstructed mutated genome did not match the simulation.")
                else:
                    accounted_mutations += 1
        else:
            errors.append(f"Missing mutation key record for child {cid}")

    print(f"Total reproduction genome mismatches due to mutations: {mismatch_count} out of {len(reproduction_events)} events.")
    print(f"Successfully accounted for {accounted_mutations} mutations using deterministic mutation reconstruction!")
    if accounted_mutations != mismatch_count:
        errors.append(f"Failed to account for {mismatch_count - accounted_mutations} mutated genomes!")

    print("\n--- STEP 4: Verifying Other Output Files ---")
    output_dirs = sorted(glob.glob(f"{base_path}/run_1000_cycles_seed_42_*"), key=os.path.getmtime)
    if not output_dirs:
        errors.append("No output directory found for the run!")
        out_dir = test_dir # fallback to prevent crashes
    else:
        out_dir = output_dirs[-1]
    
    stats_file = f"{out_dir}/simulation_stats.pkl"
    if not os.path.exists(stats_file):
        errors.append(f"simulation_stats.pkl is missing in {out_dir}!")
    else:
        with open(stats_file, 'rb') as f:
            stats = pickle.load(f)
        if len(stats) != 1000:
            errors.append(f"simulation_stats.pkl contains {len(stats)} chunks, expected 1000")
            
    snapshot_sr_genomes = {}
    snapshot_f_genomes = {}
    
    for i in range(1, 1001):
        snapshot_file = f"{test_dir}/snapshot_{i}.npz"
        if not os.path.exists(snapshot_file):
            continue
            
        loaded = np.load(snapshot_file)
        alive = loaded['alive']
        status = loaded['status']
        mask_sr = alive & (status == SELF_REPLICATING)
        mask_f = alive & (status == FERTILE)
        
        hashes_sr = loaded['hash'][mask_sr]
        genomes_sr = loaded['genome'][mask_sr]
        lens_sr = loaded['genome_len'][mask_sr]
        
        for h, g, l in zip(hashes_sr, genomes_sr, lens_sr):
            h_key = f"{h[0]}_{h[1]}"
            if h_key not in snapshot_sr_genomes:
                snapshot_sr_genomes[h_key] = g[:l]
                
        hashes_f = loaded['hash'][mask_f]
        genomes_f = loaded['genome'][mask_f]
        lens_f = loaded['genome_len'][mask_f]
        
        for h, g, l in zip(hashes_f, genomes_f, lens_f):
            h_key = f"{h[0]}_{h[1]}"
            if h_key not in snapshot_f_genomes:
                snapshot_f_genomes[h_key] = g[:l]
        loaded.close()
        
    sr_file = f"{out_dir}/self_replicating_genomes_details.npz"
    if os.path.exists(sr_file):
        sr_loaded = np.load(sr_file)
        for k in sr_loaded.files:
            g = sr_loaded[k]
            if k not in snapshot_sr_genomes:
                errors.append(f"Self-replicating genome {k} found in details but not in snapshots!")
            else:
                snap_len = len(snapshot_sr_genomes[k])
                if not np.array_equal(g[:snap_len], snapshot_sr_genomes[k]):
                    errors.append(f"Self-replicating genome {k} in details does not match snapshot sequence!")
        
        for k in snapshot_sr_genomes:
            if k not in sr_loaded.files:
                errors.append(f"Self-replicating genome {k} found in snapshots but missing from details!")
        sr_loaded.close()
    elif len(snapshot_sr_genomes) > 0:
        errors.append(f"self_replicating_genomes_details.npz is missing but {len(snapshot_sr_genomes)} were found in snapshots!")
        
    f_file = f"{out_dir}/fertile_genomes_details.npz"
    if os.path.exists(f_file):
        f_loaded = np.load(f_file)
        for k in f_loaded.files:
            g = f_loaded[k]
            if k not in snapshot_f_genomes:
                errors.append(f"Fertile genome {k} found in details but not in snapshots!")
            else:
                snap_len = len(snapshot_f_genomes[k])
                if not np.array_equal(g[:snap_len], snapshot_f_genomes[k]):
                    errors.append(f"Fertile genome {k} in details does not match snapshot sequence!")
                
        for k in snapshot_f_genomes:
            if k not in f_loaded.files:
                errors.append(f"Fertile genome {k} found in snapshots but missing from details!")
        f_loaded.close()
    elif len(snapshot_f_genomes) > 0:
        errors.append(f"fertile_genomes_details.npz is missing but {len(snapshot_f_genomes)} were found in snapshots!")

    print("\n--- STEP 5: Verifying Self-Modification Hashes ---")
    # Verify that all living genomes have correctly updated hashes, proving
    # that self-modifying genomes get a new hash and category.
    hash_errors = 0
    for i in range(1, 1001, 100): # Check a subset of snapshots to save time
        snapshot_file = f"{test_dir}/snapshot_{i}.npz"
        if not os.path.exists(snapshot_file):
            continue
            
        loaded = np.load(snapshot_file)
        alive = loaded['alive']
        if not np.any(alive):
            loaded.close()
            continue
            
        hashes = loaded['hash'][alive]
        genomes = loaded['genome'][alive]
        lens = loaded['genome_len'][alive]
        loaded.close()
        
        # Batch compute hashes for all alive genomes in this snapshot
        computed_hashes = jax.vmap(Agent._hash_genome, in_axes=(0, 0, None))(jnp.array(genomes), jnp.array(lens), cfg)
        computed_hashes = np.array(computed_hashes)
        
        mismatches = np.sum((computed_hashes[:, 0] != hashes[:, 0]) | (computed_hashes[:, 1] != hashes[:, 1]))
        if mismatches > 0:
            hash_errors += mismatches
            errors.append(f"Snapshot {i}: {mismatches} alive genomes have a hash that does not match their current sequence!")
            
    if hash_errors == 0:
        print("Success! All genome hashes perfectly match their sequences, proving self-modifications are tracked correctly.")

    # shutil.rmtree(test_dir)
    # if 'out_dir' in locals() and out_dir != test_dir and os.path.exists(out_dir):
    #     shutil.rmtree(out_dir)
    
    if errors:
        print("\n" + "="*40)
        print("TEST FAILURES SUMMARY")
        print("="*40)
        for e in errors:
            print(e)
        pytest.fail(f"Test failed with {len(errors)} errors. See stdout for details.")
    
    print("\nSUCCESS: All tests passed and loaded genomes perfectly match VM execution!")
