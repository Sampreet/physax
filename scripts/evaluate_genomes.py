import argparse
import pickle
import numpy as np
import json
import jax
import jax.numpy as jnp
from physis.sim.config import make_config, UNCLASSIFIED, SELF_REPLICATING, FERTILE
from physis.sim.agent import Agent
from physis.analysis.virtual_machine import VirtualMachine
from physis.analysis.genome_evaluator import run_batch_until_division
from decode_genome_illustration import decode_genome
import os
import random

# Force CPU to avoid CUDA mismatch warnings or OOM if not needed
os.environ["JAX_PLATFORMS"] = "cpu"

def get_hash_str(h):
    if h.ndim == 1:
        return f"{h[0]}_{h[1]}"
    return f"{h >> 32}_{h & 0xffffffff}"



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True, help="Path to the simulation results folder or just the folder name if using .env")
    parser.add_argument("--base_path", type=str, default="output", help="Base path of the simulation runs")
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--num-genomes", type=int, required=True)
    args = parser.parse_args()
    
    from pathlib import Path
    # Read base path from .env if it exists
    base_path = Path(args.base_path)
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                if line.startswith("BASE_PATH="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    base_path = Path(val)
                    break
    
    folder = Path(args.folder)
    if not folder.is_absolute():
        folder = base_path / folder
        
    folder_str = str(folder)
    
    print(f"Loading simulation_stats.pkl from {folder_str}...")
    with open(os.path.join(folder_str, "simulation_stats.pkl"), "rb") as f:
        stats = pickle.load(f)
        
    chunk = next((c for c in stats if c["cycle"] == args.cycle), None)
    if chunk is None:
        print(f"Cycle {args.cycle} not found in stats.")
        return
        
    snap = chunk["snapshot"]
    alive = np.array(snap["alive"])
    hashes = np.array(snap["hash"])
    gest = np.array(snap["gestation_time"])
    status = np.array(snap["status"])
    genome_lens = np.array(snap["genome_len"])
    
    # Filter for alive self-replicators and fertile
    self_rep_mask = alive & (status == SELF_REPLICATING)
    fertile_mask = alive & (status == FERTILE)
    
    self_rep_indices = np.where(self_rep_mask)[0]
    fertile_indices = np.where(fertile_mask)[0]
    
    # Sample indices
    random.seed(42)
    selected_self_rep = random.sample(list(self_rep_indices), min(args.num_genomes, len(self_rep_indices)))
    selected_fertile = random.sample(list(fertile_indices), min(args.num_genomes, len(fertile_indices)))
    
    # Combine selected
    selected_indices = selected_self_rep + selected_fertile
    
    # We need the full genome arrays from the history npz files because the snapshot doesn't store the genomes.
    print(f"Loading genome details from npz in {folder_str}...")
    self_rep_npz = np.load(os.path.join(folder_str, "self_replicating_genomes_details.npz"))
    fertile_npz = np.load(os.path.join(folder_str, "fertile_genomes_details.npz"))
    
    genomes = []
    gestations = []
    hashes_str = []
    lens = []
    categories = []
    
    for idx in selected_indices:
        h = hashes[idx]
        h_str = get_hash_str(h)
        h_int = (int(h[0]) << 32) | (int(h[1]) & 0xffffffff) if h.ndim > 0 and len(h) > 1 else int(h)
        h_int_str = str(h_int)
        
        arr = None
        if h_str in self_rep_npz:
            arr = self_rep_npz[h_str]
        elif h_str in fertile_npz:
            arr = fertile_npz[h_str]
            
        if arr is None:
            # Fallback
            continue
            
        genomes.append(arr)
        gestations.append(gest[idx])
        hashes_str.append(h_str)
        lens.append(genome_lens[idx])
        categories.append("Self-Replicating" if idx in selected_self_rep else "Fertile")
        
    if not genomes:
        print("No valid genomes found in NPZ.")
        return
        
    cfg = make_config(pop_size=len(genomes), initial_pop=len(genomes), max_genome_len=512)
    
    agents_list = []
    for i, arr in enumerate(genomes):
        p_len = lens[i] # Use EXACT length from snapshot!
        p_genome = np.full(cfg.max_genome_len, -1, dtype=np.int32)
        p_genome[:min(p_len, cfg.max_genome_len)] = arr[:min(p_len, cfg.max_genome_len)]
        
        p_agent = Agent.init_organism(jnp.array(p_genome, dtype=jnp.int32), jnp.int32(p_len), jnp.int32(-1), jnp.int32(UNCLASSIFIED), jnp.int32(-1), cfg)
        agents_list.append(p_agent)
        
    import jax.tree_util
    agents_batch = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *agents_list)
    
    max_gestation_cycles = max([g for g in gestations if g > 0] + [100])
    # Multiply by steps_per_update (34) to get max_steps
    global_max_steps = (max_gestation_cycles * cfg.steps_per_update) + 1000
    
    print(f"Running vmapped simulation for max_steps={global_max_steps}...", flush=True)
    keys_batch = jax.random.split(jax.random.PRNGKey(42), len(genomes))
    final_agents, final_steps, _, final_finished_steps = run_batch_until_division(agents_batch, jnp.int32(global_max_steps), keys_batch, cfg)
    print(f"Simulation finished at step {final_steps}.")
    
    final_agents_list = [jax.tree_util.tree_map(lambda x: x[i], final_agents) for i in range(len(genomes))]
    final_finished_steps_list = [int(final_finished_steps[i]) for i in range(len(genomes))]
    
    md = [f"# Decoded Genomes Analysis (Cycle {args.cycle})"]
    md.append(f"\nEvaluating {args.num_genomes} self-replicating and {args.num_genomes} fertile genomes.")
    
    def format_decoded(decoded):
        lines = []
        line = []
        for item in decoded:
            if item == "I" or item == "SEP":
                if line:
                    lines.append("  " + ", ".join(line))
                line = [item]
            elif item == "BLANK":
                if line and "BLANK... (padded to end)" not in line:
                    if line:
                        lines.append("  " + ", ".join(line))
                    line = ["BLANK... (padded to end)"]
            else:
                line.append(str(item))
                if len(line) > 15:
                    lines.append("  " + ", ".join(line))
                    line = []
        if line:
            lines.append("  " + ", ".join(line))
        return "\n".join(lines)
        
    for cat_name in ["Self-Replicating", "Fertile"]:
        md.append(f"\n## {cat_name} Genomes")
        idx = 1
        for i, c in enumerate(categories):
            if c != cat_name:
                continue
                
            a = final_agents_list[i]
            p_genome = np.full(cfg.max_genome_len, -1, dtype=np.int32)
            p_len = lens[i]
            p_genome[:min(p_len, cfg.max_genome_len)] = genomes[i][:min(p_len, cfg.max_genome_len)]
            p_agent = Agent.init_organism(jnp.array(p_genome, dtype=jnp.int32), jnp.int32(p_len), jnp.int32(-1), jnp.int32(UNCLASSIFIED), jnp.int32(-1), cfg)
            decoded_parent = format_decoded(decode_genome(p_agent.genome, p_agent.separator_pos))
            
            db_gest = int(gestations[i])
            recalc_steps = final_finished_steps_list[i]
            recalc_cycles = (recalc_steps + cfg.steps_per_update - 1) // cfg.steps_per_update if recalc_steps > 0 else 0
            
            md.append(f"### Genome {idx}: {hashes_str[i]}")
            md.append(f"- **DB Gestation Time**: {db_gest} cycles")
            if recalc_steps != -1:
                md.append(f"- **Recalculated Gestation Time**: {recalc_steps} steps (~{recalc_cycles} cycles)")
            else:
                md.append(f"- **Recalculated Gestation Time**: DNF (Did not finish within {global_max_steps} steps)")
                
            md.append("\n#### Parent Genome\n```\n" + decoded_parent + "\n```\n")
            
            if bool(a.has_child):
                child_len = int(a.child_len)
                child_agent = Agent.init_organism(
                    jnp.array(a.child, dtype=jnp.int32),
                    jnp.int32(child_len), jnp.int32(-1), jnp.int32(UNCLASSIFIED), jnp.int32(-1), cfg
                )
                decoded_child = format_decoded(decode_genome(child_agent.genome, child_agent.separator_pos))
                md.append("#### Child Genome\n```\n" + decoded_child + "\n```\n")
            else:
                md.append("#### Child Genome\n*(No child produced)*\n")
            md.append("---\n")
            idx += 1
            
    report_file = os.path.join(folder_str, f"report_cycle_{args.cycle}.md")
    with open(report_file, "w") as f:
        f.write("\n".join(md))
        
    print(f"Generated {report_file}")

if __name__ == "__main__":
    main()
