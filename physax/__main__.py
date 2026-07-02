import jax
import jax.numpy as jnp
from jax import random
from physax.config import make_config, PERCENTILES, UNCLASSIFIED
from physax.model import Model
from physax.agent import Agent
from physax.visualization import generate_all_visualizations
import argparse
from pathlib import Path
from datetime import datetime
import os
from dotenv import load_dotenv
import numpy as np

print('device:', jax.devices()[0].platform)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pop_size', type=int, default=4096 * 40)
    parser.add_argument('--initial_pop', type=int, default=1000)
    parser.add_argument('--total_cycles', type=int, default=50_000)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--toy', action='store_true', help='Run a very small toy scenario for debugging')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for the simulation')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--no-caching', dest='caching', action='store_false', help='Keep all genomes on the slow track all the time')
    parser.add_argument('--no-compile-cache', dest='compile_cache', action='store_false', help='Disable the JAX persistent compilation cache')
    args = parser.parse_args()

    # SS: persistent compilation cache so reruns skip recompilation; --no-compile-cache turns it off
    if args.compile_cache:
        jax.config.update("jax_compilation_cache_dir", "/home/eleni/.cache/jax")
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    if args.toy:
        print("--- RUNNING IN TOY/DEBUG MODE ---")
        cfg = make_config(
            pop_size=32,
            initial_pop=1,
            max_genome_len=128,
            seed=args.seed,
            caching=args.caching
        )
        args.total_cycles = 500
        log_interval = 1
    else:
        cfg = make_config(
            pop_size=args.pop_size,
            initial_pop=args.initial_pop,
            seed=args.seed,
            caching=args.caching
        )
        log_interval = args.log_interval



    # Read base path from .env if it exists
    load_dotenv()
    base_path = Path(os.getenv("BASE_PATH", "output"))

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    path = base_path / f"run_{args.total_cycles}_cycles_seed_{args.seed}_{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving results to: {path}")

    model = Model(cfg)
    key = random.PRNGKey(args.seed)
    try:
        pop, stats = model.run_simulation(
            key,
            total_cycles=args.total_cycles,
            log_interval=log_interval,
            use_wandb=args.wandb,
            output_dir=str(path),
            toy_mode=args.toy
        )
    except Exception as e:
        print(f"Unhandled exception in main: {e}")
        pass

    print("\n=== FINAL STATE ===")
    alive = pop.alive
    alive_count = jnp.sum(alive)
    q_lens = jnp.nanpercentile(jnp.where(alive, pop.genome_len, jnp.nan), PERCENTILES )

    print(f"Alive: {int(alive_count)}")
    print(f"min={q_lens[0]:.0f}, lq={q_lens[1]:.1f}, med={q_lens[2]:.0f}, uq={q_lens[3]:.0f}, max={q_lens[4]:.0f}")

    if len(stats) > 0:
        top_hashes = generate_all_visualizations(stats, path, cfg)

        from physax.model import global_self_replicating_genomes, global_fertile_genomes

        all_genomes_dict = {f"{h[0]}_{h[1]}" if isinstance(h, tuple) else str(h): g for h, g in global_self_replicating_genomes.items()}
        if all_genomes_dict:
            np.savez(str(path / "self_replicating_genomes_details.npz"), **all_genomes_dict)
            print(f"Saved {len(all_genomes_dict)} self-replicating genomes to genomes_details.npz")

        fertile_genomes_dict = {f"{h[0]}_{h[1]}" if isinstance(h, tuple) else str(h): g for h, g in global_fertile_genomes.items()}
        if fertile_genomes_dict:
            np.savez(str(path / "fertile_genomes_details.npz"), **fertile_genomes_dict)
            print(f"Saved {len(fertile_genomes_dict)} fertile genomes to fertile_genomes_details.npz")
    else:
        print("No stats collected, skipping plots and saves.")

    if args.toy:
        print("\n--- TOY DEBUG LOG ---")
        for chunk in stats:
            cyc = chunk['cycle']
            pop_cnt = chunk['pop_size']
            b = chunk['births']
            status_arr = chunk['snapshot']['status']
            alive_arr = chunk['snapshot']['alive']

            # Count statuses among alive
            status_counts = {}
            for st in [0, 1, 2, 3, 4]:
                status_counts[st] = int(jnp.sum((status_arr == st) & alive_arr))

            print(f"Cycle {cyc}: Pop={pop_cnt}, Births={b}, Status Breakdown (UNCLASSIFIED, SELF_REPLICATING, FERTILE, NON_FERTILE, NON_STANDARD)={status_counts}")
