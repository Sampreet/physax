import numpy as np
import sys
import argparse
import pickle
from pathlib import Path

# Resolve status constants directly to decouple from external dependencies if possible,
# or we can import them from physis.sim.config. Let's do both safely.
try:
    from physis.sim.config import (
        UNCLASSIFIED, SELF_REPLICATING, FERTILE, NON_FERTILE, NON_STANDARD
    )
except ImportError:
    # Fallback status definitions in case import fails
    UNCLASSIFIED = 0
    SELF_REPLICATING = 1
    FERTILE = 2
    NON_FERTILE = 3
    NON_STANDARD = 4

STATUS_NAME_TO_VAL = {
    "UNCLASSIFIED": UNCLASSIFIED,
    "SELF_REPLICATING": SELF_REPLICATING,
    "FERTILE": FERTILE,
    "NON_FERTILE": NON_FERTILE,
    "NON_STANDARD": NON_STANDARD
}
STATUS_VAL_TO_NAME = {v: k for k, v in STATUS_NAME_TO_VAL.items()}

def main():
    parser = argparse.ArgumentParser(
        description="Inspect simulation stats at a given cycle and list genomes of a given status sorted by gestation time."
    )
    parser.add_argument("--cycle", type=int, required=True, help="The simulation cycle to inspect")
    parser.add_argument("--status", type=str, default="SELF_REPLICATING", 
                        help="The genome status to filter by (integer value or name: SELF_REPLICATING, FERTILE, etc.)")
    parser.add_argument("--folder", type=str, default=None, help="Folder name of the simulation run inside the base path")
    parser.add_argument("--base_path", type=str, default="output", help="Base path of the simulation runs")
    parser.add_argument("--top_n", type=int, default=None, help="Only show the top N genomes with the shortest gestation times")
    
    args = parser.parse_args()

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

    # Determine simulation run folder
    if args.folder:
        folder_path = base_path / args.folder
    else:
        runs = [p for p in base_path.glob("run_*") if p.is_dir()]
        if not runs:
            print(f"Error: No run folders found in {base_path}")
            sys.exit(1)
            
        folder_path = max(runs, key=lambda p: p.stat().st_mtime)
        print(f"Auto-selected most recent run: {folder_path}")

    stats_file = folder_path / "simulation_stats.pkl"
    if not stats_file.exists():
        print(f"Error: Could not find simulation_stats.pkl at {stats_file}")
        sys.exit(1)

    print(f"Loading simulation stats from {stats_file}...")
    with open(stats_file, "rb") as f:
        stats = pickle.load(f)

    # Parse status parameter
    try:
        # Check if it is an integer
        status_val = int(args.status)
        if status_val not in STATUS_VAL_TO_NAME:
            print(f"Error: Invalid status integer {status_val}. Must be one of {list(STATUS_VAL_TO_NAME.keys())}")
            sys.exit(1)
    except ValueError:
        # Check if it matches a string key (case-insensitive)
        status_name = args.status.upper().strip()
        if status_name not in STATUS_NAME_TO_VAL:
            print(f"Error: Invalid status name '{args.status}'. Must be one of {list(STATUS_NAME_TO_VAL.keys())}")
            sys.exit(1)
        status_val = STATUS_NAME_TO_VAL[status_name]

    # Find the chunk matching args.cycle
    chunk = None
    available_cycles = [c["cycle"] for c in stats]
    for c in stats:
        if c["cycle"] == args.cycle:
            chunk = c
            break
            
    if chunk is None:
        if not available_cycles:
            print("Error: No cycle data found in simulation_stats.pkl")
            sys.exit(1)
        # Find closest cycle
        closest_cycle = min(available_cycles, key=lambda x: abs(x - args.cycle))
        print(f"Warning: Cycle {args.cycle} not found. Using the closest recorded cycle: {closest_cycle}")
        for c in stats:
            if c["cycle"] == closest_cycle:
                chunk = c
                break
    else:
        print(f"Found exact stats for cycle {args.cycle}")

    snap = chunk["snapshot"]
    
    # Extract arrays
    alive = np.array(snap["alive"])
    status = np.array(snap["status"])
    hashes = np.array(snap["hash"])
    gest = np.array(snap["gestation_time"])

    # Reconstruct 64-bit combined hash if it is stored in 2D array
    if hashes.ndim == 2:
        hashes_64 = (hashes[:, 0].astype(np.int64) << 32) | (hashes[:, 1].astype(np.uint32).astype(np.int64))
    else:
        hashes_64 = hashes.astype(np.int64)

    # Filter agents of given status who are alive
    mask = alive & (status == status_val)
    valid_hashes = hashes_64[mask]
    valid_gest = gest[mask]

    # Group by unique hash and find the minimum gestation time
    hash_to_min_gest = {}
    for h, g in zip(valid_hashes, valid_gest):
        # Filter out dummy/uninitialized gestation times (e.g. >= 2000000000 or <= 0)
        if 0 < g < 2000000000:
            if h not in hash_to_min_gest:
                hash_to_min_gest[h] = g
            else:
                hash_to_min_gest[h] = min(hash_to_min_gest[h], g)

    # Sort hashes by gestation time (short to long)
    sorted_hashes = sorted(hash_to_min_gest.items(), key=lambda x: x[1])

    # Print summary
    print(f"\nRun Folder: {folder_path}")
    print(f"Genome Status: {STATUS_VAL_TO_NAME[status_val]} (value {status_val})")
    print(f"Total active agents with this status: {np.sum(mask)}")
    print(f"Unique genome hashes with recorded gestation: {len(sorted_hashes)}")
    
    if len(sorted_hashes) == 0:
        print("\nNo genomes found with the specified status and a valid gestation time at this cycle.")
        return

    # Limit to top_n if specified
    hashes_to_show = sorted_hashes
    if args.top_n is not None and args.top_n > 0:
        hashes_to_show = sorted_hashes[:args.top_n]
        title_suffix = f" (showing top {args.top_n} of {len(sorted_hashes)})"
    else:
        title_suffix = ""

    # Print table
    print(f"\nList of hashes sorted from short to long gestation time{title_suffix}:")
    print(f"{'No.':<4} | {'Combined Hash':<22} | {'Decomposed (h0_h1)':<24} | {'Gestation Time':<14}")
    print("-" * 75)
    for idx, (h, g) in enumerate(hashes_to_show, 1):
        h0 = h >> 32
        h1 = h & 0xffffffff
        decomp_str = f"{h0}_{h1}"
        print(f"{idx:<4} | {h:<22} | {decomp_str:<24} | {g:<14}")

    # Print helper decoding command for the user
    if sorted_hashes:
        best_hash = sorted_hashes[0][0]
        folder_arg = f"--folder {args.folder}" if args.folder else ""
        print(f"\nTo decode the genome with the shortest gestation time ({sorted_hashes[0][1]} cycles), run:")
        print(f"  python decode_genome_illustration.py --hash {best_hash} {folder_arg}")

if __name__ == "__main__":
    main()
