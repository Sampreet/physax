import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import sys
import re
import argparse
from pathlib import Path

class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.__stdout__
        self.log = open(filename, "w", encoding="utf-8")
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(self.ansi_escape.sub('', message))
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

from physax.sim.config import make_config, BLANK, OP_NAMES, N_OPERANDS, UNCLASSIFIED
from physax.sim.agent import Agent
from physax.analysis.virtual_machine import VirtualMachine

def decode_compound(ops_arr):
    i = 0
    decoded = []
    ops = [int(x) for x in ops_arr]
    
    while i < len(ops):
        val = ops[i]
        opcode = abs(val) % 44
        n_args = int(N_OPERANDS[opcode])
        
        args = []
        for j in range(n_args):
            if i + 1 + j < len(ops):
                args.append(str(ops[i + 1 + j]))
            else:
                break
                
        op_str = OP_NAMES.get(opcode, str(val))
        if args:
            op_str += f" {' '.join(args)}"
            
        decoded.append(op_str)
        i += 1 + len(args)
        
    return decoded

def decode_genome(genome_arr, sep_pos):
    i = 0
    decoded = []
    genome = [int(x) for x in genome_arr]
    sep_pos = int(sep_pos)
    
    in_instruction = False
    while i < len(genome):
        val = genome[i]
        
        if val == BLANK:
            decoded.append("BLANK")
            i += 1
            continue
            
        if i > sep_pos:
            decoded.append(str(val))
            i += 1
            continue
            
        if val == 36: # SEP
            decoded.append(OP_NAMES.get(36))
            in_instruction = False
            i += 1
            continue
            
        if val == 34: # I
            decoded.append(OP_NAMES.get(34))
            in_instruction = True
            i += 1
            continue
            
        if val in [31, 32, 33, 35] and not in_instruction:
            # Structural markers R, S, Q, B
            decoded.append(OP_NAMES.get(val))
            i += 1
            continue
            
        # We are processing an opcode
        opcode = abs(val) % 44
        n_args = int(N_OPERANDS[opcode])
        
        args = []
        for j in range(n_args):
            if i + 1 + j < len(genome) and i + 1 + j < sep_pos:
                args.append(str(genome[i + 1 + j]))
            else:
                break
                
        op_str = OP_NAMES.get(opcode, str(val))
        if args:
            op_str += f" {' '.join(args)}"
            
        decoded.append(op_str)
        i += 1 + len(args)
        
    return decoded

def run_illustration(genome_arr, hash_val, folder_path, max_steps):
    output_filename = folder_path / f"genome_{hash_val}_division_logs.txt"
    sys.stdout = TeeLogger(str(output_filename))
    
    print(f"=== Genome Illustration for Hash {hash_val} ===")
    print(f"Run folder: {folder_path}")
    print(f"Output saved to: {output_filename}\n")
    
    # Calculate genome length (excluding trailing padding of -1)
    non_blanks = np.where(genome_arr != -1)[0]
    genome_len = int(non_blanks[-1]) + 1 if len(non_blanks) > 0 else 0
    
    # Create JAX configuration
    cfg = make_config(pop_size=1, initial_pop=1, max_genome_len=max(256, len(genome_arr)))
    
    # Initialize agent state using Agent.init_organism
    agent = Agent.init_organism(
        jnp.array(genome_arr, dtype=jnp.int32),
        jnp.int32(genome_len),
        jnp.int32(-1),
        jnp.int32(UNCLASSIFIED),
        jnp.int32(-1),
        cfg
    )
    
    vm = VirtualMachine(cfg)
    key = random.PRNGKey(42)
    
    print("=== Initial Agent State ===")
    print(f"Agent genome (raw values):\n{genome_arr[:genome_len]}")
    print("Agent genome interpretation:")
    decoded_genome = decode_genome(agent.genome, agent.separator_pos)
    
    line = []
    for item in decoded_genome:
        if item == "I" or item == "SEP":
            if line:
                print("  " + ", ".join(line))
            line = [item]
        elif item == "BLANK":
            if line and "BLANK... (padded to end)" not in line:
                if line:
                    print("  " + ", ".join(line))
                line = ["BLANK... (padded to end)"]
        else:
            line.append(str(item))
            if len(line) > 15:
                print("  " + ", ".join(line))
                line = []
    if line:
        print("  " + ", ".join(line))
        
    print(f"\nInitial Instruction Pointer (IP): {int(agent.se_values[0])}; SEP position: {int(agent.separator_pos)}")
    print("(IP position is right after the SEP)")
    print(f"Genome Length: {int(agent.genome_len)}")
    print(f"Total instructions defined: {int(agent.n_instructions)}")
    
    execute_one_jit = jax.jit(vm.execute_one)
    
    step = 0
    # ANSI Colors
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'
    
    while step < max_steps:
        # Check if agent has successfully divided in the previous step
        if bool(agent.has_child):
            print(f"\n{MAGENTA}[{step}] === Agent has successfully divided! ==={RESET}")
            print(f"Child length: {int(agent.child_len)}")
            child_tape = np.array(agent.child[:int(agent.child_len)])
            print(f"Child tape:\n{child_tape}")
            break
            
        ip = int(agent.se_values[0])
        
        # Fetch the instruction manually just to display it
        if 0 <= ip < agent.genome_len:
            instr_val = int(agent.genome[ip])
        else:
            instr_val = BLANK
            
        print(f"\n{CYAN}{'='*40}{RESET}")
        print(f"{CYAN}--- Step {step} ---{RESET}")
        
        # Visualize full tape up to the last non-blank element
        genome_vals = np.array(agent.genome)
        non_blanks = np.where(genome_vals != -1)[0]
        last_idx = int(non_blanks[-1]) if len(non_blanks) > 0 else 0
        
        tape_snippet = []
        for j in range(last_idx + 1):
            val = int(agent.genome[j])
            if j == ip:
                tape_snippet.append(f"{GREEN}>>[{val}]<<{RESET}")
            else:
                tape_snippet.append(str(val))
        print(f"Tape: {' '.join(tape_snippet)}")
        
        se = np.array(agent.se_values)
        print(f"Registers (SE) (n_ses={int(agent.n_ses)}):")
        print(f"  [0] IP:         {se[0]}    | [1] Counter:    {se[1]}")
        print(f"  [2] Pointer:    {se[2]}    | [3] Loop Start: {se[3]}")
        print(f"  [4] Copy Buffer:{se[4]}")
        if len(se) > 5:
            print(f"  [5-{len(se)-1}] Others:    {se[5:int(agent.n_ses)]}")
            
        if int(agent.n_instructions) > 0 and instr_val != BLANK:
            instr_idx = abs(instr_val) % int(agent.n_instructions)
            length = int(agent.instruction_lengths[instr_idx])
            ops = np.array(agent.instruction_table[instr_idx][:length])
            op_names_list = decode_compound(ops)
            print(f"Compound instruction [{instr_idx}]: {ops}")
            print(f"Micro-ops: {op_names_list}")
            
        if bool(agent.already_allocated):
            copied = int(jnp.sum(agent.child_copied))
            print(f"Child length target: {int(agent.child_len)}, Copied so far: {copied}")
            
            # Show the child tape being populated
            child_tape = np.array(agent.child[:int(agent.child_len)])
            print(f"Child Tape: {child_tape}")
            
        # Execute one compound instruction
        k, key = random.split(key)
        agent = execute_one_jit(agent, k)
        
        ip_after = int(agent.se_values[0])
        print(f"IP after execution: {ip_after}")
        
        step += 1
        
    if step >= max_steps:
        print("\nReached max steps without dividing.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode and illustrate division of a specific genome hash.")
    parser.add_argument("--hash", type=int, default=None, help="The hash of the genome to decode (hyperparamter)")
    parser.add_argument("--folder", type=str, default=None, help="Folder name of the simulation run inside the base path")
    parser.add_argument("--base_path", type=str, default="output", help="Base path of the simulation runs")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum simulation steps to run")
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
        runs = list(base_path.glob("run_*"))
        if not runs:
            print(f"Error: No run folders found in {base_path}")
            sys.exit(1)
            
        folder_path = max(runs, key=lambda p: p.stat().st_mtime)
        print(f"Auto-selected most recent run: {folder_path}")
        
    npz_path = folder_path / "self_replicating_genomes_details.npz"
    fertile_npz_path = folder_path / "fertile_genomes_details.npz"
    if not npz_path.exists() and not fertile_npz_path.exists():
        print(f"Error: Could not find genomes details files: {npz_path} or {fertile_npz_path}")
        sys.exit(1)
        
    # Check if hash was provided
    if args.hash is None:
        print("Error: --hash parameter is required.")
        if npz_path.exists():
            try:
                genomes_data = np.load(npz_path)
                keys_list = list(genomes_data.keys())
                print(f"There are {len(keys_list)} self-replicating genome hashes saved in this run.")
                print("Here are the first 20 available self-replicating hashes:")
                print(keys_list[:20])
            except Exception as e:
                print(f"Error loading {npz_path}: {e}")
        if fertile_npz_path.exists():
            try:
                fertile_data = np.load(fertile_npz_path)
                keys_list = list(fertile_data.keys())
                print(f"There are {len(keys_list)} fertile genome hashes saved in this run.")
                print("Here are the first 20 available fertile hashes:")
                print(keys_list[:20])
            except Exception as e:
                print(f"Error loading {fertile_npz_path}: {e}")
        sys.exit(1)
        
    # Load from genomes_details.npz or fertile_genomes_details.npz
    genomes_data = None
    npz_used_path = None
    hash_str = str(args.hash)
    
    def find_key_in_data(data, h_str):
        # 1. Try direct match
        for k in data.keys():
            if k == h_str or k.startswith(h_str + "_"):
                return k
        # 2. Try 64-bit combined hash decomposition
        try:
            h_int = int(h_str)
            h0 = h_int >> 32
            h1 = h_int & 0xffffffff
            target_key = f"{h0}_{h1}"
            if target_key in data:
                return target_key
        except ValueError:
            pass
        return None
    
    if npz_path.exists():
        try:
            data = np.load(npz_path)
            matching_key = find_key_in_data(data, hash_str)
            if matching_key is not None:
                genomes_data = data
                npz_used_path = npz_path
                hash_str = matching_key
        except Exception as e:
            pass
            
    if genomes_data is None and fertile_npz_path.exists():
        try:
            data = np.load(fertile_npz_path)
            matching_key = find_key_in_data(data, hash_str)
            if matching_key is not None:
                genomes_data = data
                npz_used_path = fertile_npz_path
                hash_str = matching_key
        except Exception as e:
            pass
            
    if genomes_data is None:
        print(f"Error: Genome hash '{args.hash}' not found in genomes_details.npz or fertile_genomes_details.npz.")
        
        # Try to search simulation_stats.pkl to explain the status of this hash
        stats_file = folder_path / "simulation_stats.pkl"
        if stats_file.exists():
            try:
                print("Checking simulation stats to see the status of this hash...")
                import pickle
                with open(stats_file, "rb") as f:
                    stats = pickle.load(f)
                statuses = []
                for chunk in stats:
                    snap = chunk['snapshot']
                    h_arr = snap['hash']
                    s_arr = snap['status']
                    if h_arr.ndim == 2:
                        try:
                            h_int = int(args.hash)
                            h0 = h_int >> 32
                            h1 = h_int & 0xffffffff
                            mask = (h_arr[:, 0] == h0) & (h_arr[:, 1] == h1)
                            if not np.any(mask):
                                mask = (h_arr[:, 0] == args.hash) | (h_arr[:, 1] == args.hash)
                        except ValueError:
                            mask = (h_arr[:, 0] == args.hash) | (h_arr[:, 1] == args.hash)
                    else:
                        mask = h_arr == args.hash
                    if np.any(mask):
                        statuses.extend(list(np.unique(s_arr[mask])))
                if statuses:
                    unique_statuses = sorted(list(set(statuses)))
                    status_names = {0: "UNCLASSIFIED", 1: "SELF_REPLICATING", 2: "FERTILE", 3: "NON_FERTILE", 4: "NON_STANDARD"}
                    status_strs = [status_names.get(s, str(s)) for s in unique_statuses]
                    print(f"\n  Note: Hash {args.hash} was found in the simulation history with status: {', '.join(status_strs)}.")
                    print("  Only SELF_REPLICATING (status 1) and FERTILE (status 2) genomes are saved.")
                    print("  Genomes with status UNCLASSIFIED (status 0), NON_FERTILE (status 3) or NON_STANDARD (status 4) do not have their sequences stored.\n")
            except Exception as e:
                pass
                
        if npz_path.exists():
            try:
                genomes_data = np.load(npz_path)
                keys_list = list(genomes_data.keys())
                print(f"There are {len(keys_list)} self-replicating genome hashes saved in this run.")
                print("Here are the first 20 available self-replicating hashes:")
                print(keys_list[:20])
            except Exception as e:
                pass
        if fertile_npz_path.exists():
            try:
                fertile_data = np.load(fertile_npz_path)
                keys_list = list(fertile_data.keys())
                print(f"There are {len(keys_list)} fertile genome hashes saved in this run.")
                print("Here are the first 20 available fertile hashes:")
                print(keys_list[:20])
            except Exception as e:
                pass
        sys.exit(1)
        
    print(f"Loading genome {args.hash} from {npz_used_path}...")
    genome_arr = genomes_data[hash_str]
    
    run_illustration(genome_arr, args.hash, folder_path, args.max_steps)
    print("Done!")
