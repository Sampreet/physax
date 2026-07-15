import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import sys
import re

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

def run_test():
    # Redirect stdout to save to both terminal and file
    sys.stdout = TeeLogger("full_division_logs.txt")
    
    cfg = make_config(pop_size=1, initial_pop=1)
    
    # Initialize the ancestor genome
    ancestor_genome, ancestor_len = Agent.create_ancestor_genome(cfg)
    # Create the agent state
    agent = Agent.init_organism(
        ancestor_genome,
        ancestor_len,
        jnp.int32(-1),
        jnp.int32(UNCLASSIFIED),
        jnp.int32(-1),
        cfg
    )
    vm = VirtualMachine(cfg)
    key = random.PRNGKey(42)
    
    print("=== Initial Agent State ===")
    print(f"Agent genome (raw values):\n{agent.genome}")
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
            # Wrap execution instructions after SEP
            if len(line) > 15:
                print("  " + ", ".join(line))
                line = []
    if line:
        print("  " + ", ".join(line))

    print(f"\nInitial Instruction Pointer (IP): {int(agent.se_values[0])}; SEP position: {int(agent.separator_pos)}")
    print("(IP postition is right after the SEP)")
    print(f"Genome Length: {int(agent.genome_len)}")
    print(f"Total instructions defined: {int(agent.n_instructions)}")
    
    # JIT compile the execution step for speed, though we run it step-by-step
    execute_one_jit = jax.jit(vm.execute_one)
    
    step = 0
    max_steps = 10_000 * cfg.steps_per_update  # 10,000 cycles * 34 steps/cycle
    
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
        
        # Visualize full tape up to the first BLANK
        genome_vals = np.array(agent.genome)
        blanks = np.where(genome_vals == -1)[0]
        last_idx = blanks[0] if len(blanks) > 0 else len(genome_vals) - 1
        
        tape_snippet = []
        for j in range(last_idx + 1):
            val = int(agent.genome[j])
            if j == ip:
                tape_snippet.append(f"{GREEN}>>[{val}]<<{RESET}")
            else:
                tape_snippet.append(str(val))
        print(f"Tape: {' '.join(tape_snippet)}")
        
        se = np.array(agent.se_values)
        print(f"Registers (SE):")
        print(f"  [0] IP:         {se[0]}    | [1] Counter:    {se[1]}")
        print(f"  [2] Pointer:    {se[2]}    | [3] Loop Start: {se[3]}")
        print(f"  [4] Copy Buffer:{se[4]}    | [5-15] Unused:  {se[5:16]}")
        
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
    run_test()
