import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random
from typing import NamedTuple
from physax.config import Config, N_OPERANDS, UP_IS_SIZE, BLANK, I, SEP, R, S, Q, B, MOVE, NOP, CLEAR, INC, CINC, LOAD, IS_SEP, IFZERO, JUMP, ALLOCATE, REL_STORE, DEC, IFNOTZERO, DIVIDE, UNCLASSIFIED, SELF_REPLICATING, FERTILE, NON_FERTILE, NON_STANDARD

class Agent(NamedTuple):
    """Immutable state representation of a single organism."""
    genome: jnp.ndarray
    genome_len: jnp.ndarray
    se_values: jnp.ndarray
    n_ses: jnp.ndarray
    separator_pos: jnp.ndarray
    n_instructions: jnp.ndarray
    instruction_table: jnp.ndarray
    instruction_lengths: jnp.ndarray
    child: jnp.ndarray
    child_len: jnp.ndarray
    child_copied: jnp.ndarray
    already_allocated: jnp.ndarray
    age: jnp.ndarray
    alive: jnp.ndarray
    has_child: jnp.ndarray
    counter: jnp.ndarray
    gestation_time: jnp.ndarray
    executed_instructions: jnp.ndarray
    child_tape: jnp.ndarray
    child_tape_len: jnp.ndarray
    executed: jnp.ndarray
    read_from_child: jnp.ndarray
    status: jnp.ndarray
    genome_hash: jnp.ndarray

    @property
    def can_execute(self) -> jnp.ndarray:
        """Returns True if the organism is alive and hasn't just divided."""
        return self.alive & ~self.has_child

    @property
    def is_fertile(self) -> jnp.ndarray:
        """Returns True if the organism has successfully reproduced."""
        return self.gestation_time < 2147483647

    @classmethod
    def create_empty(cls, cfg: Config) -> "Agent":
        """Create an empty organism state with default initialized arrays."""
        return cls(
            genome=jnp.full(cfg.max_genome_len, BLANK, dtype=jnp.int32),
            genome_len=jnp.int32(0),
            # Structural elements: SE[0] = IP, SE[1..] = registers/stacks/queues
            se_values=jnp.zeros(cfg.max_se_count, dtype=jnp.int32),
            n_ses=jnp.int32(1),  # At least IP
            separator_pos=jnp.int32(0),
            n_instructions=jnp.int32(0),
            # Instruction table: raw normalized micro-ops per instruction
            instruction_table=jnp.full((cfg.max_instructions, cfg.max_micro_ops), BLANK, dtype=jnp.int32),
            instruction_lengths=jnp.zeros(cfg.max_instructions, dtype=jnp.int32),
            # Child allocation
            child=jnp.full(cfg.max_genome_len, BLANK, dtype=jnp.int32),
            child_len=jnp.int32(0),
            child_copied=jnp.zeros(cfg.max_genome_len, dtype=jnp.bool_),
            already_allocated=jnp.bool_(False),
            # Execution state
            age=jnp.int32(0),
            alive=jnp.bool_(True),
            has_child=jnp.bool_(False),
            counter=jnp.int32(0),
            gestation_time=jnp.int32(2147483647),
            executed_instructions=jnp.int32(0),
            # Child tape for placement after divide
            child_tape=jnp.full(cfg.max_genome_len, BLANK, dtype=jnp.int32),
            child_tape_len=jnp.int32(0),
            # Executed tracking (for fitness/merit computation)
            executed=jnp.zeros(cfg.max_genome_len, dtype=jnp.bool_),
            # Caching
            read_from_child=jnp.bool_(False),
            status=jnp.int32(UNCLASSIFIED),
            genome_hash=jnp.zeros(2, dtype=jnp.int32),
        )

    @staticmethod
    def create_ancestor_genome(cfg: Config):
        """Create the 78-gene arche.replicator ancestor genome."""
        # Hardcoded from physis/data/genebank/arche/arche.replicator
        g = jnp.array([
            R, B, R, B, R, B, R,                           # Structure: 4 registers (+ IP = 5 SEs)
            I, MOVE, NOP, 2,                                # I0: move 0 2 (store_ptr)
            I, CLEAR, 1,                                    # I1: clear 1 (clear_counter)
            I, MOVE, NOP, 3,                                # I2: move 0 3 (s_lab: store IP to return_reg)
            I, INC, 1,                                      # I3: inc 1 (inc_counter)
            I, CINC, 2,                                     # I4: cinc 2 (cinc_pointer)
            I, LOAD, 2, 4, IS_SEP, 4, 4, IFZERO, 4,        # I5: load 2 4; is_sep 4 4; ifzero 4
            I, JUMP, 3,                                     # I6: jump 3 (jump_by_reg)
            I, ALLOCATE, 1,                                 # I7: allocate 1
            I, MOVE, 1, 2,                                  # I8: move 1 2 (fill_pointer)
            I, LOAD, 1, 4,                                  # I9: load 1 4 (load_data)
            I, REL_STORE, 1, 2, 4,                          # I10: rel_store 1 2 4
            I, DEC, 1,                                      # I11: dec 1
            I, IFNOTZERO, 1,                                # I12: ifnotzero 1
            I, DIVIDE,                                      # I13: divide
            SEP,                                            # Separator at position 60
            # Code section (positions 61-77): 17 entries mapping to 14 instructions
            0, 1, 2, 3, 4, 5, 6, 3, 7, 8, 2, 11, 9, 10, 12, 6, 13
        ], dtype=jnp.int32)

        genome = jnp.full(cfg.max_genome_len, BLANK, dtype=jnp.int32)
        genome = genome.at[:len(g)].set(g)
        genome_len = jnp.int32(len(g))

        return genome, genome_len

    @classmethod
    def init_organism(cls, genome, genome_len, parent_hash, parent_status, parent_gestation, cfg: Config) -> "Agent":
        """Initialize a new living organism from a genome. 
        Parses the structure and instructions automatically."""
        parsed = cls._parse_genome(genome, genome_len, cfg)
        
        # IP starts right after separator
        ip_start = (parsed['separator_pos'] + 1) % jnp.maximum(genome_len, 1)
        
        hash_val = cls._hash_genome(genome, genome_len, cfg)
        hash_match = jnp.all(hash_val == parent_hash)
        
        state = cls.create_empty(cfg)
        return state._replace(
            genome=genome,
            genome_len=genome_len,
            alive=jnp.bool_(True),
            n_ses=parsed['n_ses'],
            separator_pos=parsed['separator_pos'],
            n_instructions=parsed['n_instructions'],
            instruction_table=parsed['instruction_table'],
            instruction_lengths=parsed['instruction_lengths'],
            se_values=state.se_values.at[0].set(ip_start),
            genome_hash=hash_val,
            status=jnp.where(hash_match, parent_status, state.status),
            gestation_time=jnp.where(hash_match, parent_gestation, state.gestation_time),
            executed_instructions=jnp.where(hash_match, jnp.int32(0), state.executed_instructions),
            child=jnp.where(hash_match, genome, state.child),
            child_len=jnp.where(hash_match, genome_len, state.child_len),
            child_copied=jnp.where(
                hash_match, 
                jnp.arange(cfg.max_genome_len) < genome_len, 
                state.child_copied
            )
        )

    @classmethod
    def _hash_genome(cls, genome, genome_len, cfg: Config):
        # Dual rolling hash to prevent collisions
        positions = jnp.arange(cfg.max_genome_len)
        valid = positions < genome_len
        
        def hash_step(carry, i):
            h1, h2 = carry
            val = jnp.where(valid[i], genome[i], jnp.int32(0))
            new_h1 = (h1 * jnp.int32(31) + val) % jnp.int32(2147483647)
            new_h2 = (h2 * jnp.int32(37) + val) % jnp.int32(2147483579)
            return (new_h1, new_h2), None
            
        (h1_final, h2_final), _ = lax.scan(hash_step, (jnp.int32(0), jnp.int32(0)), positions)
        return jnp.stack([h1_final, h2_final])

    @classmethod
    def _parse_genome(cls, genome, genome_len, cfg: Config):
        """Private method to map genotype to phenotype structure and instructions."""
        n_ses, separator_pos = cls._build_structure(genome, genome_len, cfg)
        instruction_table, instruction_lengths, n_instructions = cls._build_instruction_set(
            genome, genome_len, separator_pos, cfg
        )
        return {
            'n_ses': n_ses,
            'separator_pos': separator_pos,
            'n_instructions': n_instructions,
            'instruction_table': instruction_table,
            'instruction_lengths': instruction_lengths,
        }

    @staticmethod
    def _build_structure(genome, genome_len, cfg: Config):
        """Scan genome before SEP. Count R/S/Q markers as structural elements.
        SE[0] is always IP (implicit). R/S/Q each add one SE (simplification:
        all treated as registers; stacks/queues not deeply supported).
        Returns n_ses, separator_pos.
        """
        # Scan positions 0..genome_len-1 looking for R, S, Q, SEP
        # Stop at first SEP
        positions = jnp.arange(cfg.max_genome_len)
        valid = positions < genome_len

        def scan_fn(carry, pos):
            n_ses, sep_pos, found_sep = carry
            gene = jnp.where(valid[pos], genome[pos], BLANK)
            is_sep = (gene == SEP) & ~found_sep
            is_r = (gene == R) & ~found_sep
            is_s = (gene == S) & ~found_sep
            is_q = (gene == Q) & ~found_sep
            is_se = is_r | is_s | is_q
            n_ses = jnp.where(is_se & (n_ses < cfg.max_se_count), n_ses + 1, n_ses)
            sep_pos = jnp.where(is_sep, pos, sep_pos)
            found_sep = found_sep | is_sep
            return (n_ses, sep_pos, found_sep), None

        # Start with 1 SE (the implicit IP = SE[0])
        (n_ses, separator_pos, _), _ = lax.scan(
            scan_fn,
            (jnp.int32(1), jnp.int32(0), jnp.bool_(False)),
            jnp.arange(cfg.max_genome_len)
        )

        return n_ses, separator_pos

    @staticmethod
    def _build_instruction_set(genome, genome_len, separator_pos, cfg: Config):
        """Build instruction table from genome, matching UP.java buildInstructionSet + createInstruction.

        Scans genome[0..separator_pos-1] for I markers.
        Each pair of consecutive I markers (or I→SEP) defines one instruction.
        Content between markers is read, opcodes normalized via abs(val) % UP_IS_SIZE,
        then operands skipped based on N_OPERANDS lookup.
        """
        positions = jnp.arange(cfg.max_genome_len)
        valid = positions < separator_pos

        # Find I marker positions
        is_i_marker = (genome == I) & valid
        # Collect I positions (sorted, padded with max_genome_len)
        i_positions = jnp.where(is_i_marker, positions, cfg.max_genome_len)
        i_positions = jnp.sort(i_positions)
        # Count actual I markers
        n_i_markers = jnp.sum(is_i_marker.astype(jnp.int32))

        def create_one_instruction(instr_idx):
            """Create instruction number instr_idx."""
            # For each instruction index, determine start (after I marker) and stop (next I or SEP)
            i_pos = i_positions[instr_idx]
            valid_instr = instr_idx < n_i_markers
            start = i_pos + 1

            # Find the end: next I marker or separator_pos
            next_i_pos = jnp.where(
                (instr_idx + 1) < n_i_markers,
                i_positions[instr_idx + 1],
                separator_pos
            )
            stop = next_i_pos
            size = stop - start

            # Read raw content from genome[start:stop]
            raw = jnp.full(cfg.max_micro_ops, BLANK, dtype=jnp.int32)
            def read_raw(j):
                pos = start + j
                return jnp.where((j < size) & (pos < genome_len), genome[pos], BLANK)
            raw = jax.vmap(read_raw)(jnp.arange(cfg.max_micro_ops))

            # Normalize: walk through raw, normalize opcodes, skip operands
            def normalize_step(carry, j):
                out_arr, next_opcode_pos = carry
                is_opcode_pos = (j == next_opcode_pos) & (j < size)
                val = raw[j]
                # Normalize opcode: abs(val) % UP_IS_SIZE
                normalized = jnp.abs(val) % UP_IS_SIZE
                # Look up operand count for this opcode
                n_ops = N_OPERANDS[normalized]
                # If this is an opcode position, normalize it; operand positions stay raw
                out_val = jnp.where(is_opcode_pos, normalized, val)
                out_arr = jnp.where(j < size, out_arr.at[j].set(out_val), out_arr)
                # Next opcode position is current + 1 + n_ops
                next_opcode_pos = jnp.where(is_opcode_pos, j + 1 + n_ops, next_opcode_pos)
                return (out_arr, next_opcode_pos), None

            init_arr = jnp.full(cfg.max_micro_ops, BLANK, dtype=jnp.int32)
            (result_arr, _), _ = lax.scan(
                normalize_step,
                (init_arr, jnp.int32(0)),  # First opcode at position 0
                jnp.arange(cfg.max_micro_ops)
            )

            # Mask invalid instruction
            result_arr = jnp.where(valid_instr, result_arr, jnp.full(cfg.max_micro_ops, BLANK, dtype=jnp.int32))
            length = jnp.where(valid_instr, jnp.minimum(size, cfg.max_micro_ops), jnp.int32(0))

            return result_arr, length

        instruction_table, instruction_lengths = jax.vmap(create_one_instruction)(
            jnp.arange(cfg.max_instructions)
        )
        n_instructions = jnp.minimum(n_i_markers, cfg.max_instructions)

        return instruction_table, instruction_lengths, n_instructions

    @staticmethod
    def apply_divide_mutations(key, child_tape, child_tape_len, status, cfg: Config):
        """Apply divide mutations to child tape after successful divide.
        Order: point mutation, insertion, deletion (matching CellGeneticCodeTape.divide()).
        Also applies deferred copy_mutation if status is WELL_BEHAVED.
        """
        k_copy_1, k_copy_2, k1, k2, k3, k4, k5, k6, k7 = random.split(key, 9)

        # 0. Deferred copy mutation (applied to all dividing agents)
        do_copy = random.uniform(k_copy_1, (cfg.max_genome_len,)) < cfg.copy_mutation_rate
        copy_vals = random.randint(k_copy_2, (cfg.max_genome_len,), 0, UP_IS_SIZE).astype(jnp.int32)
        valid_mask = jnp.arange(cfg.max_genome_len) < child_tape_len
        child_tape = jnp.where(do_copy & valid_mask, copy_vals, child_tape)

        # 1. Point mutation
        do_point = random.uniform(k1) < cfg.divide_mutation_rate
        point_pos = random.randint(k2, (), 0, jnp.maximum(child_tape_len, 1))
        point_val = random.randint(k3, (), 0, UP_IS_SIZE).astype(jnp.int32)
        child_tape = jnp.where(
            do_point & (point_pos < child_tape_len),
            child_tape.at[point_pos].set(point_val),
            child_tape
        )

        # Insertion (rate = divide_insert_rate)
        do_insert = random.uniform(k4) < cfg.divide_insert_rate
        insert_pos = random.randint(k5, (), 0, jnp.maximum(child_tape_len + 1, 1))
        insert_val = random.randint(k3, (), 0, UP_IS_SIZE).astype(jnp.int32)
        can_insert = do_insert & (child_tape_len < cfg.max_genome_len - 1)
        # Shift right from insert_pos
        indices = jnp.arange(cfg.max_genome_len)
        shifted_right = jnp.where(
            indices > insert_pos,
            jnp.where(indices - 1 < cfg.max_genome_len, child_tape[jnp.clip(indices - 1, 0, cfg.max_genome_len - 1)], BLANK),
            child_tape[indices]
        )
        shifted_right = shifted_right.at[insert_pos].set(insert_val)
        child_tape = jnp.where(can_insert, shifted_right, child_tape)
        child_tape_len = jnp.where(can_insert, child_tape_len + 1, child_tape_len)

        # Deletion (rate = divide_delete_rate)
        do_delete = random.uniform(k6) < cfg.divide_delete_rate
        can_delete = do_delete & (child_tape_len > 1)
        delete_pos = random.randint(k7, (), 0, jnp.maximum(child_tape_len - 1, 1))
        # Shift left from delete_pos
        shifted_left = jnp.where(
            indices >= delete_pos,
            jnp.where(indices + 1 < cfg.max_genome_len, child_tape[jnp.clip(indices + 1, 0, cfg.max_genome_len - 1)], BLANK),
            child_tape[indices]
        )
        child_tape = jnp.where(can_delete, shifted_left, child_tape)
        child_tape_len = jnp.where(can_delete, child_tape_len - 1, child_tape_len)

        return child_tape, child_tape_len
