"""Instruction Set Architecture (ISA) for the Physis virtual machine.

This module defines *what the machine is*, independent of any particular
simulation run:

  * Opcode constants (``NOP`` … ``IFNOTZERO``) and their human-readable
    ``OP_NAMES`` / operand-arity ``N_OPERANDS`` lookup tables.
  * ``OpState`` / ``OpArgs`` — the per-instruction execution state and the
    decoded-instruction arguments threaded through every opcode.
  * The memory model helpers ``se_read`` / ``se_write`` (scratch registers)
    and ``tape_read`` / ``tape_write`` (parent+child genome tape).
  * ``get_opcode_functions`` — the interpreter core: a list of 44 pure
    ``(state, args) -> state`` functions dispatched by ``jax.lax.switch`` in
    :mod:`physis.analysis.virtual_machine`.

The scalar CUDA port in :mod:`physis.sim.vm_kernel` mirrors these semantics
bit-for-bit; any change to an opcode body here must be mirrored there.

Tunable per-run parameters live in :mod:`physis.sim.config` (``Config``), which
re-exports everything here so existing ``from physis.sim.config import ...`` call
sites keep working.
"""
import jax.numpy as jnp
import jax
from typing import NamedTuple

NOP = 0
IN = 1
OUT = 2
LOAD = 3
STORE = 4
MOVE = 5
ALLOCATE = 6
COMPARE = 7
IFZERO = 8
JUMP = 9
DEC = 10
INC = 11
DIVIDE = 12
SDIR = 13
GDIR = 14
SEND = 15
RECEIVE = 16
ADD = 17
SUB = 18
MUL = 19
DIV_OP = 20
MOD = 21
AND = 22
OR = 23
XOR = 24
NEG = 25
NOT = 26
SHIFT_L = 27
SHIFT_R = 28
FORK_TH = 29
KILL_TH = 30
R = 31
S = 32
Q = 33
I = 34
B = 35
SEP = 36
CLEAR = 37
CINC = 38
CDEC = 39
IS_SEP = 40
REL_LOAD = 41
REL_STORE = 42
IFNOTZERO = 43

UP_IS_SIZE = 44
BLANK = -1

OP_NAMES = {
    0: 'NOP', 1: 'IN', 2: 'OUT', 3: 'LOAD', 4: 'STORE', 5: 'MOVE', 6: 'ALLOCATE',
    7: 'COMPARE', 8: 'IFZERO', 9: 'JUMP', 10: 'DEC', 11: 'INC', 12: 'DIVIDE',
    13: 'SDIR', 14: 'GDIR', 15: 'SEND', 16: 'RECEIVE', 17: 'ADD', 18: 'SUB',
    19: 'MUL', 20: 'DIV_OP', 21: 'MOD', 22: 'AND', 23: 'OR', 24: 'XOR', 25: 'NEG',
    26: 'NOT', 27: 'SHIFT_L', 28: 'SHIFT_R', 29: 'FORK_TH', 30: 'KILL_TH',
    31: 'R', 32: 'S', 33: 'Q', 34: 'I', 35: 'B', 36: 'SEP', 37: 'CLEAR',
    38: 'CINC', 39: 'CDEC', 40: 'IS_SEP', 41: 'REL_LOAD', 42: 'REL_STORE',
    43: 'IFNOTZERO'
}

N_OPERANDS = jnp.array([
    0,  # 0: NOP
    1,  # 1: IN
    1,  # 2: OUT
    2,  # 3: LOAD
    2,  # 4: STORE
    2,  # 5: MOVE
    1,  # 6: ALLOCATE
    3,  # 7: COMPARE
    1,  # 8: IFZERO
    1,  # 9: JUMP
    1,  # 10: DEC
    1,  # 11: INC
    0,  # 12: DIVIDE
    1,  # 13: SDIR
    1,  # 14: GDIR
    1,  # 15: SEND
    1,  # 16: RECEIVE
    3,  # 17: ADD
    3,  # 18: SUB
    3,  # 19: MUL
    3,  # 20: DIV
    3,  # 21: MOD
    3,  # 22: AND
    3,  # 23: OR
    3,  # 24: XOR
    2,  # 25: NEG
    2,  # 26: NOT
    2,  # 27: SHIFT_L
    2,  # 28: SHIFT_R
    1,  # 29: FORK_TH
    0,  # 30: KILL_TH
    0,  # 31: R
    0,  # 32: S
    0,  # 33: Q
    0,  # 34: I
    0,  # 35: B
    0,  # 36: SEPARATOR
    1,  # 37: CLEAR
    1,  # 38: CINC
    1,  # 39: CDEC
    2,  # 40: IS_SEP
    3,  # 41: REL_LOAD
    3,  # 42: REL_STORE
    1,  # 43: IFNOTZERO
], dtype=jnp.int32)


class OpState(NamedTuple):
    se_vals: jnp.ndarray
    child_arr: jnp.ndarray
    child_cop: jnp.ndarray
    genome_arr: jnp.ndarray
    already_alloc: jnp.ndarray
    child_l: jnp.ndarray
    gest_time: jnp.ndarray
    exec_inst: jnp.ndarray
    has_ch: jnp.ndarray
    divide_returned: jnp.ndarray
    did_jump: jnp.ndarray
    read_from_child: jnp.ndarray

class OpArgs(NamedTuple):
    step_key: jnp.ndarray
    genome_len: jnp.ndarray
    n_ses: jnp.ndarray
    tape_size: jnp.ndarray
    cntr: jnp.ndarray
    o0: jnp.ndarray
    o1: jnp.ndarray
    o2: jnp.ndarray
    pos_in_instr: jnp.ndarray
    next_opcode_pos: jnp.ndarray
    ip_for_overflow: jnp.ndarray


def se_read(state: OpState, args: OpArgs, cfg, idx):
    return state.se_vals[jnp.clip(idx, 0, cfg.max_se_count - 1)]

def se_write(state: OpState, args: OpArgs, cfg, idx, val):
    return state._replace(
        se_vals=state.se_vals.at[jnp.clip(idx, 0, cfg.max_se_count - 1)].set(val)
    )

def tape_read(state: OpState, args: OpArgs, position):
    total_size = jnp.maximum(args.tape_size, 1)
    pos = jnp.abs(position) % total_size
    in_parent = pos < args.genome_len
    parent_val = state.genome_arr[jnp.clip(pos, 0, args.genome_len - 1)]
    child_idx = jnp.clip(pos - args.genome_len, 0, state.child_l - 1)
    child_val = state.child_arr[child_idx]
    return jnp.where(in_parent, parent_val, child_val)

def tape_write(state: OpState, args: OpArgs, cfg, position, value):
    total_size = jnp.maximum(args.tape_size, 1)
    pos = jnp.abs(position) % total_size
    in_parent = pos < args.genome_len

    # Parent write
    parent_idx = jnp.clip(pos, 0, args.genome_len - 1)
    new_genome = jnp.where(in_parent, state.genome_arr.at[parent_idx].set(value), state.genome_arr)

    # Child write
    child_idx = jnp.clip(pos - args.genome_len, 0, state.child_l - 1)
    in_child = ~in_parent & state.already_alloc
    # Copy mutation must NOT be applied here because the CUDA kernel (vm_kernel.py)
    # cannot easily execute random ops per-instruction. It is instead applied at
    # the end of division in Agent.apply_divide_mutations.
    # k1, k2 = random.split(args.step_key)
    # do_mutate = random.uniform(k1) < cfg.copy_mutation_rate
    # mutated_value = random.randint(k2, (), 0, UP_IS_SIZE).astype(jnp.int32)
    # final_value = jnp.where(do_mutate & in_child, mutated_value, value)
    # new_child = jnp.where(in_child, state.child_arr.at[child_idx].set(final_value), state.child_arr)

    new_child = jnp.where(in_child, state.child_arr.at[child_idx].set(value), state.child_arr)
    new_child_cop = jnp.where(in_child, state.child_cop.at[child_idx].set(True), state.child_cop)

    return state._replace(genome_arr=new_genome, child_arr=new_child, child_cop=new_child_cop)


def get_opcode_functions(cfg):
    """Returns a list of 44 pure functions mapping (state, args) -> state for jax.lax.switch."""

    def op_nop(op_input) -> OpState:
        state, args = op_input
        """No operation."""
        return state

    def op_load(op_input) -> OpState:
        state, args = op_input
        """Load from tape at address SE[o0] into SE[o1]."""
        addr = se_read(state, args, cfg, args.o0)
        val = tape_read(state, args, addr)

        # Track if read from child
        total_size = jnp.maximum(args.tape_size, 1)
        pos = jnp.abs(addr) % total_size
        read_child = pos >= args.genome_len
        state = state._replace(read_from_child=state.read_from_child | read_child)

        return se_write(state, args, cfg, args.o1, val)

    def op_store(op_input) -> OpState:
        state, args = op_input
        """Store value from SE[o1] into tape at address SE[o0]."""
        return tape_write(state, args, cfg, se_read(state, args, cfg, args.o0), se_read(state, args, cfg, args.o1))

    def op_move(op_input) -> OpState:
        state, args = op_input
        """Copy value from SE[o0] into SE[o1]."""
        return se_write(state, args, cfg, args.o1, se_read(state, args, cfg, args.o0))

    def op_allocate(op_input) -> OpState:
        state, args = op_input
        """Allocate child tape of size SE[o0]."""
        alloc_size = se_read(state, args, cfg, args.o0)
        alloc_possible = (
            ~state.already_alloc &
            (alloc_size > (cfg.min_allocation_ratio * args.genome_len).astype(jnp.int32)) &
            (alloc_size < (cfg.max_allocation_ratio * args.genome_len).astype(jnp.int32))
        )
        blank_child = jnp.full(cfg.max_genome_len, BLANK, dtype=jnp.int32)
        blank_cop = jnp.zeros(cfg.max_genome_len, dtype=jnp.bool_)

        return state._replace(
            already_alloc=jnp.where(alloc_possible, True, state.already_alloc),
            child_l=jnp.where(alloc_possible, alloc_size, state.child_l),
            child_arr=jnp.where(alloc_possible, blank_child, state.child_arr),
            child_cop=jnp.where(alloc_possible, blank_cop, state.child_cop)
        )

    def op_compare(op_input) -> OpState:
        state, args = op_input
        """Compare SE[o0] and SE[o1], write result (-1, 0, 1) into SE[o2]."""
        a, b = se_read(state, args, cfg, args.o0), se_read(state, args, cfg, args.o1)
        res = jnp.where(a < b, -1, jnp.where(a == b, 0, 1))
        return se_write(state, args, cfg, args.o2, res)

    def op_ifzero(op_input) -> OpState:
        state, args = op_input
        """If SE[o0] is NOT zero, skip the next instruction (IP += 1)."""
        skip = se_read(state, args, cfg, args.o0) != 0
        return se_write(state, args, cfg, jnp.int32(0), jnp.where(skip, se_read(state, args, cfg, jnp.int32(0)) + 1, se_read(state, args, cfg, jnp.int32(0))))

    def op_jump(op_input) -> OpState:
        state, args = op_input
        """Jump IP to the value stored in SE[o0]."""
        return se_write(state, args, cfg, jnp.int32(0), se_read(state, args, cfg, args.o0))._replace(did_jump=jnp.bool_(True))

    def op_dec(op_input) -> OpState:
        state, args = op_input
        """Decrement SE[o0] by 1."""
        return se_write(state, args, cfg, args.o0, se_read(state, args, cfg, args.o0) - 1)

    def op_inc(op_input) -> OpState:
        state, args = op_input
        """Increment SE[o0] by 1."""
        return se_write(state, args, cfg, args.o0, se_read(state, args, cfg, args.o0) + 1)

    def op_divide(op_input) -> OpState:
        state, args = op_input
        """Trigger cell division if copying thresholds are met."""
        n_copied = jnp.sum(state.child_cop.astype(jnp.int32))
        prolif_possible = state.already_alloc & (n_copied > (cfg.min_proliferation_ratio * args.genome_len).astype(jnp.int32))

        new_se_vals = jnp.where(
            ~prolif_possible,
            state.se_vals.at[jnp.clip(jnp.int32(0), 0, cfg.max_se_count - 1)].set(se_read(state, args, cfg, jnp.int32(0)) + 1),
            state.se_vals
        )
        return state._replace(
            exec_inst=jnp.where(prolif_possible, args.cntr, state.exec_inst),
            has_ch=jnp.where(prolif_possible, True, state.has_ch),
            divide_returned=jnp.bool_(True),
            se_vals=new_se_vals
        )

    def op_add(op_input) -> OpState:
        state, args = op_input
        """Add SE[o0] and SE[o1], store in SE[o2]."""
        return se_write(state, args, cfg, args.o2, se_read(state, args, cfg, args.o0) + se_read(state, args, cfg, args.o1))

    def op_sub(op_input) -> OpState:
        state, args = op_input
        """Subtract SE[o1] from SE[o0], store in SE[o2]."""
        return se_write(state, args, cfg, args.o2, se_read(state, args, cfg, args.o0) - se_read(state, args, cfg, args.o1))

    def op_mul(op_input) -> OpState:
        state, args = op_input
        """Multiply SE[o0] and SE[o1], store in SE[o2]."""
        return se_write(state, args, cfg, args.o2, se_read(state, args, cfg, args.o0) * se_read(state, args, cfg, args.o1))

    def op_div(op_input) -> OpState:
        state, args = op_input
        """Divide SE[o0] by SE[o1], store in SE[o2]."""
        div = se_read(state, args, cfg, args.o1)
        safe_div = jnp.where(div == 0, 1, div)
        res = se_read(state, args, cfg, args.o0) // safe_div
        new_state = se_write(state, args, cfg, args.o2, res)
        return jax.tree.map(lambda n, o: jnp.where(div != 0, n, o), new_state, state)

    def op_mod(op_input) -> OpState:
        state, args = op_input
        """Modulo SE[o0] by SE[o1], store in SE[o2]."""
        div = se_read(state, args, cfg, args.o1)
        safe_div = jnp.where(div == 0, 1, div)
        res = se_read(state, args, cfg, args.o0) % safe_div
        new_state = se_write(state, args, cfg, args.o2, res)
        return jax.tree.map(lambda n, o: jnp.where(div != 0, n, o), new_state, state)

    def op_and(op_input) -> OpState:
        state, args = op_input
        """Bitwise AND of SE[o0] and SE[o1], store in SE[o2]."""
        return se_write(state, args, cfg, args.o2, se_read(state, args, cfg, args.o0) & se_read(state, args, cfg, args.o1))

    def op_or(op_input) -> OpState:
        state, args = op_input
        """Bitwise OR of SE[o0] and SE[o1], store in SE[o2]."""
        return se_write(state, args, cfg, args.o2, se_read(state, args, cfg, args.o0) | se_read(state, args, cfg, args.o1))

    def op_xor(op_input) -> OpState:
        state, args = op_input
        """Bitwise XOR of SE[o0] and SE[o1], store in SE[o2]."""
        return se_write(state, args, cfg, args.o2, se_read(state, args, cfg, args.o0) ^ se_read(state, args, cfg, args.o1))

    def op_neg(op_input) -> OpState:
        state, args = op_input
        """Negate SE[o0], store in SE[o1]."""
        return se_write(state, args, cfg, args.o1, -se_read(state, args, cfg, args.o0))

    def op_not(op_input) -> OpState:
        state, args = op_input
        """Bitwise NOT of SE[o0], store in SE[o1]."""
        return se_write(state, args, cfg, args.o1, ~se_read(state, args, cfg, args.o0))

    def op_shift_l(op_input) -> OpState:
        state, args = op_input
        """Bitwise left shift SE[o0] by 1, store in SE[o1]."""
        return se_write(state, args, cfg, args.o1, se_read(state, args, cfg, args.o0) << 1)

    def op_shift_r(op_input) -> OpState:
        state, args = op_input
        """Bitwise right shift SE[o0] by 1, store in SE[o1]."""
        return se_write(state, args, cfg, args.o1, se_read(state, args, cfg, args.o0) >> 1)

    def op_clear(op_input) -> OpState:
        state, args = op_input
        """Set SE[o0] to 0."""
        return se_write(state, args, cfg, args.o0, 0)

    def op_cinc(op_input) -> OpState:
        state, args = op_input
        """Circular increment SE[o0] (wraps around at tape_size)."""
        return se_write(state, args, cfg, args.o0, (se_read(state, args, cfg, args.o0) + 1) % jnp.maximum(args.tape_size, 1))

    def op_cdec(op_input) -> OpState:
        state, args = op_input
        """Circular decrement SE[o0] (wraps around at tape_size)."""
        val = se_read(state, args, cfg, args.o0) - 1
        sz = jnp.maximum(args.tape_size, 1)
        return se_write(state, args, cfg, args.o0, jnp.where(val < 0, sz - 1, val))

    def op_is_sep(op_input) -> OpState:
        state, args = op_input
        """Check if tape[SE[o0]] == SEP token. Store 1 (True) or 0 (False) in SE[o1]."""
        return se_write(state, args, cfg, args.o1, jnp.where(se_read(state, args, cfg, args.o0) == SEP, jnp.int32(1), jnp.int32(0)))

    def op_rel_load(op_input) -> OpState:
        state, args = op_input
        """Load from tape at (SE[o0] + SE[o1]) into SE[o2]."""
        addr = se_read(state, args, cfg, args.o0) + se_read(state, args, cfg, args.o1)
        val = tape_read(state, args, addr)

        # Track if read from child
        total_size = jnp.maximum(args.tape_size, 1)
        pos = jnp.abs(addr) % total_size
        read_child = pos >= args.genome_len
        state = state._replace(read_from_child=state.read_from_child | read_child)

        return se_write(state, args, cfg, args.o2, val)

    def op_rel_store(op_input) -> OpState:
        state, args = op_input
        """Store SE[o2] to tape at (SE[o0] + SE[o1])."""
        addr = se_read(state, args, cfg, args.o0) + se_read(state, args, cfg, args.o1)
        return tape_write(state, args, cfg, addr, se_read(state, args, cfg, args.o2))

    def op_ifnotzero(op_input) -> OpState:
        state, args = op_input
        """If SE[o0] IS zero, skip the next instruction (IP += 1)."""
        skip = se_read(state, args, cfg, args.o0) == 0
        return se_write(state, args, cfg, jnp.int32(0), jnp.where(skip, se_read(state, args, cfg, jnp.int32(0)) + 1, se_read(state, args, cfg, jnp.int32(0))))

    return [
        op_nop, op_nop, op_nop, op_load, op_store, op_move, op_allocate, op_compare,
        op_ifzero, op_jump, op_dec, op_inc, op_divide, op_nop, op_nop, op_nop, op_nop,
        op_add, op_sub, op_mul, op_div, op_mod, op_and, op_or, op_xor, op_neg, op_not,
        op_shift_l, op_shift_r, op_nop, op_nop, op_nop, op_nop, op_nop, op_nop, op_nop,
        op_nop, op_clear, op_cinc, op_cdec, op_is_sep, op_rel_load, op_rel_store, op_ifnotzero
    ]
