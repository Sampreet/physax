"""Numba CUDA kernel port of VirtualMachine.update (the slow-track interpreter).

One CUDA thread runs one organism: it interprets that organism's genome for
`steps_per_update` compound instructions, exactly reproducing the semantics of
physis/virtual_machine.py so results are bitwise-identical to the JAX VM.

Why this exists: under jax.vmap the opcode dispatch (lax.switch over 44 ops)
evaluates *every* branch for *every* agent at *every* micro-op, and all state
lives in full-population arrays that ping-pong through HBM. A thread-per-cell
kernel executes only the taken branch and keeps each organism's working set in
its own memory -- see the cubff reference under inspiration/ for the same idea.

The VM is deterministic (no opcode consumes randomness), so no RNG is needed
here and the port can be validated bitwise against the JAX implementation.
"""

import numpy as np
import jax
from numba import cuda, int32

# --- constants mirrored from physis/config.py (kept in sync by value) ---
BLANK = -1
SEP = 36
NOP = 0
UP_IS_SIZE = 44          # number of opcodes / normalization modulus
SHORT_MAX = 32767        # operand normalization modulus (Short.MAX_VALUE)

# Operands consumed by each opcode, indexed by opcode value 0..43.
# Must match config.N_OPERANDS exactly.
_N_OPERANDS = np.array([
    0, 1, 1, 2, 2, 2, 1, 3, 1, 1, 1, 1, 0, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3,
    3, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 3, 3, 1,
], dtype=np.int32)


@cuda.jit(device=True, inline=True)
def _clip(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


@cuda.jit(device=True, inline=True)
def _tape_read(G, C, i, pos, tape_size, genome_len, child_len):
    """Read one tape cell: parent genome for pos < genome_len, else child."""
    total = tape_size if tape_size > 1 else 1
    p = (pos if pos >= 0 else -pos) % total
    if p < genome_len:
        return G[i, _clip(p, 0, genome_len - 1)]
    return C[i, _clip(p - genome_len, 0, child_len - 1 if child_len > 0 else 0)]


@cuda.jit(device=True, inline=True)
def _tape_write(G, C, CC, i, pos, val, tape_size, genome_len, child_len, already_alloc):
    total = tape_size if tape_size > 1 else 1
    p = (pos if pos >= 0 else -pos) % total
    if p < genome_len:
        G[i, _clip(p, 0, genome_len - 1)] = val
    elif already_alloc:
        ci = _clip(p - genome_len, 0, child_len - 1 if child_len > 0 else 0)
        C[i, ci] = val
        CC[i, ci] = 1


def build_vm_kernel(cfg):
    """Compile the VM kernel for a given Config (bakes in the fixed sizes and
    allocation/proliferation thresholds). Returns the launchable kernel."""
    MAX_GENOME = int(cfg.max_genome_len)
    MAX_SE = int(cfg.max_se_count)
    MAX_MICRO = int(cfg.max_micro_ops)
    STEPS = int(cfg.steps_per_update)
    MIN_ALLOC_R = float(cfg.min_allocation_ratio)
    MAX_ALLOC_R = float(cfg.max_allocation_ratio)
    MIN_PROLIF_R = float(cfg.min_proliferation_ratio)

    @cuda.jit
    def vm_kernel(
        is_slow,            # bool[pop]        run mask (alive & slow-track)
        alive,             # bool[pop]
        # structural (read-only during update)
        genome_len,        # int32[pop]
        n_ses,             # int32[pop]
        n_instructions,    # int32[pop]
        separator_pos,     # int32[pop]
        instr_table,       # int32[pop, MAX_INSTR, MAX_MICRO]
        instr_lengths,     # int32[pop, MAX_INSTR]
        N_OPERANDS,        # int32[44]
        # mutable state (updated in place for is_slow rows)
        S,                 # int32[pop, MAX_SE]      se_values (S[i,0] = IP)
        G,                 # int32[pop, MAX_GENOME]  genome
        C,                 # int32[pop, MAX_GENOME]  child
        CC,                # int8[pop, MAX_GENOME]   child_copied
        executed,          # int8[pop, MAX_GENOME]
        already_alloc,     # int8[pop]
        child_len,         # int32[pop]
        has_child,         # int8[pop]
        counter,           # int32[pop]
        executed_inst,     # int32[pop]
    ):
        i = cuda.grid(1)
        if i >= is_slow.shape[0]:
            return
        if not is_slow[i]:
            return

        # per-thread mutable scalars (write back at the end)
        gl = genome_len[i]
        nses = n_ses[i]
        ninstr = n_instructions[i]
        sep = separator_pos[i]
        c_len = child_len[i]
        alloc = already_alloc[i]
        haschild = has_child[i]
        cntr = counter[i]
        exec_inst = executed_inst[i]
        entry_haschild_update = haschild  # has_child at entry to each execute_one

        for _step in range(STEPS):
            # can_execute = alive & ~has_child (evaluated per step)
            if (not alive[i]) or haschild:
                break

            entry_haschild = haschild
            ip_val = S[i, 0]

            # 1. fetch instruction from parent tape
            in_bounds = (ip_val >= 0) and (ip_val < gl)
            fetched = G[i, _clip(ip_val, 0, gl - 1 if gl > 0 else 0)] if in_bounds else BLANK

            # mark IP position executed
            if in_bounds:
                executed[i, _clip(ip_val, 0, MAX_GENOME - 1)] = 1

            # 2. map to instruction index
            safe_n = ninstr if ninstr > 1 else 1
            af = fetched if fetched >= 0 else -fetched
            instr_idx = (af % safe_n) if ninstr > 0 else 0
            instr_len = instr_lengths[i, instr_idx]

            # tape size is fixed for the whole compound instruction
            tape_size = gl + (c_len if alloc else 0)

            pos = 0
            next_op = 0
            ip_ov = ip_val
            divide_returned = False

            for _m in range(MAX_MICRO):
                at_valid = (pos < instr_len) and (not divide_returned)
                if not at_valid:
                    break

                is_opcode = (pos == next_op)
                if not is_opcode:
                    # non-opcode position: nothing happens, position unchanged ->
                    # this cannot make progress, so stop (matches JAX no-op+revert)
                    break

                cur_val = instr_table[i, instr_idx, _clip(pos, 0, MAX_MICRO - 1)]
                opcode = _clip(cur_val, 0, UP_IS_SIZE - 1)
                n_ops = N_OPERANDS[opcode]

                ops_start = pos + 1
                remaining = instr_len - ops_start
                if remaining < 0:
                    remaining = 0

                # --- read up to 3 operands (from instruction, overflow to tape) ---
                raw0 = int32(0)
                raw1 = int32(0)
                raw2 = int32(0)
                cur_ip = ip_ov
                if n_ops >= 1:
                    if 0 < remaining:
                        raw0 = instr_table[i, instr_idx, _clip(ops_start + 0, 0, MAX_MICRO - 1)]
                    else:
                        raw0 = _tape_read(G, C, i, cur_ip, tape_size, gl, c_len)
                        cur_ip = cur_ip + 1
                if n_ops >= 2:
                    if 1 < remaining:
                        raw1 = instr_table[i, instr_idx, _clip(ops_start + 1, 0, MAX_MICRO - 1)]
                    else:
                        raw1 = _tape_read(G, C, i, cur_ip, tape_size, gl, c_len)
                        cur_ip = cur_ip + 1
                if n_ops >= 3:
                    if 2 < remaining:
                        raw2 = instr_table[i, instr_idx, _clip(ops_start + 2, 0, MAX_MICRO - 1)]
                    else:
                        raw2 = _tape_read(G, C, i, cur_ip, tape_size, gl, c_len)
                        cur_ip = cur_ip + 1
                ip_after = cur_ip

                # normalize operands to SE indices
                safe_nses = nses if nses > 1 else 1
                a0 = raw0 if raw0 >= 0 else -raw0
                a1 = raw1 if raw1 >= 0 else -raw1
                a2 = raw2 if raw2 >= 0 else -raw2
                o0 = (a0 % SHORT_MAX) % safe_nses
                o1 = (a1 % SHORT_MAX) % safe_nses
                o2 = (a2 % SHORT_MAX) % safe_nses

                # --- execute opcode (scalar dispatch; only the taken branch runs) ---
                if opcode == 3:                       # LOAD
                    addr = S[i, _clip(o0, 0, MAX_SE - 1)]
                    S[i, _clip(o1, 0, MAX_SE - 1)] = _tape_read(G, C, i, addr, tape_size, gl, c_len)
                elif opcode == 4:                     # STORE
                    addr = S[i, _clip(o0, 0, MAX_SE - 1)]
                    val = S[i, _clip(o1, 0, MAX_SE - 1)]
                    _tape_write(G, C, CC, i, addr, val, tape_size, gl, c_len, alloc)
                elif opcode == 5:                     # MOVE
                    S[i, _clip(o1, 0, MAX_SE - 1)] = S[i, _clip(o0, 0, MAX_SE - 1)]
                elif opcode == 6:                     # ALLOCATE
                    alloc_size = S[i, _clip(o0, 0, MAX_SE - 1)]
                    lo_t = int32(MIN_ALLOC_R * gl)
                    hi_t = int32(MAX_ALLOC_R * gl)
                    if (not alloc) and (alloc_size > lo_t) and (alloc_size < hi_t):
                        alloc = True
                        c_len = alloc_size
                        for k in range(MAX_GENOME):
                            C[i, k] = BLANK
                            CC[i, k] = 0
                elif opcode == 7:                     # COMPARE
                    a = S[i, _clip(o0, 0, MAX_SE - 1)]
                    b = S[i, _clip(o1, 0, MAX_SE - 1)]
                    r = int32(-1) if a < b else (int32(0) if a == b else int32(1))
                    S[i, _clip(o2, 0, MAX_SE - 1)] = r
                elif opcode == 8:                     # IFZERO: skip next if SE[o0] != 0
                    if S[i, _clip(o0, 0, MAX_SE - 1)] != 0:
                        S[i, 0] = S[i, 0] + 1
                elif opcode == 9:                     # JUMP
                    S[i, 0] = S[i, _clip(o0, 0, MAX_SE - 1)]
                elif opcode == 10:                    # DEC
                    S[i, _clip(o0, 0, MAX_SE - 1)] = S[i, _clip(o0, 0, MAX_SE - 1)] - 1
                elif opcode == 11:                    # INC
                    S[i, _clip(o0, 0, MAX_SE - 1)] = S[i, _clip(o0, 0, MAX_SE - 1)] + 1
                elif opcode == 12:                    # DIVIDE
                    n_copied = int32(0)
                    for k in range(MAX_GENOME):
                        if CC[i, k] != 0:
                            n_copied += 1
                    prolif = alloc and (n_copied > int32(MIN_PROLIF_R * gl))
                    if not prolif:
                        S[i, 0] = S[i, 0] + 1
                    else:
                        exec_inst = cntr
                        haschild = True
                    divide_returned = True
                elif opcode == 17:                    # ADD
                    S[i, _clip(o2, 0, MAX_SE - 1)] = S[i, _clip(o0, 0, MAX_SE - 1)] + S[i, _clip(o1, 0, MAX_SE - 1)]
                elif opcode == 18:                    # SUB
                    S[i, _clip(o2, 0, MAX_SE - 1)] = S[i, _clip(o0, 0, MAX_SE - 1)] - S[i, _clip(o1, 0, MAX_SE - 1)]
                elif opcode == 19:                    # MUL
                    S[i, _clip(o2, 0, MAX_SE - 1)] = S[i, _clip(o0, 0, MAX_SE - 1)] * S[i, _clip(o1, 0, MAX_SE - 1)]
                elif opcode == 20:                    # DIV (floor; no-op if divisor 0)
                    d = S[i, _clip(o1, 0, MAX_SE - 1)]
                    if d != 0:
                        S[i, _clip(o2, 0, MAX_SE - 1)] = S[i, _clip(o0, 0, MAX_SE - 1)] // d
                elif opcode == 21:                    # MOD (floor; no-op if divisor 0)
                    d = S[i, _clip(o1, 0, MAX_SE - 1)]
                    if d != 0:
                        S[i, _clip(o2, 0, MAX_SE - 1)] = S[i, _clip(o0, 0, MAX_SE - 1)] % d
                elif opcode == 22:                    # AND
                    S[i, _clip(o2, 0, MAX_SE - 1)] = S[i, _clip(o0, 0, MAX_SE - 1)] & S[i, _clip(o1, 0, MAX_SE - 1)]
                elif opcode == 23:                    # OR
                    S[i, _clip(o2, 0, MAX_SE - 1)] = S[i, _clip(o0, 0, MAX_SE - 1)] | S[i, _clip(o1, 0, MAX_SE - 1)]
                elif opcode == 24:                    # XOR
                    S[i, _clip(o2, 0, MAX_SE - 1)] = S[i, _clip(o0, 0, MAX_SE - 1)] ^ S[i, _clip(o1, 0, MAX_SE - 1)]
                elif opcode == 25:                    # NEG
                    S[i, _clip(o1, 0, MAX_SE - 1)] = -S[i, _clip(o0, 0, MAX_SE - 1)]
                elif opcode == 26:                    # NOT
                    S[i, _clip(o1, 0, MAX_SE - 1)] = ~S[i, _clip(o0, 0, MAX_SE - 1)]
                elif opcode == 27:                    # SHIFT_L
                    S[i, _clip(o1, 0, MAX_SE - 1)] = S[i, _clip(o0, 0, MAX_SE - 1)] << 1
                elif opcode == 28:                    # SHIFT_R (arithmetic)
                    S[i, _clip(o1, 0, MAX_SE - 1)] = S[i, _clip(o0, 0, MAX_SE - 1)] >> 1
                elif opcode == 37:                    # CLEAR
                    S[i, _clip(o0, 0, MAX_SE - 1)] = 0
                elif opcode == 38:                    # CINC (circular)
                    m = tape_size if tape_size > 1 else 1
                    S[i, _clip(o0, 0, MAX_SE - 1)] = (S[i, _clip(o0, 0, MAX_SE - 1)] + 1) % m
                elif opcode == 39:                    # CDEC (circular)
                    m = tape_size if tape_size > 1 else 1
                    v = S[i, _clip(o0, 0, MAX_SE - 1)] - 1
                    S[i, _clip(o0, 0, MAX_SE - 1)] = (m - 1) if v < 0 else v
                elif opcode == 40:                    # IS_SEP
                    S[i, _clip(o1, 0, MAX_SE - 1)] = 1 if S[i, _clip(o0, 0, MAX_SE - 1)] == SEP else 0
                elif opcode == 41:                    # REL_LOAD
                    addr = S[i, _clip(o0, 0, MAX_SE - 1)] + S[i, _clip(o1, 0, MAX_SE - 1)]
                    S[i, _clip(o2, 0, MAX_SE - 1)] = _tape_read(G, C, i, addr, tape_size, gl, c_len)
                elif opcode == 42:                    # REL_STORE
                    addr = S[i, _clip(o0, 0, MAX_SE - 1)] + S[i, _clip(o1, 0, MAX_SE - 1)]
                    _tape_write(G, C, CC, i, addr, S[i, _clip(o2, 0, MAX_SE - 1)], tape_size, gl, c_len, alloc)
                elif opcode == 43:                    # IFNOTZERO: skip next if SE[o0] == 0
                    if S[i, _clip(o0, 0, MAX_SE - 1)] == 0:
                        S[i, 0] = S[i, 0] + 1
                # else: NOP / unimplemented opcodes do nothing

                # advance instruction position, counter, and overflow IP
                consumed = n_ops if n_ops < remaining else remaining
                pos = pos + 1 + consumed
                next_op = pos
                cntr = cntr + 1
                ip_ov = ip_after

            # --- end micro-op loop: finalize IP for this compound instruction ---
            if divide_returned:
                new_ip = S[i, 0]
            else:
                new_ip = S[i, 0] + 1

            do_divide_success = haschild and (not entry_haschild)
            if do_divide_success:
                gl_safe = gl if gl > 0 else 1
                new_ip = (sep + 1) % gl_safe
                cntr = 0
            S[i, 0] = new_ip

        # write back per-thread scalars
        child_len[i] = c_len
        already_alloc[i] = 1 if alloc else 0
        has_child[i] = 1 if haschild else 0
        counter[i] = cntr
        executed_inst[i] = exec_inst

    return vm_kernel


class VMRunner:
    """Drives the VM kernel over a JAX population, in place, with zero host copies.

    JAX arrays expose __cuda_array_interface__, so numba.cuda.as_cuda_array gives
    the kernel a zero-copy view of each buffer that it can read and write. The
    kernel touches only the is_slow rows, exactly reproducing the JAX model's
    `where(is_slow, vm.update(pop), pop)`. Because numba and JAX use separate CUDA
    streams, we block_until_ready before launch and cuda.synchronize() after so the
    two never race on the same buffers.
    """

    # Agent fields the kernel mutates in place (must be the only ones it writes).
    MUTATED = (
        "se_values", "genome", "child", "child_copied", "executed",
        "already_allocated", "child_len", "has_child", "counter",
        "executed_instructions",
    )
    # Agent fields the kernel reads but never writes.
    READONLY = (
        "alive", "genome_len", "n_ses", "n_instructions", "separator_pos",
        "instruction_table", "instruction_lengths",
    )

    def __init__(self, cfg, threads=128):
        self.cfg = cfg
        self.threads = threads
        self.kernel = build_vm_kernel(cfg)
        self._n_ops = cuda.to_device(_N_OPERANDS)  # 44 int32, allocated once

    def run(self, pop, is_slow):
        """Execute one VM update (steps_per_update compound instructions) on the
        slow-track agents of `pop`, mutating pop's arrays in place. Returns pop."""
        pop_size = is_slow.shape[0]

        # Ensure all buffers the kernel will touch are fully computed by XLA before
        # numba reads/writes them on its own stream.
        jax.block_until_ready(
            [is_slow] + [getattr(pop, f) for f in self.MUTATED + self.READONLY]
        )

        v = cuda.as_cuda_array
        blocks = (pop_size + self.threads - 1) // self.threads
        self.kernel[blocks, self.threads](
            v(is_slow), v(pop.alive),
            v(pop.genome_len), v(pop.n_ses), v(pop.n_instructions), v(pop.separator_pos),
            v(pop.instruction_table), v(pop.instruction_lengths), self._n_ops,
            v(pop.se_values), v(pop.genome), v(pop.child), v(pop.child_copied),
            v(pop.executed), v(pop.already_allocated), v(pop.child_len),
            v(pop.has_child), v(pop.counter), v(pop.executed_instructions),
        )
        cuda.synchronize()
        # pop's arrays were mutated in place; the same Agent now holds updated state.
        return pop
