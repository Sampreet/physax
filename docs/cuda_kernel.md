# The CUDA VM kernel: what it is and what changed

> **Status:** the CUDA kernel is now the **only** slow-track VM backend for the
> simulation, so a CUDA GPU is required to run `python -m physis`. The old
> `--kernel` flag (and the pure-JAX `_vm_jax` sim path it toggled) has been
> removed. The pure-JAX interpreter (`physis.virtual_machine.VirtualMachine`)
> still exists as the reference used for offline genome evaluation and for the
> reproducibility test that validates the kernel. Historical `--kernel` /
> `use_kernel` references below describe the original change.

This note explains the CUDA VM backend: why it exists, how a custom CUDA kernel
works if you've only used JAX before, the one genuinely non-obvious trick that
makes it fast (a zero-copy bridge between JAX and Numba), and exactly which
functions in the codebase changed.

**Result:** at the experiment size (`pop_size=16384`, `--no-caching`) a cycle went
from **~1067 ms → ~31 ms**, about **34×**, i.e. ~59 h → ~1.7 h for a 200k-cycle
run. Results are **bitwise-identical** to the pure-JAX simulation — the kernel is
an optimization, not a change in behavior.

---

## 1. Why the JAX VM was the bottleneck

The simulation spends almost all its time in `VirtualMachine.update`
([physis/virtual_machine.py](../physis/analysis/virtual_machine.py)): a little interpreter
that runs each organism's genome as bytecode — fetch an instruction, decode it,
execute one of 44 opcodes, repeat.

In JAX we ran that interpreter over the whole population with `jax.vmap`. That is
the worst case for the XLA execution model, for two reasons:

1. **`lax.switch` can't branch under `vmap`.** The opcode dispatch is a 44-way
   `lax.switch` (["run opcode number `k`"]). Different organisms execute different
   opcodes, so under `vmap` there is no single `k` — XLA evaluates **all 44 opcode
   implementations for every organism at every micro-op** and then selects the one
   result. That's ~44× wasted compute, and every branch does full-array scatter
   writes into 256-element genome/child tapes, so it's wasted memory traffic too.
2. **Everything is masked, never skipped.** Dead cells, fast-track cells, and
   already-finished cells still run the full interpreter; the result is thrown
   away with a `jnp.where`. There is no early exit.

Measured: the VM phase alone was **~1668 ms per update** at pop 16384.

A CUDA kernel fixes exactly these two things, and the reference project under
`inspiration/cubff/` (Google's "computational life" BFF simulator) does the same.

---

## 2. CUDA kernels for a JAX person

In JAX you write array expressions and XLA decides how to run them on the GPU. A
**CUDA kernel** is the opposite: you write the code for *one* GPU thread, and you
launch millions of copies of it.

- A **thread** runs your function once. You get its index with `cuda.grid(1)`.
- Threads are grouped into **blocks** (we use 128 threads/block); blocks form the
  **grid**. Launching `kernel[num_blocks, threads_per_block](args...)` runs
  `num_blocks × threads_per_block` threads.
- The GPU runs threads in lock-step groups of 32 called **warps** that share one
  instruction scheduler. When threads in a warp take different `if` branches
  ("warp divergence"), the cost is only the *distinct* branches those 32 threads
  take — not all 44 opcodes. This is the whole win over `vmap`'s `lax.switch`, and
  it's worth understanding in detail (below).

Our design is **one thread per organism**: thread `i` interprets organism `i`'s
genome for `steps_per_update` compound instructions with a plain Python-looking
`if/elif` opcode dispatch. Each thread reads and writes only its own row of the
population arrays, so there's no cross-thread coordination.

### Why this beats `vmap`: warp divergence vs. `lax.switch`

A GPU has no independent instruction decoder per thread. The 32 threads in a warp
share **one** program counter and execute **the same instruction together**, each
on its own data — this is **SIMT** (Single Instruction, Multiple Threads). The
interesting question is what happens when they *want* to do different things, i.e.
when our `if opcode == 3: … elif opcode == 4: …` sends them down different paths.

The hardware handles this by **serializing the distinct paths** with an *active
mask*: it picks a branch some thread needs, disables the threads that don't take
it (they idle, their writes suppressed), runs that branch, then flips the mask to
the next path some thread needs, and so on until every path *any* thread wanted has
run. So the dispatch cost for a warp is proportional to the number of **distinct
opcodes actually present** in its 32 threads. If all 32 are running `LOAD`, there's
*zero* divergence — one path, full speed. If they split across `LOAD`/`INC`/`STORE`,
the warp runs three masked passes. Crucially, **the ~41 opcodes no thread is running
cost nothing** — the warp never enters them.

`vmap` + `lax.switch` has no such mechanism. A `vmap`'d program is a single
*data-parallel* computation: the same ops on every lane, with no way for one lane to
skip work another lane does. When the `opcode` fed to `lax.switch` becomes a vector,
XLA can't make it divergent control flow — its only option is to **compute all 44
branch bodies for every organism, then `select` the one each wanted**. So all 44
run, every micro-op, no matter how uniform the population is.

|                         | `vmap` + `lax.switch`                    | CUDA warp                                  |
|-------------------------|-------------------------------------------|--------------------------------------------|
| Branches evaluated      | **All 44**, per organism per micro-op     | Only the **distinct** ones in the 32 threads |
| Unused opcodes          | Computed, then discarded                  | **Skipped entirely (free)**                |
| Best case (all uniform) | Still 44×                                 | **1×** (no divergence)                     |
| Worst case              | 44×                                       | ≤ 32× (a warp has ≤ 32 distinct paths)     |

Even the worst warp is no worse than `vmap`, and the typical case is far better:
warps are consecutive organisms (`i`, `i+1`, …), and once the population sorts into
species/lineages, neighbours tend to run the same opcode at the same step, so few
distinct paths occur per warp.

Two things compound the win beyond "44 vs. few":

1. **Each branch is cheaper.** A `lax.switch` branch does full-array work (e.g. a
   scatter into a 256-element tape for every lane); the kernel's opcode body is a
   few scalar instructions on that one thread's registers.
2. **Real early exit.** A thread whose organism divided or died just `return`s;
   dead / fast-track organisms never enter the loop at all (`if not is_slow[i]:
   return`). `vmap` has no early exit — every lane runs the full
   `steps_per_update × max_micro_ops` bound and masks the surplus away.

We write the kernel in **Numba CUDA** (`from numba import cuda`), which compiles a
subset of Python to GPU code with the `@cuda.jit` decorator — so we stay in Python
and don't maintain a separate C++/`.cu` build.

Two facts made this port tractable:

- **The VM is deterministic.** No opcode consumes randomness (the `step_key`
  threaded through the JAX VM is never actually read). So the kernel needs no RNG,
  and — crucially — we can check it produces the *exact same bytes* as the JAX VM.
- **We only had to port `update`.** Everything else (classification, the genome
  cache, mutation, spatial placement, logging) stays in JAX.

---

## 3. The kernel — [physis/vm_kernel.py](../physis/sim/vm_kernel.py) (new file)

This is the whole new file. Its shape:

- **Module constants** (`BLANK`, `SEP`, `UP_IS_SIZE`, `_N_OPERANDS`, …) mirror
  `config.py` by value — a kernel can't import JAX config objects.
- **Device helper functions** marked `@cuda.jit(device=True)` — these are callable
  only from other GPU code (like `__device__` in CUDA C): `_clip`, `_tape_read`,
  `_tape_write` reproduce the tape addressing of the JAX VM (parent genome for
  positions `< genome_len`, child tape beyond that, with wrap-around).
- **`build_vm_kernel(cfg)`** returns the compiled kernel. It's a factory so the
  fixed sizes (`max_genome_len`, `max_micro_ops`, `steps_per_update`) and the
  allocation thresholds get *baked in as compile-time constants* — that lets Numba
  generate tight loops.
- **`vm_kernel(...)`** is the kernel itself. Per thread `i`:
  1. `if not is_slow[i]: return` — dead / fast-track / non-slow organisms are
     **skipped outright** (the thing `vmap` couldn't do).
  2. Loop `steps_per_update` times: fetch the instruction at the IP, decode it,
     and run **one** opcode via a scalar `if opcode == 3: … elif opcode == 4: …`
     chain — the port of the 44-way dispatch. Operand reads, the micro-op loop,
     `ALLOCATE`, `DIVIDE`, and the IP bookkeeping all mirror
     `virtual_machine.py:execute_one` line for line.
  3. The organism's state (`se_values`, `genome`, `child`, `child_copied`,
     `executed`, and the scalars) is **mutated in place** in the population
     arrays. No 44× fan-out, no masked throwaway work.
- **`VMRunner`** is the host-side driver used by the model (see §4).

Every line of opcode logic has a comment tying it back to the JAX version, because
the two must stay byte-identical.

---

## 4. The clever part: a zero-copy bridge between JAX and Numba

The kernel needs to read and write the population arrays. Those arrays are JAX
arrays living on the GPU. The naïve approach — copy them to the CPU, hand them to
Numba, copy the results back — would move hundreds of MB across the PCIe bus every
cycle and erase the entire speedup.

The trick: **JAX arrays and Numba device arrays can point at the same GPU memory.**
JAX arrays implement the `__cuda_array_interface__` protocol (a standard way for
GPU libraries to describe "here is a pointer to device memory of this shape and
dtype"). Numba's `cuda.as_cuda_array(x)` reads that protocol and hands back a
*view* — no copy. The kernel writes through the view, and because it's the same
memory, the JAX array now holds the updated values.

That's what `VMRunner.run` does ([physis/vm_kernel.py](../physis/sim/vm_kernel.py)):

```python
v = cuda.as_cuda_array                 # JAX array -> Numba view, zero copy
self.kernel[blocks, threads](
    v(is_slow), v(pop.alive),
    v(pop.genome), v(pop.se_values), …  # read + written in place
)
```

Two correctness details:

- **Stream synchronization.** JAX/XLA and Numba use different CUDA "streams"
  (queues of GPU work). Before launching the kernel we call
  `jax.block_until_ready(...)` so XLA has finished producing the arrays; after, we
  call `cuda.synchronize()` so the kernel has finished before JAX reads the
  results. Without these the two runtimes could race on the same memory.
- **In-place mutation of JAX arrays** is normally something you never do, but it's
  safe here because these arrays are freshly produced each cycle and nothing else
  references them. The end-to-end bitwise validation is what proves it.

No host copies, and Numba allocates essentially nothing (it just views JAX's
buffers), so this also sidesteps GPU out-of-memory issues.

---

## 5. Wiring it into the cycle — [physis/model.py](../physis/sim/model.py)

A Numba kernel **cannot run inside `jax.jit` / `lax.scan`.** But the original
`run_simulation` ran many cycles fused inside one big jitted `lax.scan`. So the
integration had to break the cycle open:

- **`cycle_step` was split** into three methods:
  - `_pre_vm` — DB cache lookup + the fast-track step (jittable).
  - `_vm_jax` — the original JAX interpreter VM (still the reference path).
  - `_post_vm` — aging, classification, birth/mutation, placement, stats (jittable).
- The **JAX path** (`cycle_step`) just calls the three in sequence — behavior is
  unchanged, and this path is what the kernel is validated against.
- The **kernel path** (`run_simulation(use_kernel=True)`) drives cycles from a
  plain Python loop (`run_chunk`), calling the jitted halves with the kernel in
  between:

  ```
  pop, is_slow = pre_jit(pop, db)        # JAX (jitted)
  pop          = vm_runner.run(pop, is_slow)   # CUDA kernel, in place
  pop, db, ... = post_jit(pop, db, ...)  # JAX (jitted)
  ```

  We lose the cross-cycle `lax.scan` fusion, but per-cycle Python dispatch is
  negligible next to the GPU work, and each half is still individually jitted.

---

## 6. The second bottleneck: the genome-collection callbacks

Once the VM dropped to ~2 ms, a **new** bottleneck appeared that had been hidden:
two `jax.debug.callback`s in `_post_vm` that copy newly-classified genomes to the
host for the end-of-run archive (`collect_self_replicating` / `collect_fertile`).

These cost **~90–120 ms per cycle** — far more than the actual work. The subtle
reason: `jax.debug.callback` is an **ordered side-effect**. Its mere *presence* in
the jitted graph forces a host-synchronization barrier every cycle, whether or not
it collects anything. So gating it *inside* the graph (with `lax.cond`, or a
periodic flag) doesn't help — the barrier is still there.

The fix (commit `5c73d7c`): on the kernel path, **keep the callbacks out of the
jitted graph entirely** (`post_jit(..., do_collect=False)`) and instead collect the
genomes in **plain Python** inside `run_chunk`, every `--collect_interval` cycles:

```python
pop, db, stats = post_jit(pop, db, cycle_idx, k, is_slow, False)   # no callbacks in graph
if cyc_num % collect_interval == 0 or cyc_num == total_cycles:
    collect_self_replicating(pop.genome_hash, pop.genome,
                             pop.alive & (pop.status == SELF_REPLICATING))
    collect_fertile(pop.genome_hash, pop.genome,
                    pop.alive & (pop.status == FERTILE))
```

`run_chunk` is already ordinary Python (there's no `lax.scan` forbidding a host
transfer there), so the collection is just a normal array read every N cycles. The
archive is a deduplicated, standing-set sample; `pop`/`db` are untouched, so the
simulation trajectory is unchanged. This alone was another **~4.8×** (148 → 31
ms/cycle).

The `_post_vm` callbacks still exist (now `lax.cond`-gated) for the pure-JAX scan
path, which can't collect in Python because it runs inside `lax.scan`.

---

## 7. Exactly what changed

| File | Change |
|---|---|
| **`physis/sim/vm_kernel.py`** | **New file.** The CUDA kernel (`build_vm_kernel`, `vm_kernel`), device helpers (`_clip`, `_tape_read`, `_tape_write`), and the `VMRunner` host driver + zero-copy bridge. |
| **`physis/sim/model.py`** | `cycle_step` split into `_pre_vm` / `_vm_jax` / `_post_vm`. `_post_vm` gained a static `do_collect` flag and `lax.cond`-gated callbacks. `run_simulation` gained `use_kernel` and `collect_interval`; the kernel path adds `run_chunk` (Python-driven loop + Python genome collection). |
| **`physis/__main__.py`** | New flags: `--kernel` (use the CUDA backend) and `--collect_interval N` (how often to archive genomes on the kernel path). |
| **`pyproject.toml`** | Added the `numba-cuda` dependency. |

The pure-JAX path is untouched in behavior and remains the correctness reference.

---

## 8. Using it, and how we know it's correct

The kernel is the default (and only) VM backend, so no flag is needed:

```bash
python -m physis --pop_size 16384 --initial_pop 50 --total_cycles 200000 \
  --log_interval 50 --seed 62 --no-caching --max_micro_ops 32 \
  --track_lineage --snapshot_interval 1000 --wandb \
  --collect_interval 1
```

**Validation.** Because the VM is deterministic, the kernel was checked to be
byte-for-byte identical to the JAX VM: first on a toy population over hundreds of
updates (exercising the `ALLOCATE`/`DIVIDE` paths), then end-to-end — the full
`pop` and the genome DB match the pure-JAX model exactly, cycle by cycle, with
caching both on and off. The reproducibility test still cross-checks kernel
output against `VirtualMachine`, so the kernel affects only speed, not results.

**Performance summary** (pop 16384, `--no-caching`):

| Stage | ms/cycle | vs pure-JAX |
|---|---|---|
| Pure JAX (`vmap` VM) | ~1067 | 1× |
| + CUDA VM kernel | ~127–148 | ~8× |
| + Python genome collection | **~31** | **~34×** |

**Caveats / future work.** The remaining ~29 ms/cycle is now the JAX `_post_vm`
(classification, DB, birth, placement), not the VM. The kernel is memory-bound and
the GPU is under-utilized at pop 16384 — the same kernel would run a much larger
population at similar speed, which is often the better use of the headroom than
raw cycles/sec.
