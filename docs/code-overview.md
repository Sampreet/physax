# Physax — Code Overview

Physax is a [JAX](https://github.com/google/jax) reimplementation of **Physis** (a.k.a. ARCHE),
the Tierra-like digital-evolution system described in the accompanying paper:

> Egri-Nagy & Nehaniv (2003), *Evolvability of the Genotype–Phenotype Relation in Populations of
> Self-Replicating Digital Organisms in a Tierra-like System.*

This document explains what the code does and how it maps onto the ideas in the paper. It is written
for someone (possibly future-you) coming back to the project after a break.

---

## 1. The idea, in one paragraph

In classic Tierra/Avida, digital organisms are programs that self-replicate. The **processor** that
runs them — its registers and its instruction set — is fixed, designed by a human. The paper's key
move is to **put the description of the processor *and* its instruction set inside the genome
itself**. An organism's life-cycle therefore has two phases: first it *builds its own processor* from
the structural part of its genome, then it *executes the rest of its genome* on that freshly built
processor. Because the processor and instruction set are now part of the heritable genome, they too
are subject to mutation and selection — the **genotype→phenotype mapping is itself evolvable**. This
is what the paper means by a "universal processor" (analogous to a universal Turing machine).

Physax runs this model for a whole **population in parallel** on GPU: organisms are `vmap`-ed across
the population and simulation cycles run inside a `lax.scan`, with fixed-size padded arrays so
everything stays JIT-compatible.

---

## 2. Genome layout

A genome is a 1-D array of integers (padded to `max_genome_len = 256` with `BLANK = -1`). It has two
sections divided by a single `SEP` (separator) token:

```
[  structure + instruction definitions  ] SEP [        program        ]
            (the "processor")                         (the "code")
```

- **Structure part** (before `SEP`): defines the organism's processor.
  - `R` / `S` / `Q` markers each declare a **structural element (SE)** — a register, stack, or queue.
    (In this implementation all three are treated as registers.) `SE[0]` is always the implicit
    **instruction pointer (IP)**; each `R`/`S`/`Q` adds one more SE. `B` is a filler/blank marker.
  - `I` markers each begin one **instruction definition**. The genes between an `I` and the next
    `I` (or `SEP`) are the micro-ops making up that instruction: an opcode followed by its operands.
- **Program part** (after `SEP`): the actual executed code — a sequence of **indices into the
  instruction table** built from the definitions above.

Parsing happens in [`agent.py`](../physax/agent.py):
- `_build_structure` scans up to `SEP`, counts SEs, and records the separator position.
- `_build_instruction_set` scans the `I` markers before `SEP` and compiles each definition into a
  normalized micro-op row of the `instruction_table` (opcodes normalized via `abs(v) % 44`, operand
  counts looked up from `N_OPERANDS`).

The **ancestor genome** (`Agent.create_ancestor_genome`) is a hand-written 78-gene self-replicator
(`arche.replicator`): it declares a few registers, defines ~14 instructions, and its program allocates
a child tape, copies itself gene-by-gene into it, and calls `DIVIDE`.

---

## 3. The instruction set (opcodes)

Opcodes and their operand counts are defined in [`config.py`](../physax/config.py) (`OP_NAMES`,
`N_OPERANDS`). The executable bodies live in `get_opcode_functions`, which returns 44 pure
`(state, args) -> state` functions dispatched by `jax.lax.switch`. Categories:

| Group | Opcodes |
|---|---|
| Control / flow | `NOP`, `JUMP`, `IFZERO`, `IFNOTZERO`, `COMPARE` |
| Registers ↔ tape | `LOAD`, `STORE`, `MOVE`, `REL_LOAD`, `REL_STORE`, `CLEAR` |
| Arithmetic | `INC`, `DEC`, `ADD`, `SUB`, `MUL`, `DIV_OP`, `MOD`, `NEG` |
| Bitwise | `AND`, `OR`, `XOR`, `NOT`, `SHIFT_L`, `SHIFT_R` |
| Tape addressing | `CINC`, `CDEC` (circular inc/dec), `IS_SEP` |
| Replication | `ALLOCATE`, `DIVIDE` |

Operands index into the SE array (`abs(v) % 32767 % n_ses`), so they always refer to a valid
structural element. Opcodes listed in the paper but not implemented here (`IN`, `OUT`, `SDIR`,
`GDIR`, `SEND`, `RECEIVE`, `FORK_TH`, `KILL_TH`) are aliased to `NOP`. Division-by-zero and other
degenerate cases are made safe rather than crashing — matching the paper's "fault tolerance"
requirement that no instruction can crash the processor.

---

## 4. Execution model

One organism is stepped by the **`VirtualMachine`** in [`virtual_machine.py`](../physax/virtual_machine.py):

- `update` runs `steps_per_update` (= 34) compound instructions via `lax.scan`, but only for organisms
  that are `alive` and have not just produced a child.
- `execute_one` performs a single compound instruction:
  1. **Fetch** the gene at `IP` (`SE[0]`) from the program and map it to an instruction:
     `instr_idx = abs(gene) % n_instructions`.
  2. **Run its micro-ops** in an inner `lax.scan` over `max_micro_ops` (= 32) slots. For each opcode
     it reads the required operands — from the instruction body if present, otherwise it "overflows"
     into the tape, advancing the IP (the `fillOperands` logic). It then dispatches the opcode with
     `lax.switch`.
  3. **Advance IP** by 1, unless the instruction jumped or the organism just divided.
- The **`OpState` / `OpArgs`** split (in `config.py`) separates the mutable per-op state (SE values,
  child tape, allocation flags, …) from the read-only per-op arguments (keys, sizes, decoded
  operands). This split gave the ~30% speedup noted in the git history.

Every tape position that the IP visits is recorded in `executed`, which later feeds fitness/merit
computation.

---

## 5. Replication, mutation, and the population loop

The **`Model`** in [`model.py`](../physax/model.py) ties everything together and runs the
population-level simulation.

**Replication.** An organism reproduces by:
1. `ALLOCATE` — reserve a child tape whose size is within `[0.5, 2.0] × genome_len`.
2. Copy its genome into the child (via `LOAD` / `REL_STORE`), each copied gene marked in `child_copied`.
3. `DIVIDE` — succeeds once at least `min_proliferation_ratio` (= 0.80) of the genome has been copied.

**Mutation** happens at two points:
- **Copy mutation** (`copy_mutation_rate` = 0.009): during copying, a gene may be replaced by a random
  one (in `tape_write`, `config.py`).
- **Divide mutations** (`Agent.apply_divide_mutations`): after a successful divide, the child tape may
  undergo a point substitution, an insertion, or a deletion (`divide_insert_rate` /
  `divide_delete_rate` = 0.0013). Insertions/deletions change genome length — this is how the paper's
  variable-length structural/instruction-set evolution actually occurs.

**Cycle** (`cycle_step`), run once per simulation tick over the whole population:
1. **Execute** every alive organism (`vmap` over the VM's `update`).
2. **Age** everyone.
3. **Birth**: for organisms that divided, build the mutated child genome and re-parse it into a fresh
   `Agent` (so the child's processor/instruction set is re-derived from its possibly-mutated genome).
4. **Placement**: children are placed on a 2-D **toroidal grid** using the **OldestNurse** rule —
   each parent picks the emptiest/oldest of its 8 neighbours to overwrite (`_place_children_on_grid`).
   This is the ecological selection pressure: slow replicators get overwritten and vanish.
5. **Cleanup**: reset the transient child/allocation fields.

Colours (HSV) are inherited with small mutations so lineages are visually trackable.

`run_simulation` drives many cycles in JIT-compiled chunks, prints/logs stats (population size,
births, genome-length percentiles), optionally logs to Weights & Biases, and collects per-chunk
snapshots.

---

## 6. Analysis & visualization

- [`analysis.py`](../physax/analysis.py) — `compute_snapshot_properties` computes, per organism,
  **effective length** (executed genes), **merit** (= effective length, no external tasks), and
  **fitness** (= merit / gestation time). Fitness is only nonzero for organisms that actually
  reproduced.
- [`visualization.py`](../physax/visualization.py) —
  - `plot_metrics`: population size + births + genome-length percentiles over time.
  - `save_grid_gif`: an animation of the 2-D population grid coloured by lineage.
  - `save_physis_view_gif`: a richer per-organism view.

---

## 7. Key parameters (`Config` in `config.py`)

| Parameter | Default | Meaning |
|---|---|---|
| `max_genome_len` | 256 | Padded genome/tape length |
| `pop_size` | 1024 (4096 in `__main__`) | Grid capacity |
| `initial_pop` | 1 (10 in `__main__`) | Seed organisms |
| `steps_per_update` | 34 | Compound instructions executed per organism per cycle |
| `max_instructions` / `max_micro_ops` | 64 / 32 | Instruction-table dimensions |
| `max_se_count` | 16 | Max structural elements (registers/stacks/queues) |
| `copy_mutation_rate` | 0.009 | Per-gene copy error |
| `divide_insert_rate` / `divide_delete_rate` | 0.0013 | Length-changing divide mutations |
| `min_proliferation_ratio` | 0.80 | Copy fraction required to divide |
| `min/max_allocation_ratio` | 0.5 / 2.0 | Allowed child-tape size range |

---

## 8. Running it

```bash
uv venv && source .venv/bin/activate && uv sync
python -m physax          # runs the simulation defined in physax/__main__.py
```

Edit `physax/__main__.py` (or pass overrides through `make_config`) to change `pop_size`,
`initial_pop`, `total_cycles`, and `log_interval`. It produces `simulation_metrics.png` and
`evolution.gif`.

There is also a standalone script, [`ancestor_full_division_illustration.py`](../ancestor_full_division_illustration.py),
which traces a single ancestor organism through one complete replication for debugging/illustration,
and a self-contained monolithic port in the top-level [`__main__.py`](../__main__.py) (the code was
later split into the `physax/` package for readability).

---

## 9. File map

| File | Role |
|---|---|
| `physax/config.py` | Opcode constants, `N_OPERANDS`, `Config`, `OpState`/`OpArgs`, opcode function bodies |
| `physax/agent.py` | `Agent` state, genome parsing (structure + instruction set), ancestor genome, divide mutations |
| `physax/virtual_machine.py` | Single-organism execution (fetch → micro-op scan → dispatch) |
| `physax/model.py` | Population init, per-cycle loop, birth/placement/mutation, run loop, logging |
| `physax/analysis.py` | Fitness / merit / effective-length computation |
| `physax/visualization.py` | Metric plots and grid/organism GIFs |
| `physax/__main__.py` | Entry point wiring config → model → plots |
```
