# Physax: A JAX implementation of Physis



#### Installation

To install the project and set up the environment using [`uv`](https://docs.astral.sh/uv/), run:

```bash
uv venv
source .venv/bin/activate
uv sync
```

This will create a virtual environment, activate it, and install all dependencies (including PyTorch with the `cu121` setup).


### Reproducing the main experiment

The experiment behind `docs/experiment_summary.tex` is a set of independent seeds,
each a 200k-cycle run on the `128×128` grid (`pop_size 16384`) seeded with 50
`arche.replicator` founders. Launch one process per seed (`--seed` can be any integer,
e.g. 0):

```bash
python -m physax --pop_size 16384 --initial_pop 50 --total_cycles 200000 --log_interval 50 \
    --seed 62 --wandb --track_lineage --no-caching --max_micro_ops 32 --snapshot_interval 1000 --kernel

python -m physax --pop_size 16384 --initial_pop 50 --total_cycles 200000 --log_interval 50 \
    --seed 63 --wandb --track_lineage --no-caching --max_micro_ops 32 --snapshot_interval 1000 --kernel
```

Each run preallocates a nearly-full GPU, so give each seed its own device
(`CUDA_VISIBLE_DEVICES=<gpu>`). Flag notes:
- `--no-caching` **and** `--max_micro_ops 32` keep self-replicators alive; caching-on
  and/or the default `max_micro_ops 16` cause a die-off after ~40k cycles.
- `--snapshot_interval 1000` dumps population snapshots to
  `<run>/lineage/snapshot_<cycle>.npz` (needed for the figures).
- `--kernel` runs the bitwise-identical Numba CUDA VM backend (~34× faster, ~1.7 h/seed);
  drop it for the pure-JAX path.

Runs land in `output/run_200000_cycles_seed_<seed>_<timestamp>/`.

### Producing the figures and the report

`docs/make_figures.py` picks up the newest run per seed under `output/`, recomputes the
per-snapshot metrics (cached in `<run>/figure_cache.pkl`), and writes PDFs into
`docs/figures/`. Run on CPU to keep the GPUs free:

```bash
JAX_PLATFORMS=cpu python docs/make_figures.py
cd docs && pdflatex experiment_summary.tex
```
