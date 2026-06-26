# Physax: A JAX implementation of Physis



#### Installation

To install the project and set up the environment using [`uv`](https://docs.astral.sh/uv/), run:

```bash
uv venv
source .venv/bin/activate
uv sync
```

This will create a virtual environment, activate it, and install all dependencies (including PyTorch with the `cu121` setup).


### Execute the simulation:

```bash

CUDA_VISIBLE_DEVICES=2 python -m physax --pop_size 65536 --initial_pop 1000 --total_cycles 6000 --log_interval 50
CUDA_VISIBLE_DEVICES=2 python -m physax --pop_size 16384 --initial_pop 50 --total_cycles 6000 --log_interval 50

CUDA_VISIBLE_DEVICES=0 python -m physax --toy

# re-run visualization of the run (pass folder name, base path should be in the .env file):
python -m physax.visualization --folder run_30000_cycles_seed_49_2026-06-15_00-09


# view the list of genomes of a certain status at a given cycle
python show_cycle_genomes.py --cycle 100000 --status SELF_REPLICATING --folder run_100000_cycles_seed_54_2026-06-16_23-08 --top_n 10

# decode genome from folder
python decode_genome_illustration.py --folder run_100000_cycles_seed_54_2026-06-16_23-08 --hash 3427106465219753476

python evaluate_genomes.py --folder run_200000_cycles_seed_54_2026-06-19_17-27 --cycle 190000 --num-genomes 20
```