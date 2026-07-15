"""Post-hoc analysis, evaluation, and visualization for Physis.

Code that consumes a run's outputs (snapshots, edge files, stats) rather than
driving the simulation: GP-map structure (`gp_map`), genome/language statistics
(`genome_stats`), the reference JAX VM and offline genome evaluation
(`virtual_machine`, `genome_evaluator`), plotting (`visualization`), and W&B
logging (`wandb_logger`).

The simulation engine itself lives in the sibling `physax.sim` package.
"""
