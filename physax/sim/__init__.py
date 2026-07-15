"""The Physis simulation engine.

Everything needed to *run* the evolutionary simulation: the instruction set
(`isa`), configuration (`config`), the `Agent` organism state, the CUDA VM
(`vm_kernel`), genome classification/routing (`classification`), and the
population loop (`model`).

Post-hoc measurement, evaluation, and visualization of a run's outputs live in
the sibling `physax.analysis` package.
"""
