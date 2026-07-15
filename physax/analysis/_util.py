"""Small shared helpers for the analysis layer (NumPy)."""
import numpy as np


def fold_hash(h):
    """Fold a genome's two int32 hash words into a single int64 key.

    Operates on the last axis: a ``(..., 2)`` array of ``[hi, lo]`` hash words
    becomes a ``(...)`` int64 array (or a scalar for a single ``(2,)`` pair). The
    low word is read as unsigned so the fold is collision-free across the full
    32-bit range.
    """
    h = np.asarray(h)
    hi = h[..., 0].astype(np.int64)
    lo = h[..., 1].astype(np.uint32).astype(np.int64)
    return (hi << 32) | lo


def shannon_entropy(counts):
    """Shannon entropy (nats) of a non-negative count vector."""
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log(p)))
