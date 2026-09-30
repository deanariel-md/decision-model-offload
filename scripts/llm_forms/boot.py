"""Paired bootstrap helpers for scripts/llm_forms/analyze.py.

- B, score, boot_diff: as scripts/summaries/hybrid_all.py (whose SEED, 20260927, is not used here; SEED below).
- stratified_index: as scripts/summaries/pool_summary.py, dropin() (the resample index stratified by report set).
- workflow_pool240_index: as scripts/summaries/workflow_groups.py, pool240() (its own per-set generator), used only to
  reproduce the primary pairing as a check.
"""
from __future__ import annotations

import numpy as np

B = 2000                   # hybrid_all.B
SEED = 20260929            # the seed of the 240-report pool
SETS = ["main", "heldout", "misleading_feature", "nonregional_sites"]


def score(ok, strat):
    """ok: bool array over items; strat: None or bool array (balanced accuracy over two strata). (hybrid_all.score)"""
    if strat is None:
        return ok.mean(axis=-1)
    return (np.where(strat, ok, 0).sum(-1) / strat.sum(-1) + np.where(~strat, ok, 0).sum(-1) / (~strat).sum(-1)) / 2


def boot_diff(a, b, strat, idx):
    """Mean of a minus b under resampled item index rows idx (shape B x n). (hybrid_all.boot_diff)"""
    s = None if strat is None else strat[idx]
    return score(a[idx], s) - score(b[idx], s)


def stratified_index(sets: np.ndarray, seed: int = SEED, n_boot: int = B, order=SETS) -> np.ndarray:
    """B x n resample index over items in their given order, stratified by set (pool_summary.dropin): for each set in
    `order`, that set's positions resampled with replacement to the set's size, the sets' columns side by side."""
    sets = np.asarray(sets)
    rng = np.random.default_rng(seed)
    return np.concatenate([np.flatnonzero(sets == k)[rng.integers(0, (sets == k).sum(), size=(n_boot, (sets == k).sum()))]
                           for k in order if (sets == k).any()], axis=1)


def workflow_pool240_index(sizes: dict, seed: int = 20260927, n_boot: int = B, order=SETS) -> dict:
    """workflow_groups.pool240's resamples: one generator, for each set in order an n_boot x n_k index into that set's
    pool reports (sorted by report id)."""
    rng = np.random.default_rng(seed)
    return {k: rng.integers(0, sizes[k], size=(n_boot, sizes[k])) for k in order if sizes.get(k)}


def diff_ci(a: np.ndarray, b: np.ndarray, idx: np.ndarray) -> dict:
    """Paired difference of accuracies a - b in points: full-sample estimate and 95% percentile interval."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    d = boot_diff(a, b, None, idx)
    return {"est_pts": float(100 * (score(a, None) - score(b, None))),
            "ci_pts": [float(100 * np.quantile(d, 0.025)), float(100 * np.quantile(d, 0.975))]}
