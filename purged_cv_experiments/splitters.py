"""Cross-validation splitters: naive, blocked, purged (Lopez de Prado), walk-forward.

All splitters operate on ``n_samples`` consecutive sample times ``0..n-1`` and
return a list of ``(train_idx, test_idx)`` integer arrays.  Purging is defined
on per-sample *information intervals* ``[label_start[i], label_end[i]]`` (for
forward h-period labels: ``[i, i+h]``): two samples *overlap* iff their
intervals intersect.  Implemented from scratch; correctness (zero residual
train/test overlap after purging) is proven exhaustively in the test suite.

Splitters
---------
``naive_kfold``     shuffled k-fold (the i.i.d. textbook procedure -- wrong here)
``blocked_kfold``   contiguous k-fold WITHOUT purge (overlap survives at the
                    2 boundaries of every test block)
``purged_kfold``    contiguous k-fold + purge + one-sided embargo per Lopez de
                    Prado (2018, Advances in Financial Machine Learning, ch. 7);
                    ``symmetric_embargo=True`` additionally embargoes the
                    *left* boundary (non-standard; quantified in the paper)
``walk_forward``    expanding-window splits; ``purge=True`` drops the training
                    tail whose label windows cross into the test block
"""

from __future__ import annotations

import numpy as np

Split = tuple[np.ndarray, np.ndarray]


# --------------------------------------------------------------------------- #
# overlap primitives
# --------------------------------------------------------------------------- #
def intervals_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    """Closed-interval intersection test."""
    return start_a <= end_b and start_b <= end_a


def count_overlap_pairs(
    splits: list[Split], label_start: np.ndarray, label_end: np.ndarray
) -> int:
    """Number of (train, test) sample pairs whose label windows overlap,
    summed over folds.  Zero for a correctly purged splitter.  Vectorized but
    exact: checks every pair."""
    total = 0
    for train_idx, test_idx in splits:
        ts = label_start[test_idx][None, :]   # (1, n_test)
        te = label_end[test_idx][None, :]
        rs = label_start[train_idx][:, None]  # (n_train, 1)
        re = label_end[train_idx][:, None]
        total += int(np.sum((rs <= te) & (ts <= re)))
    return total


def _test_blocks(n_samples: int, n_folds: int) -> list[np.ndarray]:
    """Contiguous, equal-as-possible test blocks covering 0..n-1."""
    return np.array_split(np.arange(n_samples), n_folds)


# --------------------------------------------------------------------------- #
# splitters
# --------------------------------------------------------------------------- #
def naive_kfold(
    n_samples: int, n_folds: int, rng: np.random.Generator
) -> list[Split]:
    """Shuffled k-fold: random partition into k test folds, train = rest."""
    perm = rng.permutation(n_samples)
    splits: list[Split] = []
    for chunk in np.array_split(perm, n_folds):
        test_idx = np.sort(chunk)
        mask = np.ones(n_samples, dtype=bool)
        mask[test_idx] = False
        splits.append((np.flatnonzero(mask), test_idx))
    return splits


def blocked_kfold(n_samples: int, n_folds: int) -> list[Split]:
    """Contiguous k-fold without purging: train = everything outside the block."""
    splits: list[Split] = []
    for block in _test_blocks(n_samples, n_folds):
        mask = np.ones(n_samples, dtype=bool)
        mask[block] = False
        splits.append((np.flatnonzero(mask), block))
    return splits


def purged_kfold(
    n_samples: int,
    n_folds: int,
    label_start: np.ndarray,
    label_end: np.ndarray,
    embargo_frac: float = 0.0,
    symmetric_embargo: bool = False,
) -> list[Split]:
    """Purged k-fold with embargo (Lopez de Prado 2018, ch. 7).

    For each contiguous test block:

    * **Purge**: drop every training candidate whose label window
      ``[label_start[i], label_end[i]]`` intersects the union of the test
      samples' label windows.
    * **Embargo**: additionally drop candidates *after* the test block whose
      time index lies within ``label_end[test].max() + embargo_n`` where
      ``embargo_n = round(embargo_frac * n_samples)`` -- i.e. the embargo
      extends the right-side purge window, the mlfinlab/AFML convention.
    * ``symmetric_embargo=True`` mirrors the embargo on the left boundary
      (drops candidates whose label window ends within ``embargo_n`` of the
      test windows' start).  Off by default; LdP's embargo is one-sided.
    """
    label_start = np.asarray(label_start)
    label_end = np.asarray(label_end)
    embargo_n = int(round(embargo_frac * n_samples))
    splits: list[Split] = []
    for block in _test_blocks(n_samples, n_folds):
        t_lab_start = int(label_start[block].min())
        t_lab_end = int(label_end[block].max())
        cand_mask = np.ones(n_samples, dtype=bool)
        cand_mask[block] = False
        cand = np.flatnonzero(cand_mask)

        keep = ~((label_start[cand] <= t_lab_end) & (label_end[cand] >= t_lab_start))
        if embargo_n > 0:
            after = cand > block[-1]
            keep &= ~(after & (cand <= t_lab_end + embargo_n))
            if symmetric_embargo:
                before = cand < block[0]
                keep &= ~(before & (label_end[cand] >= t_lab_start - embargo_n))
        splits.append((cand[keep], block))
    return splits


def walk_forward(
    n_samples: int,
    n_folds: int,
    min_train_frac: float = 0.4,
    purge: bool = False,
    label_end: np.ndarray | None = None,
) -> list[Split]:
    """Expanding-window walk-forward: train on ``[0, test_start)``, test on the
    next block, slide forward.  The first ``min_train_frac`` of the samples is
    never tested.

    With ``purge=False`` (the textbook version, as in the source draft) the
    training tail's label windows run into the test block -- a boundary leak.
    With ``purge=True`` training samples with ``label_end >= test_start`` are
    dropped (the draft's ``WalkForwardPurgedCV``).
    """
    first_test = int(round(min_train_frac * n_samples))
    blocks = np.array_split(np.arange(first_test, n_samples), n_folds)
    splits: list[Split] = []
    for block in blocks:
        train_idx = np.arange(0, block[0])
        if purge:
            if label_end is None:
                raise ValueError("purge=True requires label_end")
            train_idx = train_idx[np.asarray(label_end)[train_idx] < block[0]]
        splits.append((train_idx, block))
    return splits


# --------------------------------------------------------------------------- #
# the standard suite compared in the paper
# --------------------------------------------------------------------------- #
def splitter_suite(
    n_samples: int,
    n_folds: int,
    label_start: np.ndarray,
    label_end: np.ndarray,
    rng: np.random.Generator,
    embargo_frac: float,
    wf_min_train_frac: float,
) -> dict[str, list[Split]]:
    """All compared schemes, keyed by the names used in records and figures."""
    return {
        "naive_kfold": naive_kfold(n_samples, n_folds, rng),
        "blocked_kfold": blocked_kfold(n_samples, n_folds),
        "purged_kfold": purged_kfold(
            n_samples, n_folds, label_start, label_end, embargo_frac=0.0),
        "purged_embargo": purged_kfold(
            n_samples, n_folds, label_start, label_end, embargo_frac=embargo_frac),
        "walk_forward": walk_forward(
            n_samples, n_folds, min_train_frac=wf_min_train_frac, purge=False),
        "walk_forward_purged": walk_forward(
            n_samples, n_folds, min_train_frac=wf_min_train_frac,
            purge=True, label_end=label_end),
    }
