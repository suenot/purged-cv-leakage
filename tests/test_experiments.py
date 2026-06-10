"""Sanity tests for the purged-CV leakage experiments.

Run: python -m pytest -q   (from the project root)
"""

from __future__ import annotations

import numpy as np
import pytest

from purged_cv_experiments import (
    DGPConfig,
    EvalConfig,
    blocked_kfold,
    canonical_configs,
    count_overlap_pairs,
    generate,
    naive_kfold,
    population_ic,
    purged_kfold,
    run_experiment,
    strength_for_population_ic,
    walk_forward,
)
from purged_cv_experiments.simulate import evaluate_scheme, run_embargo_sweep


def _windows(n: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(n)
    return t, t + h


# --------------------------------------------------------------------------- #
# purge correctness
# --------------------------------------------------------------------------- #
def test_purged_no_residual_overlap_exhaustive():
    """After purging, NO train/test label-window pair overlaps -- exhaustively
    checked on a small case (every pair, brute force)."""
    n, h, k = 120, 10, 4
    ls, le = _windows(n, h)
    splits = purged_kfold(n, k, ls, le, embargo_frac=0.0)
    for train_idx, test_idx in splits:
        for i in train_idx:
            for j in test_idx:
                assert not (ls[i] <= le[j] and ls[j] <= le[i]), \
                    f"train {i} overlaps test {j}"
    assert count_overlap_pairs(splits, ls, le) == 0


def test_purged_no_overlap_across_horizons_and_embargo():
    """Zero residual overlap for several horizons, fold counts and embargoes."""
    for n, h, k, emb in [(200, 1, 5, 0.0), (200, 5, 5, 0.02), (150, 20, 3, 0.05),
                         (300, 10, 7, 0.0)]:
        ls, le = _windows(n, h)
        splits = purged_kfold(n, k, ls, le, embargo_frac=emb)
        assert count_overlap_pairs(splits, ls, le) == 0


def test_purge_removes_only_overlapping():
    """With zero embargo, every sample dropped by the purge DOES overlap the
    test fold's label windows -- no over-purging."""
    n, h, k = 150, 8, 5
    ls, le = _windows(n, h)
    purged = purged_kfold(n, k, ls, le, embargo_frac=0.0)
    blocked = blocked_kfold(n, k)
    for (tr_p, te), (tr_b, _) in zip(purged, blocked):
        dropped = np.setdiff1d(tr_b, tr_p)
        t_lab_start, t_lab_end = ls[te].min(), le[te].max()
        for i in dropped:
            assert ls[i] <= t_lab_end and le[i] >= t_lab_start, \
                f"sample {i} was purged but does not overlap"
        # and the purge drops at most 2h samples per fold (h each side)
        assert len(dropped) <= 2 * h


def test_naive_and_blocked_have_residual_overlap():
    """The unpurged baselines really do contain overlapping pairs (h > 1)."""
    n, h, k = 200, 10, 5
    ls, le = _windows(n, h)
    rng = np.random.default_rng(0)
    assert count_overlap_pairs(naive_kfold(n, k, rng), ls, le) > 0
    assert count_overlap_pairs(blocked_kfold(n, k), ls, le) > 0
    assert count_overlap_pairs(
        walk_forward(n, k, purge=False), ls, le) > 0
    assert count_overlap_pairs(
        walk_forward(n, k, purge=True, label_end=le), ls, le) == 0


def test_splits_are_disjoint_and_cover():
    """Test folds partition the samples; train and test never intersect."""
    n, h, k = 173, 7, 5  # n not divisible by k on purpose
    ls, le = _windows(n, h)
    rng = np.random.default_rng(1)
    for splits in (naive_kfold(n, k, rng), blocked_kfold(n, k),
                   purged_kfold(n, k, ls, le, embargo_frac=0.03)):
        all_test = np.concatenate([te for _, te in splits])
        assert len(all_test) == n and len(np.unique(all_test)) == n
        for tr, te in splits:
            assert len(np.intersect1d(tr, te)) == 0


def test_embargo_monotonic_train_size_and_placement():
    """Train size is non-increasing in the embargo, and the extra samples the
    embargo removes sit strictly AFTER the test block (LdP one-sided)."""
    n, h, k = 400, 10, 5
    ls, le = _windows(n, h)
    prev_sizes = None
    base = purged_kfold(n, k, ls, le, embargo_frac=0.0)
    for emb in (0.0, 0.01, 0.02, 0.05, 0.1):
        splits = purged_kfold(n, k, ls, le, embargo_frac=emb)
        sizes = [len(tr) for tr, _ in splits]
        if prev_sizes is not None:
            assert all(a <= b for a, b in zip(sizes, prev_sizes))
        prev_sizes = sizes
        for (tr, te), (tr0, _) in zip(splits, base):
            extra = np.setdiff1d(tr0, tr)
            assert np.all(extra > te[-1])


def test_walk_forward_is_causal():
    """Plain walk-forward: every training index precedes every test index."""
    for splits in (walk_forward(300, 5),
                   walk_forward(300, 5, purge=True,
                                label_end=_windows(300, 10)[1])):
        for tr, te in splits:
            assert len(tr) == 0 or tr.max() < te.min()


# --------------------------------------------------------------------------- #
# DGP ground truth
# --------------------------------------------------------------------------- #
def test_labels_are_forward_cumulative_returns():
    """y_t equals the sum of the next h returns, recomputed independently."""
    cfg = DGPConfig(t_obs=300, t_holdout=100, horizon=7,
                    n_noise_features=2, noise_phi=0.5, label="t")
    rng = np.random.default_rng(2)
    # reconstruct from a fresh generation with the same seed path
    data = generate(cfg, np.random.default_rng(2))
    # recompute y from its own definition using csum identity:
    # y_t - y_{t-1} = r_{t+h} - r_t ; instead check additivity across horizons
    cfg1 = DGPConfig(t_obs=300, t_holdout=100, horizon=1,
                     n_noise_features=2, noise_phi=0.5, label="t1")
    data1 = generate(cfg1, np.random.default_rng(2))
    # same rng consumption order => identical return paths for shared prefix
    h = cfg.horizon
    y_rebuilt = np.array([data1.y[t:t + h].sum() for t in range(200)])
    assert np.allclose(y_rebuilt, data.y[:200])


def test_null_dgp_has_zero_true_skill():
    """Noise-only config: population IC is exactly 0 and the empirical
    feature/label correlation on a long series is statistically zero."""
    cfg = DGPConfig(t_obs=60_000, t_holdout=100, horizon=10,
                    n_noise_features=3, noise_phi=0.9, label="null")
    assert population_ic(cfg) == 0.0
    data = generate(cfg, np.random.default_rng(3))
    ins = data.insample_idx
    for col in range(3):
        c = np.corrcoef(data.X[ins, col], data.y[ins])[0, 1]
        assert abs(c) < 0.02


def test_population_ic_matches_empirical():
    """Closed-form population IC agrees with the empirical correlation."""
    beta = strength_for_population_ic(0.12, horizon=10, signal_phi=0.95)
    cfg = DGPConfig(t_obs=200_000, t_holdout=100, horizon=10,
                    n_noise_features=0, n_signal_features=1,
                    signal_phi=0.95, signal_strength=beta, label="sig")
    assert population_ic(cfg) == pytest.approx(0.12, abs=1e-12)
    data = generate(cfg, np.random.default_rng(4))
    ins = data.insample_idx
    emp = np.corrcoef(data.X[ins, 0], data.y[ins])[0, 1]
    assert emp == pytest.approx(0.12, abs=0.015)


def test_holdout_gap_prevents_label_overlap():
    """No in-sample label window reaches into the holdout labels."""
    cfg = DGPConfig(t_obs=500, t_holdout=200, horizon=15,
                    n_noise_features=1, label="gap")
    data = generate(cfg, np.random.default_rng(5))
    max_ins_label_end = data.label_end[data.insample_idx].max()
    min_holdout_label_start = data.label_start[data.holdout_idx].min() + 1
    assert max_ins_label_end < min_holdout_label_start


def test_feature_persistence_matches_phi():
    """Noise features have the configured AR(1) autocorrelation and ~unit var."""
    cfg = DGPConfig(t_obs=100_000, t_holdout=100, horizon=1,
                    n_noise_features=2, noise_phi=0.8, label="ar")
    data = generate(cfg, np.random.default_rng(6))
    x = data.X[data.insample_idx, 0]
    assert np.var(x) == pytest.approx(1.0, abs=0.05)
    ac1 = np.corrcoef(x[:-1], x[1:])[0, 1]
    assert ac1 == pytest.approx(0.8, abs=0.02)


# --------------------------------------------------------------------------- #
# leakage phenomenology on fixed seeds
# --------------------------------------------------------------------------- #
def test_naive_kfold_inflation_reproduces():
    """High persistence + long horizon + RF: naive k-fold reports large
    phantom skill on the null while purged k-fold stays near zero."""
    cfg = DGPConfig(t_obs=1000, t_holdout=3000, horizon=20,
                    n_noise_features=5, noise_phi=0.97, label="inflate")
    rows = run_experiment(cfg, EvalConfig(), np.random.SeedSequence(42),
                          models=("rf",))
    by = {r["splitter"]: r for r in rows}
    assert by["naive_kfold"]["leak_ic"] > 0.10
    assert abs(by["purged_kfold"]["leak_ic"]) < 0.10
    assert by["naive_kfold"]["leak_ic"] > by["purged_kfold"]["leak_ic"] + 0.05
    assert by["naive_kfold"]["leak_ic"] > by["blocked_kfold"]["leak_ic"]


def test_run_experiment_is_deterministic():
    cfg = canonical_configs(t_obs=400, t_holdout=400)["null_persistent"]
    a = run_experiment(cfg, EvalConfig(), np.random.SeedSequence(7))
    b = run_experiment(cfg, EvalConfig(), np.random.SeedSequence(7))
    assert len(a) == len(b)
    for ra, rb in zip(a, b):
        assert ra["splitter"] == rb["splitter"] and ra["model"] == rb["model"]
        assert ra["cv_ic"] == rb["cv_ic"]
        assert ra["true_ic"] == rb["true_ic"]


def test_embargo_monotonicity_sanity():
    """On a persistent-signal arm (fixed seeds), the mean residual leak of
    purged k-fold does not grow when the embargo is enlarged from 0 to 8%."""
    rows = run_embargo_sweep(
        (0.97,), (0.0, 0.08), 6, horizon=10, t_obs=600, t_holdout=2000,
        n_noise_features=2, noise_phi=0.9, target_population_ic=0.1,
        include_null_phi=None, eval_cfg=EvalConfig(), seed=11, models=("rf",))
    import pandas as pd

    df = pd.DataFrame(rows)
    means = df.groupby("embargo_frac")["leak_ic"].mean()
    assert means[0.08] <= means[0.0] + 0.02  # small slack for MC noise


def test_record_keys_and_train_fraction():
    cfg = canonical_configs(t_obs=400, t_holdout=400)["null_iid"]
    rows = run_experiment(cfg, EvalConfig(), np.random.SeedSequence(0))
    assert len(rows) == 6 * 2  # 6 splitters x 2 regression models
    for r in rows:
        for key in ("cv_ic", "true_ic", "leak_ic", "cv_sharpe", "true_sharpe",
                    "train_frac_retained", "fold_cv_std", "population_ic",
                    "cfg_horizon", "cfg_noise_phi", "eval_n_folds"):
            assert key in r
        assert 0.0 < r["train_frac_retained"] <= 1.0
        assert r["population_ic"] == 0.0
    by = {(r["splitter"], r["model"]): r for r in rows}
    # purging must cost training data relative to blocked
    assert (by[("purged_kfold", "rf")]["train_frac_retained"]
            < by[("blocked_kfold", "rf")]["train_frac_retained"])
    assert (by[("purged_embargo", "rf")]["train_frac_retained"]
            < by[("purged_kfold", "rf")]["train_frac_retained"])


def test_classification_null_auc_is_half():
    """Null DGP, purged CV, logistic regression: true AUC ~ 0.5."""
    cfg = DGPConfig(t_obs=800, t_holdout=4000, horizon=10,
                    n_noise_features=4, noise_phi=0.9, label="clf")
    rows = run_experiment(cfg, EvalConfig(), np.random.SeedSequence(3),
                          models=("logit",), splitter_names=("purged_embargo",))
    (row,) = rows
    assert row["true_auc"] == pytest.approx(0.5, abs=0.03)
    assert abs(row["leak_auc"]) < 0.06
