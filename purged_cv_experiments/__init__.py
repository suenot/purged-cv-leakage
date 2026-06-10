"""How much does purging actually fix?  A controlled quantification of
overlapping-label leakage in time-series cross-validation, on synthetic
return series with exactly known (including exactly zero) predictability."""

from .model import (
    DGPConfig,
    Dataset,
    canonical_configs,
    generate,
    population_ic,
    strength_for_population_ic,
)
from .splitters import (
    blocked_kfold,
    count_overlap_pairs,
    intervals_overlap,
    naive_kfold,
    purged_kfold,
    splitter_suite,
    walk_forward,
)
from .simulate import (
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
    EvalConfig,
    auc,
    evaluate_scheme,
    ic,
    make_model,
    run_classification_check,
    run_embargo_sweep,
    run_experiment,
    run_null_grid,
    run_selection_batch,
    strategy_sharpe,
)

__all__ = [
    # model
    "DGPConfig", "Dataset", "generate", "population_ic",
    "strength_for_population_ic", "canonical_configs",
    # splitters
    "naive_kfold", "blocked_kfold", "purged_kfold", "walk_forward",
    "splitter_suite", "count_overlap_pairs", "intervals_overlap",
    # simulate
    "EvalConfig", "make_model", "ic", "strategy_sharpe", "auc",
    "evaluate_scheme", "run_experiment", "run_null_grid",
    "run_embargo_sweep", "run_selection_batch", "run_classification_check",
    "REGRESSION_MODELS", "CLASSIFICATION_MODELS",
]
__version__ = "0.1.0"
