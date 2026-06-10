"""One experiment = one realized series evaluated by every CV scheme, end to end.

For each (splitter, model) pair we record

* the **CV-estimated skill**: pooled out-of-sample IC / strategy Sharpe (and
  AUC for classification) of the cross-validated predictions, exactly as a
  practitioner would compute it;
* the **true skill** of the same fitted models: each fold's model is evaluated
  on the forward holdout continuation of the series (which no CV scheme ever
  touches, separated by an ``h``-step gap so no label window crosses over);
* **leakage := CV estimate - truth** for each metric;
* the cost side: fraction of available training candidates retained, and the
  across-fold dispersion of the fold-level CV estimate.

Batches of these records (null grid, embargo sweep, model-selection runs) are
the raw material for the paper's analysis.  Everything is deterministic given
the seeds: each experiment gets an independently spawned ``SeedSequence``
child; sklearn estimators receive integer seeds derived from it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .model import DGPConfig, Dataset, generate, population_ic
from .splitters import Split, purged_kfold, splitter_suite

REGRESSION_MODELS: tuple[str, ...] = ("ridge", "rf")
CLASSIFICATION_MODELS: tuple[str, ...] = ("logit", "rf_clf")


# --------------------------------------------------------------------------- #
# evaluation config: every knob recorded, nothing hidden
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EvalConfig:
    """All evaluation-side constants (models, folds, embargo default)."""

    n_folds: int = 5
    embargo_frac: float = 0.02       # default embargo of the "purged_embargo" scheme
    wf_min_train_frac: float = 0.4   # walk-forward: first fraction never tested
    ridge_alpha: float = 1.0
    rf_n_estimators: int = 30
    rf_max_depth: int = 6
    rf_min_samples_leaf: int = 5
    rf_n_jobs: int = 1               # joblib per-fit overhead dominates if > 1
    knn_default_k: int = 20
    logit_c: float = 1.0
    logit_max_iter: int = 200
    min_train_size: int = 20         # folds with fewer training rows are skipped


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def make_model(name: str, eval_cfg: EvalConfig, seed: int):
    """Estimator factory.  Names: ``ridge``, ``ridge_a{alpha}``, ``rf``,
    ``rf_d{depth}``, ``knn{k}``, ``logit``, ``rf_clf``."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.neighbors import KNeighborsRegressor

    if name == "ridge":
        return Ridge(alpha=eval_cfg.ridge_alpha)
    if name.startswith("ridge_a"):
        return Ridge(alpha=float(name.removeprefix("ridge_a")))
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=eval_cfg.rf_n_estimators,
            max_depth=eval_cfg.rf_max_depth,
            min_samples_leaf=eval_cfg.rf_min_samples_leaf,
            random_state=seed, n_jobs=eval_cfg.rf_n_jobs)
    if name.startswith("rf_d"):
        return RandomForestRegressor(
            n_estimators=eval_cfg.rf_n_estimators,
            max_depth=int(name.removeprefix("rf_d")),
            min_samples_leaf=eval_cfg.rf_min_samples_leaf,
            random_state=seed, n_jobs=eval_cfg.rf_n_jobs)
    if name.startswith("knn"):
        return KNeighborsRegressor(n_neighbors=int(name.removeprefix("knn")))
    if name == "logit":
        return LogisticRegression(C=eval_cfg.logit_c, max_iter=eval_cfg.logit_max_iter)
    if name == "rf_clf":
        return RandomForestClassifier(
            n_estimators=eval_cfg.rf_n_estimators,
            max_depth=eval_cfg.rf_max_depth,
            min_samples_leaf=eval_cfg.rf_min_samples_leaf,
            random_state=seed, n_jobs=eval_cfg.rf_n_jobs)
    raise ValueError(f"unknown model name: {name}")


def is_classifier_name(name: str) -> bool:
    return name in ("logit", "rf_clf")


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def ic(pred: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation of predictions with realized labels (the IC)."""
    if len(pred) < 3 or np.std(pred) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(pred, y)[0, 1])


def strategy_sharpe(pred: np.ndarray, y: np.ndarray, center: float = 0.0) -> float:
    """Per-observation Sharpe of the sign strategy ``pnl_t = sign(pred_t - c) y_t``.

    Unannualized; with overlapping labels consecutive pnl terms are serially
    correlated, so this is a descriptive secondary metric (IC is primary)."""
    if len(pred) < 3:
        return 0.0
    pnl = np.sign(pred - center) * y
    sd = float(np.std(pnl))
    return float(np.mean(pnl) / sd) if sd > 1e-12 else 0.0


def auc(pred: np.ndarray, y_bin: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y_bin)) < 2:
        return float("nan")
    return float(roc_auc_score(y_bin, pred))


# --------------------------------------------------------------------------- #
# core: evaluate one (splits, model) pair on one dataset
# --------------------------------------------------------------------------- #
def evaluate_scheme(
    data: Dataset,
    splits: list[Split],
    model_name: str,
    eval_cfg: EvalConfig,
    model_seed_root: int,
) -> dict:
    """Fit per fold, pool OOS predictions, and score CV estimate vs truth."""
    classify = is_classifier_name(model_name)
    ins = data.insample_idx
    X_in, y_in = data.X[ins], data.y[ins]
    t_in = data.y_class[ins] if classify else y_in
    X_ho = data.X[data.holdout_idx]
    y_ho = data.y[data.holdout_idx]
    t_ho = data.y_class[data.holdout_idx] if classify else y_ho

    pooled_pred: list[np.ndarray] = []
    pooled_truth: list[np.ndarray] = []
    fold_cv: list[float] = []
    true_ics: list[float] = []
    true_sharpes: list[float] = []
    true_aucs: list[float] = []
    retained: list[float] = []
    n_folds_eff = 0

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        n_avail = len(ins) - len(test_idx)
        retained.append(len(train_idx) / n_avail if n_avail > 0 else 0.0)
        if len(train_idx) < eval_cfg.min_train_size:
            continue
        if classify and len(np.unique(t_in[train_idx])) < 2:
            continue
        n_folds_eff += 1
        seed = int((model_seed_root + 7919 * fold_i) % (2**31 - 1))
        model = make_model(model_name, eval_cfg, seed)
        model.fit(X_in[train_idx], t_in[train_idx])
        if classify:
            pred_te = model.predict_proba(X_in[test_idx])[:, 1]
            pred_ho = model.predict_proba(X_ho)[:, 1]
            fold_cv.append(auc(pred_te, t_in[test_idx]))
            true_aucs.append(auc(pred_ho, t_ho))
        else:
            pred_te = model.predict(X_in[test_idx])
            pred_ho = model.predict(X_ho)
            fold_cv.append(ic(pred_te, y_in[test_idx]))
            true_ics.append(ic(pred_ho, y_ho))
            true_sharpes.append(strategy_sharpe(pred_ho, y_ho))
        pooled_pred.append(pred_te)
        pooled_truth.append(t_in[test_idx])

    if n_folds_eff == 0:
        nan = float("nan")
        return {"cv_ic": nan, "true_ic": nan, "leak_ic": nan,
                "cv_sharpe": nan, "true_sharpe": nan, "leak_sharpe": nan,
                "cv_auc": nan, "true_auc": nan, "leak_auc": nan,
                "fold_cv_std": nan, "train_frac_retained": float(np.mean(retained)),
                "n_folds_effective": 0, "n_test_total": 0}

    pred_all = np.concatenate(pooled_pred)
    truth_all = np.concatenate(pooled_truth)
    fold_arr = np.asarray(fold_cv, dtype=float)
    out: dict = {
        "fold_cv_std": float(np.nanstd(fold_arr, ddof=1)) if n_folds_eff > 1 else 0.0,
        "train_frac_retained": float(np.mean(retained)),
        "n_folds_effective": n_folds_eff,
        "n_test_total": int(len(pred_all)),
    }
    if classify:
        cv_auc = auc(pred_all, truth_all)
        true_auc = float(np.nanmean(true_aucs))
        out.update({
            "cv_auc": cv_auc, "true_auc": true_auc, "leak_auc": cv_auc - true_auc,
            "cv_ic": float("nan"), "true_ic": float("nan"), "leak_ic": float("nan"),
            "cv_sharpe": float("nan"), "true_sharpe": float("nan"),
            "leak_sharpe": float("nan"),
        })
    else:
        cv_ic = ic(pred_all, truth_all)
        cv_sh = strategy_sharpe(pred_all, truth_all)
        true_ic = float(np.mean(true_ics))
        true_sh = float(np.mean(true_sharpes))
        out.update({
            "cv_ic": cv_ic, "true_ic": true_ic, "leak_ic": cv_ic - true_ic,
            "cv_sharpe": cv_sh, "true_sharpe": true_sh, "leak_sharpe": cv_sh - true_sh,
            "cv_auc": float("nan"), "true_auc": float("nan"), "leak_auc": float("nan"),
        })
    return out


# --------------------------------------------------------------------------- #
# one experiment: all splitters x all models on one realized series
# --------------------------------------------------------------------------- #
def _seed_root(ss: np.random.SeedSequence) -> int:
    return int(ss.generate_state(1)[0] % (2**31 - 1))


def run_experiment(
    cfg: DGPConfig,
    eval_cfg: EvalConfig,
    seed_seq: np.random.SeedSequence,
    models: tuple[str, ...] = REGRESSION_MODELS,
    splitter_names: tuple[str, ...] | None = None,
) -> list[dict]:
    """One realized series -> one record per (splitter, model)."""
    data_ss, model_ss = seed_seq.spawn(2)
    rng = np.random.default_rng(data_ss)
    data = generate(cfg, rng)
    n = len(data.insample_idx)
    suite = splitter_suite(
        n, eval_cfg.n_folds, data.label_start, data.label_end, rng,
        embargo_frac=eval_cfg.embargo_frac,
        wf_min_train_frac=eval_cfg.wf_min_train_frac)
    if splitter_names is not None:
        suite = {k: v for k, v in suite.items() if k in splitter_names}
    root = _seed_root(model_ss)
    pop_ic = population_ic(cfg)

    rows: list[dict] = []
    for s_i, (sname, splits) in enumerate(suite.items()):
        for m_i, mname in enumerate(models):
            res = evaluate_scheme(
                data, splits, mname, eval_cfg,
                model_seed_root=root + 1_000_003 * s_i + 101 * m_i)
            rows.append({
                **{f"cfg_{k}": v for k, v in asdict(cfg).items()},
                **{f"eval_{k}": v for k, v in asdict(eval_cfg).items()},
                "splitter": sname,
                "model": mname,
                "population_ic": pop_ic,
                **res,
            })
    return rows


# --------------------------------------------------------------------------- #
# batch 1: the headline null grid -- phantom skill vs (h, feature persistence)
# --------------------------------------------------------------------------- #
def run_null_grid(
    horizons: tuple[int, ...],
    noise_phis: tuple[float, ...],
    n_reps: int,
    *,
    t_obs_choices: tuple[int, ...],
    t_holdout: int,
    n_noise_features: int,
    eval_cfg: EvalConfig,
    seed: int,
    models: tuple[str, ...] = REGRESSION_MODELS,
    progress_every: int = 0,
) -> list[dict]:
    """Noise-only configs (true skill exactly zero) across the (h, phi) grid.

    ``t_obs`` is sampled per repetition from ``t_obs_choices`` (recorded in the
    record's ``cfg_t_obs``).  Any mean CV skill above zero is phantom."""
    ss = np.random.SeedSequence(seed)
    cells = [(h, phi) for h in horizons for phi in noise_phis]
    children = ss.spawn(len(cells) * n_reps)
    rows: list[dict] = []
    done = 0
    for c_i, (h, phi) in enumerate(cells):
        for r in range(n_reps):
            child = children[c_i * n_reps + r]
            pick_rng = np.random.default_rng(child.spawn(1)[0])
            t_obs = int(pick_rng.choice(t_obs_choices))
            cfg = DGPConfig(
                t_obs=t_obs, t_holdout=t_holdout, horizon=h,
                n_noise_features=n_noise_features, noise_phi=phi,
                n_signal_features=0, signal_strength=0.0,
                label=f"null_h{h}_phi{phi}")
            for row in run_experiment(cfg, eval_cfg, child, models=models):
                row["rep"] = r
                rows.append(row)
            done += 1
            if progress_every and done % progress_every == 0:
                print(f"  {done}/{len(children)}", flush=True)
    return rows


# --------------------------------------------------------------------------- #
# batch 2: embargo sweep -- does purging restore null calibration, and what
# embargo does a persistent *genuine* signal require?
# --------------------------------------------------------------------------- #
def run_embargo_sweep(
    signal_phis: tuple[float, ...],
    embargo_fracs: tuple[float, ...],
    n_reps: int,
    *,
    horizon: int,
    t_obs: int,
    t_holdout: int,
    n_noise_features: int,
    noise_phi: float,
    target_population_ic: float,
    include_null_phi: float | None,
    eval_cfg: EvalConfig,
    seed: int,
    models: tuple[str, ...] = REGRESSION_MODELS,
    progress_every: int = 0,
) -> list[dict]:
    """Purged k-fold across an embargo grid, one record per (embargo, model).

    Arms: one genuine-signal config per ``signal_phi`` (strength calibrated to
    ``target_population_ic``), plus optionally a noise-only arm with
    ``noise_phi=include_null_phi`` (the purge-only-suffices control)."""
    from .model import strength_for_population_ic

    arms: list[DGPConfig] = []
    for phi in signal_phis:
        beta = strength_for_population_ic(
            target_population_ic, horizon=horizon, signal_phi=phi)
        arms.append(DGPConfig(
            t_obs=t_obs, t_holdout=t_holdout, horizon=horizon,
            n_noise_features=n_noise_features, noise_phi=noise_phi,
            n_signal_features=1, signal_phi=phi, signal_strength=beta,
            label=f"signal_phi{phi}"))
    if include_null_phi is not None:
        arms.append(DGPConfig(
            t_obs=t_obs, t_holdout=t_holdout, horizon=horizon,
            n_noise_features=n_noise_features, noise_phi=include_null_phi,
            n_signal_features=0, signal_strength=0.0,
            label=f"null_phi{include_null_phi}"))

    ss = np.random.SeedSequence(seed)
    children = ss.spawn(len(arms) * n_reps)
    rows: list[dict] = []
    done = 0
    for a_i, cfg in enumerate(arms):
        for r in range(n_reps):
            child = children[a_i * n_reps + r]
            data_ss, model_ss = child.spawn(2)
            data = generate(cfg, np.random.default_rng(data_ss))
            root = _seed_root(model_ss)
            pop_ic = population_ic(cfg)
            n = len(data.insample_idx)
            for e_i, emb in enumerate(embargo_fracs):
                splits = purged_kfold(
                    n, eval_cfg.n_folds, data.label_start, data.label_end,
                    embargo_frac=emb)
                for m_i, mname in enumerate(models):
                    res = evaluate_scheme(
                        data, splits, mname, eval_cfg,
                        model_seed_root=root + 1_000_003 * e_i + 101 * m_i)
                    rows.append({
                        **{f"cfg_{k}": v for k, v in asdict(cfg).items()},
                        **{f"eval_{k}": v for k, v in asdict(eval_cfg).items()},
                        "splitter": "purged_kfold",
                        "embargo_frac": float(emb),
                        "model": mname,
                        "population_ic": pop_ic,
                        "rep": r,
                        **res,
                    })
            done += 1
            if progress_every and done % progress_every == 0:
                print(f"  {done}/{len(children)}", flush=True)
    return rows


# --------------------------------------------------------------------------- #
# batch 3: model selection -- does purged CV pick truly better models?
# --------------------------------------------------------------------------- #
def run_selection_batch(
    n_reps: int,
    *,
    cfg: DGPConfig,
    eval_cfg: EvalConfig,
    candidates: tuple[str, ...],
    seed: int,
    progress_every: int = 0,
) -> list[dict]:
    """Genuine-signal DGP + a model menu; select the menu's argmax by naive
    shuffled k-fold CV vs purged(+embargo) CV, then score every candidate's
    *true forward skill* (refit on all in-sample data, evaluated on the
    forward holdout).  One record per repetition."""
    from .splitters import naive_kfold

    ss = np.random.SeedSequence(seed)
    children = ss.spawn(n_reps)
    rows: list[dict] = []
    for r, child in enumerate(children):
        data_ss, model_ss = child.spawn(2)
        rng = np.random.default_rng(data_ss)
        data = generate(cfg, rng)
        n = len(data.insample_idx)
        root = _seed_root(model_ss)
        schemes = {
            "naive": naive_kfold(n, eval_cfg.n_folds, rng),
            "purged": purged_kfold(
                n, eval_cfg.n_folds, data.label_start, data.label_end,
                embargo_frac=eval_cfg.embargo_frac),
        }

        # CV score of every candidate under each scheme
        cv_scores: dict[str, dict[str, float]] = {s: {} for s in schemes}
        for s_i, (sname, splits) in enumerate(schemes.items()):
            for m_i, mname in enumerate(candidates):
                res = evaluate_scheme(
                    data, splits, mname, eval_cfg,
                    model_seed_root=root + 1_000_003 * s_i + 101 * m_i)
                cv_scores[sname][mname] = res["cv_ic"]

        # true forward skill of every candidate (refit on all in-sample data)
        ins = data.insample_idx
        X_ho, y_ho = data.X[data.holdout_idx], data.y[data.holdout_idx]
        true_skill: dict[str, float] = {}
        for m_i, mname in enumerate(candidates):
            seed_m = int((root + 31 * m_i) % (2**31 - 1))
            model = make_model(mname, eval_cfg, seed_m)
            model.fit(data.X[ins], data.y[ins])
            true_skill[mname] = ic(model.predict(X_ho), y_ho)

        oracle = max(candidates, key=lambda m: true_skill[m])
        row: dict = {
            **{f"cfg_{k}": v for k, v in asdict(cfg).items()},
            **{f"eval_{k}": v for k, v in asdict(eval_cfg).items()},
            "rep": r,
            "candidates": "|".join(candidates),
            "population_ic": population_ic(cfg),
            "oracle_pick": oracle,
            "oracle_true_ic": true_skill[oracle],
            **{f"true_ic_{m}": true_skill[m] for m in candidates},
        }
        for sname in schemes:
            pick = max(candidates, key=lambda m: cv_scores[sname][m])
            row[f"{sname}_pick"] = pick
            row[f"{sname}_cv_ic"] = cv_scores[sname][pick]
            row[f"{sname}_true_ic"] = true_skill[pick]
            row[f"{sname}_regret"] = true_skill[oracle] - true_skill[pick]
            for m in candidates:
                row[f"{sname}_cv_ic_{m}"] = cv_scores[sname][m]
        rows.append(row)
        if progress_every and (r + 1) % progress_every == 0:
            print(f"  {r + 1}/{n_reps}", flush=True)
    return rows


# --------------------------------------------------------------------------- #
# batch 4: classification check (AUC behaves like IC)
# --------------------------------------------------------------------------- #
def run_classification_check(
    noise_phis: tuple[float, ...],
    n_reps: int,
    *,
    horizon: int,
    t_obs: int,
    t_holdout: int,
    n_noise_features: int,
    eval_cfg: EvalConfig,
    seed: int,
    splitter_names: tuple[str, ...] = ("naive_kfold", "purged_embargo"),
    progress_every: int = 0,
) -> list[dict]:
    """Sign-of-return labels + logistic/RF classifiers on null configs: the
    same phantom-skill pattern, measured in AUC (truth = 0.5)."""
    ss = np.random.SeedSequence(seed)
    children = ss.spawn(len(noise_phis) * n_reps)
    rows: list[dict] = []
    done = 0
    for p_i, phi in enumerate(noise_phis):
        cfg = DGPConfig(
            t_obs=t_obs, t_holdout=t_holdout, horizon=horizon,
            n_noise_features=n_noise_features, noise_phi=phi,
            n_signal_features=0, signal_strength=0.0,
            label=f"null_clf_phi{phi}")
        for r in range(n_reps):
            child = children[p_i * n_reps + r]
            for row in run_experiment(
                    cfg, eval_cfg, child, models=CLASSIFICATION_MODELS,
                    splitter_names=splitter_names):
                row["rep"] = r
                rows.append(row)
            done += 1
            if progress_every and done % progress_every == 0:
                print(f"  {done}/{len(children)}", flush=True)
    return rows
