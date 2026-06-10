"""Reproduce every number and figure input in the paper.

    python scripts/run_all.py            # full run -> results/results.json + CSVs
    python scripts/run_all.py --quick    # small batch for a smoke check

Deterministic given the fixed seeds below.  No wall clock leaks into the
results files (stage timings are printed to stdout only).  Run from anywhere;
paths are resolved relative to this file.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sklearn

from purged_cv_experiments import __version__
from purged_cv_experiments import analysis as A
from purged_cv_experiments.model import DGPConfig, strength_for_population_ic
from purged_cv_experiments.simulate import (
    EvalConfig,
    run_classification_check,
    run_embargo_sweep,
    run_null_grid,
    run_selection_batch,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

SEEDS = {"null_grid": 101, "embargo": 202, "selection": 303, "classification": 404}
CALIBRATION_ABS_THRESHOLD = 0.01  # |mean leak| below this counts as calibrated


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    q = args.quick

    eval_cfg = EvalConfig()  # all evaluation constants (recorded in every row)

    # batch sizes (the only quick/full difference; everything else is fixed)
    horizons = (1, 5, 10, 20)
    noise_phis = (0.0, 0.3, 0.6, 0.9, 0.97)
    null_reps = 3 if q else 16
    t_obs_choices = (800,) if q else (800, 1600, 2400)
    t_holdout = 2000 if q else 5000

    signal_phis = (0.9, 0.97) if q else (0.6, 0.9, 0.97, 0.995)
    embargo_fracs = (0.0, 0.02, 0.08) if q else (0.0, 0.01, 0.02, 0.04, 0.08)
    embargo_reps = 4 if q else 24
    embargo_t_obs = 800 if q else 1600
    target_pop_ic = 0.10

    sel_reps = 6 if q else 50
    sel_candidates = ("ridge", "rf_d3", "rf_d8", "knn10", "knn50")
    sel_cfg = DGPConfig(
        t_obs=800 if q else 2000, t_holdout=t_holdout, horizon=10,
        n_noise_features=4, noise_phi=0.9,
        n_signal_features=1, signal_phi=0.97,
        signal_strength=strength_for_population_ic(
            target_pop_ic, horizon=10, signal_phi=0.97),
        label="selection")

    clf_phis = (0.0, 0.9)
    clf_reps = 4 if q else 20
    clf_t_obs = 800 if q else 1500

    RESULTS.mkdir(exist_ok=True)
    t_start = time.monotonic()

    def stage(msg: str) -> float:
        print(msg, flush=True)
        return time.monotonic()

    t0 = stage(f"[1/5] null grid: {len(horizons)}h x {len(noise_phis)}phi x "
               f"{null_reps} reps ...")
    rec_null = run_null_grid(
        horizons, noise_phis, null_reps,
        t_obs_choices=t_obs_choices, t_holdout=t_holdout, n_noise_features=5,
        eval_cfg=eval_cfg, seed=SEEDS["null_grid"],
        progress_every=max(1, len(horizons) * len(noise_phis) * null_reps // 4))
    df_null = A.to_frame(rec_null)
    df_null.to_csv(RESULTS / "records_null.csv", index=False)
    print(f"  done in {time.monotonic() - t0:.0f}s", flush=True)

    t0 = stage(f"[2/5] embargo sweep: {len(signal_phis)} signal arms + null, "
               f"{len(embargo_fracs)} embargoes x {embargo_reps} reps ...")
    rec_emb = run_embargo_sweep(
        signal_phis, embargo_fracs, embargo_reps,
        horizon=10, t_obs=embargo_t_obs, t_holdout=t_holdout,
        n_noise_features=4, noise_phi=0.9,
        target_population_ic=target_pop_ic, include_null_phi=0.9,
        eval_cfg=eval_cfg, seed=SEEDS["embargo"],
        progress_every=max(1, (len(signal_phis) + 1) * embargo_reps // 4))
    df_emb = A.to_frame(rec_emb)
    df_emb.to_csv(RESULTS / "records_embargo.csv", index=False)
    print(f"  done in {time.monotonic() - t0:.0f}s", flush=True)

    t0 = stage(f"[3/5] model selection: {sel_reps} reps x "
               f"{len(sel_candidates)} candidates ...")
    rec_sel = run_selection_batch(
        sel_reps, cfg=sel_cfg, eval_cfg=eval_cfg, candidates=sel_candidates,
        seed=SEEDS["selection"], progress_every=max(1, sel_reps // 4))
    df_sel = A.to_frame(rec_sel)
    df_sel.to_csv(RESULTS / "records_selection.csv", index=False)
    print(f"  done in {time.monotonic() - t0:.0f}s", flush=True)

    t0 = stage(f"[4/5] classification check: {len(clf_phis)} phis x {clf_reps} reps ...")
    rec_clf = run_classification_check(
        clf_phis, clf_reps, horizon=10, t_obs=clf_t_obs, t_holdout=t_holdout,
        n_noise_features=5, eval_cfg=eval_cfg, seed=SEEDS["classification"],
        progress_every=max(1, len(clf_phis) * clf_reps // 2))
    df_clf = A.to_frame(rec_clf)
    df_clf.to_csv(RESULTS / "records_classification.csv", index=False)
    print(f"  done in {time.monotonic() - t0:.0f}s", flush=True)

    print("[5/5] summaries ...", flush=True)
    summary = A.summarize(df_null, df_emb, df_sel, df_clf,
                          calibration_abs_threshold=CALIBRATION_ABS_THRESHOLD)
    results = {
        "meta": {
            "package_version": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "quick": bool(q),
            "seeds": SEEDS,
            "eval_config": asdict(eval_cfg),
            "grids": {
                "horizons": list(horizons),
                "noise_phis": list(noise_phis),
                "null_reps": null_reps,
                "t_obs_choices": list(t_obs_choices),
                "t_holdout": t_holdout,
                "signal_phis": list(signal_phis),
                "embargo_fracs": list(embargo_fracs),
                "embargo_reps": embargo_reps,
                "embargo_t_obs": embargo_t_obs,
                "target_population_ic": target_pop_ic,
                "selection_reps": sel_reps,
                "selection_candidates": list(sel_candidates),
                "selection_cfg": asdict(sel_cfg),
                "classification_phis": list(clf_phis),
                "classification_reps": clf_reps,
                "classification_t_obs": clf_t_obs,
            },
            "calibration_abs_threshold": CALIBRATION_ABS_THRESHOLD,
            "notes": "Deterministic; reproduce with python scripts/run_all.py",
        },
        **summary,
    }
    (RESULTS / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\nWrote {RESULTS / 'results.json'} and 4 records CSVs "
          f"(total {time.monotonic() - t_start:.0f}s).")

    print_headlines(summary)


def print_headlines(summary: dict) -> None:
    """Print the paper's headline numbers from a computed summary dict."""
    print("\n--- HEADLINE NUMBERS ---")
    print("\nPhantom skill under the null (CV IC - true IC), worst (h, phi) cell:")
    for r in summary["phantom_null"]["worst_cell_by_splitter"]:
        print(f"  {r['splitter']:20} {r['model']:6} "
              f"{r['worst_cell_leak']:+.3f} +/- {r['ci95']:.3f}  "
              f"(h={r['at_horizon']}, phi={r['at_noise_phi']})")
    inter = summary["phantom_null"]["interaction_naive_rf"]
    print(f"\nInteraction (naive k-fold, RF): max phantom {inter['max_leak']:+.3f}; "
          f"h=1 max {inter['h1_any_phi_max_leak']:+.3f}; "
          f"phi=0 max {inter['phi0_any_h_max_leak']:+.3f}")

    print("\nEmbargo needed to remove residual optimism "
          f"(mean leak <= {CALIBRATION_ABS_THRESHOLD} or CI contains 0):")
    for r in summary["embargo"]["required_embargo"]:
        req = r["required_embargo"]
        req_s = f"{req:.0%}" if np.isfinite(req) else "none in grid"
        print(f"  {r['arm']:18} {r['model']:6} leak@0 embargo "
              f"{r['leak_at_zero_embargo']:+.3f} -> required embargo: {req_s}")

    print("\nCost of purging (per splitter, null grid):")
    for r in summary["costs"]:
        if r["model"] != "rf":
            continue
        print(f"  {r['splitter']:20} train kept {r['train_frac_retained']:.1%}, "
              f"fold-CV std {r['mean_fold_cv_std']:.3f}, "
              f"rep-CV std {r['rep_cv_std']:.3f}")

    sel = summary["selection"]
    print(f"\nModel selection (signal DGP, population IC = {sel['population_ic']:.3f}):")
    print(f"  naive  CV pick: true IC {sel['naive']['true_ic_of_pick']['mean']:.3f}, "
          f"CV claims {sel['naive']['cv_ic_of_pick']['mean']:.3f} "
          f"(optimism {sel['naive']['optimism_of_pick']['mean']:+.3f})")
    print(f"  purged CV pick: true IC {sel['purged']['true_ic_of_pick']['mean']:.3f}, "
          f"CV claims {sel['purged']['cv_ic_of_pick']['mean']:.3f} "
          f"(optimism {sel['purged']['optimism_of_pick']['mean']:+.3f})")
    print(f"  oracle true IC {sel['oracle_true_ic']['mean']:.3f}; "
          f"purged - naive true IC {sel['purged_minus_naive_true_ic']['mean']:+.3f} "
          f"+/- {sel['purged_minus_naive_true_ic']['ci95']:.3f} "
          f"(purged wins {sel['purged_win_rate']:.0%}, ties {sel['tie_rate']:.0%})")

    print("\nClassification check (AUC - true AUC, null DGP):")
    for r in summary["classification"]:
        print(f"  {r['splitter']:16} {r['model']:7} phi={r['noise_phi']:.2f}  "
              f"leak {r['mean_leak_auc']:+.3f} +/- {r['ci95']:.3f} "
              f"(true AUC {r['mean_true_auc']:.3f})")


if __name__ == "__main__":
    main()
