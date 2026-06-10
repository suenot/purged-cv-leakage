"""Turn experiment records into the paper's quantitative results.

Five questions, all answered with measured numbers (no fabrication):

1. **Phantom skill of naive k-fold under the null** as a function of the label
   horizon ``h`` and feature persistence ``phi`` -- the headline surface, with
   normal-approximation 95% CIs.
2. **Blocked-without-purge**: how much boundary leakage survives blocking?
3. **Purged k-fold + embargo**: does purging restore null calibration, and how
   much embargo does a *persistent genuine signal* require before the residual
   CV optimism is statistically indistinguishable from zero?
4. **Costs**: training samples sacrificed by purging/embargo and the
   across-fold dispersion of the CV estimate.
5. **Model selection**: with a real signal and a model menu, does purged CV
   pick truly better models than naive CV (forward-truth comparison)?

All outputs are JSON-able dicts / lists of records.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

Z95 = 1.959963984540054  # two-sided 95% normal quantile


def to_frame(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records)


def mean_ci(series: pd.Series) -> dict:
    """Mean with a normal-approximation 95% CI (and n)."""
    s = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(s))
    if n == 0:
        return {"mean": float("nan"), "ci95": float("nan"), "n": 0}
    sd = float(s.std(ddof=1)) if n > 1 else 0.0
    return {"mean": float(s.mean()), "ci95": Z95 * sd / np.sqrt(n) if n > 1 else 0.0,
            "n": n}


# --------------------------------------------------------------------------- #
# 1 + 2: phantom skill under the null, per (splitter, model, h, phi)
# --------------------------------------------------------------------------- #
def phantom_by_cell(df: pd.DataFrame, metric: str = "leak_ic") -> list[dict]:
    """Mean phantom skill (CV - truth) with CI per (splitter, model, h, phi)."""
    rows: list[dict] = []
    grp = df.groupby(["splitter", "model", "cfg_horizon", "cfg_noise_phi"])
    for (splitter, model, h, phi), g in grp:
        stat = mean_ci(g[metric])
        cv = mean_ci(g["cv_ic"]) if "cv_ic" in g else {"mean": float("nan")}
        rows.append({
            "splitter": splitter, "model": model,
            "horizon": int(h), "noise_phi": float(phi),
            "mean_leak": stat["mean"], "ci95": stat["ci95"], "n": stat["n"],
            "mean_cv": cv["mean"],
            "mean_truth": float(g["true_ic"].mean()) if "true_ic" in g else float("nan"),
            "frac_positive": float((g[metric] > 0).mean()),
        })
    return rows


def phantom_headline(df: pd.DataFrame) -> dict:
    """Compact headline: worst-case and per-splitter phantom skill (null grid)."""
    cells = pd.DataFrame(phantom_by_cell(df))
    out: dict = {"by_cell": cells.to_dict("records")}
    by_splitter = []
    worst = cells.loc[cells.groupby(["splitter", "model"])["mean_leak"].idxmax()]
    for _, r in worst.iterrows():
        by_splitter.append({
            "splitter": r["splitter"], "model": r["model"],
            "worst_cell_leak": float(r["mean_leak"]), "ci95": float(r["ci95"]),
            "at_horizon": int(r["horizon"]), "at_noise_phi": float(r["noise_phi"]),
        })
    out["worst_cell_by_splitter"] = by_splitter

    # interaction checks: phantom requires BOTH overlap (h>1) AND persistence
    naive_rf = cells[(cells.splitter == "naive_kfold") & (cells.model == "rf")]
    out["interaction_naive_rf"] = {
        "h1_any_phi_max_leak": float(naive_rf[naive_rf.horizon == 1]["mean_leak"].max())
        if (naive_rf.horizon == 1).any() else float("nan"),
        "phi0_any_h_max_leak": float(naive_rf[naive_rf.noise_phi == 0.0]["mean_leak"].max())
        if (naive_rf.noise_phi == 0.0).any() else float("nan"),
        "max_leak": float(naive_rf["mean_leak"].max()) if len(naive_rf) else float("nan"),
    }
    return out


def blocked_residual(df: pd.DataFrame) -> list[dict]:
    """Blocked-without-purge vs naive vs purged at every (h, phi) cell (RF)."""
    cells = pd.DataFrame(phantom_by_cell(df))
    rows: list[dict] = []
    for (h, phi), _ in cells.groupby(["horizon", "noise_phi"]):
        rec: dict = {"horizon": int(h), "noise_phi": float(phi)}
        for splitter in ("naive_kfold", "blocked_kfold", "purged_kfold",
                         "purged_embargo", "walk_forward", "walk_forward_purged"):
            sub = cells[(cells.horizon == h) & (cells.noise_phi == phi)
                        & (cells.splitter == splitter) & (cells.model == "rf")]
            if len(sub):
                rec[f"{splitter}_leak"] = float(sub["mean_leak"].iloc[0])
                rec[f"{splitter}_ci95"] = float(sub["ci95"].iloc[0])
        rows.append(rec)
    return rows


# --------------------------------------------------------------------------- #
# 3: embargo sweep
# --------------------------------------------------------------------------- #
def embargo_summary(
    df: pd.DataFrame, *, calibration_abs_threshold: float = 0.01
) -> dict:
    """Mean residual leak per (arm, model, embargo) and the smallest embargo
    that removes residual optimism in each arm.

    "Calibrated" := mean leak ``<= calibration_abs_threshold`` (no optimism;
    pessimism does not count as leakage) OR the mean-leak 95% CI contains
    zero.  The threshold is recorded in the output."""
    rows: list[dict] = []
    grp = df.groupby(["cfg_label", "cfg_signal_phi", "cfg_n_signal_features",
                      "model", "embargo_frac"])
    for (label, sphi, nsig, model, emb), g in grp:
        stat = mean_ci(g["leak_ic"])
        rows.append({
            "arm": label, "signal_phi": float(sphi), "is_null": int(nsig == 0),
            "model": model, "embargo_frac": float(emb),
            "mean_leak": stat["mean"], "ci95": stat["ci95"], "n": stat["n"],
            "mean_cv_ic": float(g["cv_ic"].mean()),
            "mean_true_ic": float(g["true_ic"].mean()),
            "mean_population_ic": float(g["population_ic"].mean()),
            "train_frac_retained": float(g["train_frac_retained"].mean()),
            # "calibrated" = no residual OPTIMISM: mean leak is non-positive,
            # statistically indistinguishable from zero, or below the absolute
            # threshold.  (Pessimism, i.e. negative leak, is reported but does
            # not count as leakage to be embargoed away.)
            "calibrated": int(stat["mean"] <= calibration_abs_threshold
                              or abs(stat["mean"]) <= stat["ci95"]),
        })
    table = pd.DataFrame(rows).sort_values(["arm", "model", "embargo_frac"])

    required: list[dict] = []
    for (arm, model), g in table.groupby(["arm", "model"]):
        g = g.sort_values("embargo_frac")
        ok = g[g["calibrated"] == 1]
        required.append({
            "arm": arm, "model": model,
            "signal_phi": float(g["signal_phi"].iloc[0]),
            "is_null": int(g["is_null"].iloc[0]),
            "required_embargo": float(ok["embargo_frac"].iloc[0]) if len(ok) else float("nan"),
            "leak_at_zero_embargo": float(
                g[g["embargo_frac"] == 0.0]["mean_leak"].iloc[0])
            if (g["embargo_frac"] == 0.0).any() else float("nan"),
        })
    return {
        "calibration_abs_threshold": calibration_abs_threshold,
        "by_embargo": table.to_dict("records"),
        "required_embargo": required,
    }


# --------------------------------------------------------------------------- #
# 4: costs of purging
# --------------------------------------------------------------------------- #
def cost_summary(df_null: pd.DataFrame) -> list[dict]:
    """Training-data loss and CV-estimate dispersion per splitter (null grid).

    ``train_frac_retained`` is each fold's kept fraction of available training
    candidates; ``fold_cv_std`` is the across-fold std of the fold CV metric;
    ``rep_cv_std`` is the across-repetition std of the pooled CV estimate
    inside each (h, phi, model) cell, averaged over cells."""
    rows: list[dict] = []
    for (splitter, model), g in df_null.groupby(["splitter", "model"]):
        per_cell_std = (
            g.groupby(["cfg_horizon", "cfg_noise_phi"])["cv_ic"].std(ddof=1))
        rows.append({
            "splitter": splitter, "model": model,
            "train_frac_retained": float(g["train_frac_retained"].mean()),
            "mean_fold_cv_std": float(g["fold_cv_std"].mean()),
            "rep_cv_std": float(per_cell_std.mean()),
            "n": int(len(g)),
        })
    return rows


# --------------------------------------------------------------------------- #
# 5: model selection
# --------------------------------------------------------------------------- #
def selection_summary(df_sel: pd.DataFrame) -> dict:
    """Forward-truth quality of naive-CV vs purged-CV model picks."""
    candidates = df_sel["candidates"].iloc[0].split("|")
    diff = df_sel["purged_true_ic"] - df_sel["naive_true_ic"]
    out: dict = {
        "n_reps": int(len(df_sel)),
        "candidates": candidates,
        "population_ic": float(df_sel["population_ic"].mean()),
        "oracle_true_ic": mean_ci(df_sel["oracle_true_ic"]),
        "naive": {
            "true_ic_of_pick": mean_ci(df_sel["naive_true_ic"]),
            "cv_ic_of_pick": mean_ci(df_sel["naive_cv_ic"]),
            "regret": mean_ci(df_sel["naive_regret"]),
            "optimism_of_pick": mean_ci(df_sel["naive_cv_ic"] - df_sel["naive_true_ic"]),
            "pick_counts": df_sel["naive_pick"].value_counts().to_dict(),
        },
        "purged": {
            "true_ic_of_pick": mean_ci(df_sel["purged_true_ic"]),
            "cv_ic_of_pick": mean_ci(df_sel["purged_cv_ic"]),
            "regret": mean_ci(df_sel["purged_regret"]),
            "optimism_of_pick": mean_ci(df_sel["purged_cv_ic"] - df_sel["purged_true_ic"]),
            "pick_counts": df_sel["purged_pick"].value_counts().to_dict(),
        },
        "purged_minus_naive_true_ic": mean_ci(diff),
        "purged_win_rate": float((diff > 0).mean()),
        "tie_rate": float((diff == 0).mean()),
        "per_candidate_true_ic": {
            m: mean_ci(df_sel[f"true_ic_{m}"]) for m in candidates},
    }
    return out


# --------------------------------------------------------------------------- #
# classification check
# --------------------------------------------------------------------------- #
def classification_summary(df_clf: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for (splitter, model, phi), g in df_clf.groupby(
            ["splitter", "model", "cfg_noise_phi"]):
        stat = mean_ci(g["leak_auc"])
        rows.append({
            "splitter": splitter, "model": model, "noise_phi": float(phi),
            "mean_leak_auc": stat["mean"], "ci95": stat["ci95"], "n": stat["n"],
            "mean_cv_auc": float(g["cv_auc"].mean()),
            "mean_true_auc": float(g["true_auc"].mean()),
        })
    return rows


# --------------------------------------------------------------------------- #
# top level
# --------------------------------------------------------------------------- #
def summarize(
    df_null: pd.DataFrame,
    df_embargo: pd.DataFrame,
    df_sel: pd.DataFrame,
    df_clf: pd.DataFrame,
    *,
    calibration_abs_threshold: float = 0.01,
) -> dict:
    return {
        "phantom_null": phantom_headline(df_null),
        "blocked_residual": blocked_residual(df_null),
        "embargo": embargo_summary(
            df_embargo, calibration_abs_threshold=calibration_abs_threshold),
        "costs": cost_summary(df_null),
        "selection": selection_summary(df_sel),
        "classification": classification_summary(df_clf),
    }
