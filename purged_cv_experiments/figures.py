"""Generate the paper's figures (vector PDF) from the saved results.

    python -m purged_cv_experiments.figures   # writes paper/figures/*.pdf

Reads results/results.json and results/records_*.csv; the setup diagram is
re-generated deterministically from the splitter code itself (not hand-drawn).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .model import DGPConfig, generate
from .splitters import naive_kfold, purged_kfold

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGDIR = ROOT / "paper" / "figures"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9,
    "axes.labelsize": 9, "figure.dpi": 120, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})
C_NAIVE, C_BLOCK, C_PURGE, C_EMB, C_WF = (
    "#c0392b", "#e0a458", "#1f3b73", "#2e8b57", "#9aa6b2")
SPLITTER_STYLE = {
    "naive_kfold": (C_NAIVE, "o", "naive shuffled k-fold"),
    "blocked_kfold": (C_BLOCK, "s", "blocked k-fold (no purge)"),
    "walk_forward": (C_WF, "v", "walk-forward (no purge)"),
    "purged_kfold": (C_PURGE, "^", "purged k-fold (no embargo)"),
    "purged_embargo": (C_EMB, "D", "purged k-fold + embargo"),
    "walk_forward_purged": ("#7a4f9e", "P", "walk-forward (purged)"),
}


# --------------------------------------------------------------------------- #
# Fig 1: setup -- overlapping label windows and the purge/embargo anatomy
# --------------------------------------------------------------------------- #
def fig_setup(path: Path, *, seed: int = 7) -> None:
    h = 5
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 2.9))

    # (a) overlapping label windows under a naive shuffled assignment
    ax = axes[0]
    n_a = 16
    rng = np.random.default_rng(seed)
    splits = naive_kfold(n_a, 4, rng)
    train_idx, test_idx = splits[0]
    test_set = set(test_idx.tolist())
    leak_train = {
        int(i) for i in train_idx
        for j in test_idx if (i <= j + h) and (j <= i + h)}
    for t in range(n_a):
        if t in test_set:
            color, z = C_NAIVE, 3
        elif t in leak_train:
            color, z = "#f2b3ab", 2
        else:
            color, z = "#c9d2dc", 1
        ax.broken_barh([(t, h)], (t - 0.35, 0.7), facecolors=color, zorder=z,
                       edgecolor="white", linewidth=0.4)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in
               (C_NAIVE, "#f2b3ab", "#c9d2dc")]
    ax.legend(handles, ["test label window", "train, overlaps test", "train, clean"],
              fontsize=6.5, loc="upper left")
    ax.set_title(f"(a) shuffled k-fold, $h={h}$: train windows\noverlap test windows")
    ax.set_xlabel("time"); ax.set_ylabel("sample $t$ (label $[t,t+h]$)")
    ax.set_ylim(-1, n_a)

    # (b) purged k-fold anatomy for one middle fold (computed, not drawn)
    ax = axes[1]
    n_b, k, emb = 200, 5, 0.05
    ls = np.arange(n_b); le = ls + h
    splits_e0 = purged_kfold(n_b, k, ls, le, embargo_frac=0.0)
    splits_e = purged_kfold(n_b, k, ls, le, embargo_frac=emb)
    fold = 2
    tr0, te = splits_e0[fold]
    tre, _ = splits_e[fold]
    purged = np.setdiff1d(np.setdiff1d(np.arange(n_b), tr0), te)
    embargoed = np.setdiff1d(tr0, tre)
    for idxs, color, label in [
            (tre, "#c9d2dc", "train (kept)"), (te, C_NAIVE, "test"),
            (np.setdiff1d(purged, te), C_PURGE, "purged"),
            (embargoed, C_EMB, "embargo")]:
        for t in idxs:
            ax.axvspan(t, t + 1, color=color, lw=0)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in
               ("#c9d2dc", C_PURGE, C_NAIVE, C_EMB)]
    ax.legend(handles, ["train (kept)", "purged", "test", "embargo"],
              fontsize=6.5, loc="upper right")
    ax.set_yticks([])
    ax.set_title(f"(b) purged k-fold anatomy, fold {fold + 1}/{k}\n"
                 f"($h={h}$, embargo {emb * 100:.0f}% of samples)")
    ax.set_xlabel("time")

    # (c) the leakage mechanism: persistent feature + shared label content
    ax = axes[2]
    cfg = DGPConfig(t_obs=120, t_holdout=50, horizon=20,
                    n_noise_features=1, noise_phi=0.97, label="diagram")
    data = generate(cfg, np.random.default_rng(seed))
    x = data.X[:120, 0]
    t_test, t_train = 60, 61
    ax.plot(np.arange(120), x, color=C_PURGE, lw=1.0, label=r"feature $x_t$ (AR(1), $\phi=0.97$)")
    ax.scatter([t_test], [x[t_test]], color=C_NAIVE, s=35, zorder=5, label="test sample")
    ax.scatter([t_train], [x[t_train]], color="#e0a458", s=35, zorder=5,
               marker="s", label="train neighbor")
    ymin = x.min() - 0.6
    ax.plot([t_test, t_test + cfg.horizon], [ymin, ymin], color=C_NAIVE, lw=3,
            solid_capstyle="butt")
    ax.plot([t_train, t_train + cfg.horizon], [ymin - 0.35, ymin - 0.35],
            color="#e0a458", lw=3, solid_capstyle="butt")
    ax.annotate("label windows share\n$h-1$ of $h$ returns",
                xy=(t_test + cfg.horizon / 2, ymin - 0.15),
                xytext=(t_test + 32, ymin + 1.1), fontsize=6.5,
                arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_title("(c) mechanism: near-identical features,\nnear-identical labels")
    ax.set_xlabel("time"); ax.set_ylabel("feature value")
    ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig 2: phantom skill vs horizon by splitter (null DGP)
# --------------------------------------------------------------------------- #
def fig_phantom(path: Path, results: dict) -> None:
    cells = pd.DataFrame(results["phantom_null"]["by_cell"])
    phi_max = float(cells["noise_phi"].max())
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.0))

    for ax, model, title in [(axes[0], "rf", "(a) random forest"),
                             (axes[1], "ridge", "(b) ridge")]:
        sub = cells[(cells.model == model) & (cells.noise_phi == phi_max)]
        for name, (color, marker, label) in SPLITTER_STYLE.items():
            g = sub[sub.splitter == name].sort_values("horizon")
            if not len(g):
                continue
            ax.errorbar(g["horizon"], g["mean_leak"], yerr=g["ci95"],
                        color=color, marker=marker, ms=4, lw=1.3, capsize=2,
                        label=label)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{title}, feature persistence $\\phi={phi_max}$")
        ax.set_xlabel("label horizon $h$")
        ax.set_ylabel("phantom skill (CV IC $-$ true IC)")
        if model == "rf":
            ax.legend(fontsize=6.2, loc="upper left")

    # (c) the interaction: naive k-fold RF phantom needs BOTH h>1 and phi>0
    ax = axes[2]
    sub = cells[(cells.model == "rf") & (cells.splitter == "naive_kfold")]
    phis = sorted(sub["noise_phi"].unique())
    cmap = plt.cm.viridis
    for i, phi in enumerate(phis):
        g = sub[sub.noise_phi == phi].sort_values("horizon")
        ax.errorbar(g["horizon"], g["mean_leak"], yerr=g["ci95"],
                    color=cmap(i / max(len(phis) - 1, 1)), marker="o", ms=3.5,
                    lw=1.2, capsize=2, label=f"$\\phi={phi}$")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("(c) naive k-fold (RF): overlap $\\times$\npersistence interaction")
    ax.set_xlabel("label horizon $h$")
    ax.set_ylabel("phantom skill")
    ax.legend(fontsize=6.2, loc="upper left", title="feature persistence",
              title_fontsize=6.2)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig 3: embargo sweep vs signal persistence + the cost side
# --------------------------------------------------------------------------- #
def fig_embargo(path: Path, results: dict) -> None:
    table = pd.DataFrame(results["embargo"]["by_embargo"])
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.0))
    arms = table.sort_values(["is_null", "signal_phi"])
    arm_names = list(dict.fromkeys(arms["arm"]))
    cmap = plt.cm.plasma

    def arm_label(arm_df: pd.DataFrame) -> str:
        if arm_df["is_null"].iloc[0] == 1:
            return f"null (noise $\\phi={arm_df['signal_phi'].iloc[0]:.2f}$ n/a)"
        return f"signal $\\phi_s={arm_df['signal_phi'].iloc[0]}$"

    for ax, model, title in [(axes[0], "rf", "(a) random forest"),
                             (axes[1], "ridge", "(b) ridge")]:
        for i, arm in enumerate(arm_names):
            g = table[(table.arm == arm) & (table.model == model)].sort_values(
                "embargo_frac")
            if not len(g):
                continue
            is_null = g["is_null"].iloc[0] == 1
            color = "#9aa6b2" if is_null else cmap(0.15 + 0.7 * i / max(len(arm_names) - 1, 1))
            label = "null (no signal)" if is_null else f"signal $\\phi_s={g['signal_phi'].iloc[0]}$"
            ax.errorbar(g["embargo_frac"] * 100, g["mean_leak"], yerr=g["ci95"],
                        marker="o", ms=3.5, lw=1.3, capsize=2, color=color,
                        ls="--" if is_null else "-", label=label)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{title}: residual CV optimism\nafter purging")
        ax.set_xlabel("embargo (% of samples)")
        ax.set_ylabel("CV IC $-$ true IC")
        if model == "rf":
            ax.legend(fontsize=6.2, loc="upper right")

    # (c) cost: training data retained as embargo grows
    ax = axes[2]
    g = table[table.model == "rf"].groupby("embargo_frac")[
        "train_frac_retained"].mean().reset_index()
    ax.plot(g["embargo_frac"] * 100, g["train_frac_retained"] * 100, "o-",
            color=C_PURGE, lw=1.4, ms=4)
    ax.set_title("(c) cost: training candidates retained\n(purge + embargo)")
    ax.set_xlabel("embargo (% of samples)")
    ax.set_ylabel("% of available training samples kept")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig 4: model-selection quality -- naive vs purged CV picks vs oracle
# --------------------------------------------------------------------------- #
def fig_selection(path: Path, results: dict, df_sel: pd.DataFrame) -> None:
    sel = results["selection"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.3))

    # (a) true forward IC of the picked model, with CI
    ax = axes[0]
    entries = [
        ("naive CV pick", sel["naive"]["true_ic_of_pick"], C_NAIVE),
        ("purged CV pick", sel["purged"]["true_ic_of_pick"], C_PURGE),
        ("oracle", sel["oracle_true_ic"], C_EMB),
    ]
    x = np.arange(len(entries))
    for xi, (label, stat, color) in zip(x, entries):
        ax.bar(xi, stat["mean"], yerr=stat["ci95"], color=color, width=0.6,
               capsize=3)
        ax.text(xi, stat["mean"] + stat["ci95"] + 0.002, f"{stat['mean']:.3f}",
                ha="center", fontsize=7)
    ax.axhline(sel["population_ic"], color="k", lw=0.8, ls=":",
               label=f"population IC = {sel['population_ic']:.3f}")
    ax.set_xticks(x, [e[0] for e in entries])
    ax.set_ylabel("true forward IC of selected model")
    ax.set_title("(a) forward-truth quality of the selected model")
    ax.legend(fontsize=7, loc="lower right")

    # (b) which model each scheme picks
    ax = axes[1]
    candidates = sel["candidates"]
    width = 0.38
    xc = np.arange(len(candidates))
    n = sel["n_reps"]
    naive_counts = [sel["naive"]["pick_counts"].get(m, 0) / n for m in candidates]
    purged_counts = [sel["purged"]["pick_counts"].get(m, 0) / n for m in candidates]
    ax.bar(xc - width / 2, naive_counts, width, color=C_NAIVE, label="naive CV")
    ax.bar(xc + width / 2, purged_counts, width, color=C_PURGE, label="purged CV")
    # tick labels carry each candidate's measured true forward skill
    tick_labels = [
        f"{m}\ntrue {sel['per_candidate_true_ic'][m]['mean']:.3f}"
        for m in candidates]
    ax.set_xticks(xc, tick_labels, fontsize=6.5)
    ax.set_ylabel("pick frequency")
    ax.set_title("(b) which candidate gets selected")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    results = json.loads((RESULTS / "results.json").read_text())
    df_sel = pd.read_csv(RESULTS / "records_selection.csv")

    fig_setup(FIGDIR / "fig_setup.pdf")
    fig_phantom(FIGDIR / "fig_phantom.pdf", results)
    fig_embargo(FIGDIR / "fig_embargo.pdf", results)
    fig_selection(FIGDIR / "fig_selection.pdf", results, df_sel)
    print(f"wrote 4 figures to {FIGDIR}")


if __name__ == "__main__":
    main()
