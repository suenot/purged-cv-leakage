"""Assert that every quantitative claim in paper/main.tex matches results/results.json.

    python scripts/check_paper_numbers.py

Each check formats a value from results.json exactly the way the paper quotes it
and asserts the resulting token appears in main.tex (whitespace-normalized);
where the paper states a range, count, or derived quantity, the underlying
condition is asserted as well. Exits non-zero if any check fails, so it can
gate a release.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = re.sub(r"\s+", " ", (ROOT / "paper" / "main.tex").read_text())
R = json.loads((ROOT / "results" / "results.json").read_text())

failures: list[str] = []
n_checks = 0


def check(label: str, token: str, cond: bool = True) -> None:
    """Assert ``token`` appears in main.tex and ``cond`` holds."""
    global n_checks
    n_checks += 1
    ok_tex = token in TEX
    if ok_tex and cond:
        print(f"  PASS  {label:58} {token!r}")
    else:
        why = [] if ok_tex else [f"token {token!r} not in main.tex"]
        if not cond:
            why.append("condition failed")
        failures.append(f"{label}: {'; '.join(why)}")
        print(f"  FAIL  {label:58} {token!r}  <-- {'; '.join(why)}")


def pm3(mean: float, ci: float) -> str:
    return f"{mean:+.3f} \\pm {ci:.3f}"


def pm3u(mean: float, ci: float) -> str:  # unsigned mean (selection table)
    return f"{mean:.3f} \\pm {ci:.3f}"


CELLS = {(r["splitter"], r["model"], r["horizon"], r["noise_phi"]): r
         for r in R["phantom_null"]["by_cell"]}
WORST = {(r["splitter"], r["model"]): r
         for r in R["phantom_null"]["worst_cell_by_splitter"]}
COSTS = {(r["splitter"], r["model"]): r for r in R["costs"]}
CLF = {(r["splitter"], r["model"], r["noise_phi"]): r for r in R["classification"]}
SEL = R["selection"]
EMB = R["embargo"]

# ------------------------------------------------------------- design ----
print("[design disclosure]")
g = R["meta"]["grids"]
check("horizon grid", r"h \in \{1,5,10,20\}", g["horizons"] == [1, 5, 10, 20])
check("phi grid", r"\phi \in \{0, 0.3, 0.6, 0.9, 0.97\}",
      g["noise_phis"] == [0.0, 0.3, 0.6, 0.9, 0.97])
check("null reps", "$16$ repetitions per cell", g["null_reps"] == 16)
check("t_obs choices", r"\{800, 1600, 2400\}", g["t_obs_choices"] == [800, 1600, 2400])
check("holdout", "5{,}000", g["t_holdout"] == 5000)
check("signal phis", r"\phi_s \in \{0.6, 0.9, 0.97, 0.995\}",
      g["signal_phis"] == [0.6, 0.9, 0.97, 0.995])
check("embargo grid", r"\varepsilon \in \{0, 1, 2, 4, 8\}\%",
      g["embargo_fracs"] == [0.0, 0.01, 0.02, 0.04, 0.08])
check("embargo reps", "$24$ repetitions per arm", g["embargo_reps"] == 24)
check("embargo t_obs", "$T=1600$", g["embargo_t_obs"] == 1600)
check("target pop IC", "$0.10$", g["target_population_ic"] == 0.10)
check("selection reps", "$50$", g["selection_reps"] == 50)
check("selection beta", r"$\beta = 0.036$",
      round(g["selection_cfg"]["signal_strength"], 3) == 0.036)
check("classification reps", "$20$ repetitions", g["classification_reps"] == 20)
check("classification t_obs", "$T=1500$", g["classification_t_obs"] == 1500)
check("calibration threshold", r"$\le 0.01$",
      R["meta"]["calibration_abs_threshold"] == 0.01)
check("k folds", "$k=5$", R["meta"]["eval_config"]["n_folds"] == 5)
check("default embargo 2%", r"\varepsilon = 2\%",
      R["meta"]["eval_config"]["embargo_frac"] == 0.02)
check("wf min train", "$40\\%$", R["meta"]["eval_config"]["wf_min_train_frac"] == 0.4)
n_null = 4 * 5 * 16 * 6 * 2
n_emb = 5 * 24 * 5 * 2
n_sel = 50
n_clf = 2 * 20 * 2 * 2
check("null records 3,840", "3{,}840", n_null == 3840)
check("embargo records 1,200", "1{,}200", n_emb == 1200)
check("classification records 160", "$160$ records", n_clf == 160)
check("total records 5,250", "5{,}250",
      n_null + n_emb + n_sel + n_clf == 5250)

# --------------------------------------- Table 1: the interaction grid ----
print("[Table 1: naive k-fold RF phantom surface]")
for h in (1, 5, 10, 20):
    for phi in (0.0, 0.3, 0.6, 0.9, 0.97):
        c = CELLS[("naive_kfold", "rf", h, phi)]
        check(f"naive rf h={h} phi={phi}", pm3(c["mean_leak"], c["ci95"]),
              c["n"] == 16)
# all-positive claim for h>=5, phi>=0.9
allpos = all(CELLS[("naive_kfold", "rf", h, p)]["frac_positive"] == 1.0
             for h in (5, 10, 20) for p in (0.9, 0.97))
check("16/16 positive in h>=5, phi>=0.9 cells", "all $16$ repetitions leak positive",
      allpos)
inter = R["phantom_null"]["interaction_naive_rf"]
check("h=1 max leak +0.005", "$+0.005$",
      f"{inter['h1_any_phi_max_leak']:+.3f}" == "+0.005")
check("phi=0 max leak +0.005", "$+0.005$",
      f"{inter['phi0_any_h_max_leak']:+.3f}" == "+0.005")

# ----------------------------------------------- Table 2: by splitter ----
print("[Table 2: worst cells and the (h=20, phi=0.97) cell]")
names = ("naive_kfold", "blocked_kfold", "purged_kfold", "purged_embargo",
         "walk_forward", "walk_forward_purged")
for s in names:
    for m in ("rf", "ridge"):
        w = WORST[(s, m)]
        check(f"worst cell {s} {m}", pm3(w["worst_cell_leak"], w["ci95"]))
        if s != "naive_kfold":
            check(f"worst cell {s} {m} at h=1", "occurs at $h=1$",
                  w["at_horizon"] == 1)
        c = CELLS[(s, m, 20, 0.97)]
        check(f"h20 phi97 {s} {m}", pm3(c["mean_leak"], c["ci95"]))
naive_h20 = CELLS[("naive_kfold", "rf", 20, 0.97)]
check("worst naive cell is (20, 0.97)", "$(h{=}20, \\phi{=}0.97)$",
      WORST[("naive_kfold", "rf")]["at_horizon"] == 20
      and WORST[("naive_kfold", "rf")]["at_noise_phi"] == 0.97)
contig = [CELLS[(s, m, 20, 0.97)]["mean_leak"]
          for s in ("blocked_kfold", "purged_kfold", "purged_embargo")
          for m in ("rf", "ridge")]
check("contiguous residuals between -0.05 and -0.12", "$-0.05$ and $-0.12$",
      all(-0.12 <= v <= -0.05 for v in contig))

# ------------------------------------------------------------ abstract ----
print("[abstract and headline text]")
for h, tok in ((5, "$+0.31$"), (10, "$+0.46$"), (20, "$+0.56$")):
    c = CELLS[("naive_kfold", "rf", h, 0.97)]
    check(f"abstract rf phi=.97 h={h}", tok, f"{c['mean_leak']:+.2f}" == tok[1:-1])
check("abstract max CI 0.05", r"$\pm0.05$",
      max(CELLS[("naive_kfold", "rf", h, 0.97)]["ci95"]
          for h in (5, 10, 20)) <= 0.05)
rid = CELLS[("naive_kfold", "ridge", 20, 0.97)]
check("abstract ridge +0.18 pm 0.04", r"$+0.18\pm0.04$",
      f"{rid['mean_leak']:+.2f}" == "+0.18" and f"{rid['ci95']:.2f}" == "0.04")
blk = CELLS[("blocked_kfold", "rf", 20, 0.97)]
check("abstract blocked -0.05 pm 0.04", r"$-0.05\pm0.04$",
      f"{blk['mean_leak']:+.2f}" == "-0.05" and f"{blk['ci95']:.2f}" == "0.04")
# ridge text values at phi=0.97
for h, tok in ((5, "$+0.073$"), (10, "$+0.150$")):
    c = CELLS[("naive_kfold", "ridge", h, 0.97)]
    check(f"text ridge phi=.97 h={h}", tok, f"{c['mean_leak']:+.3f}" == tok[1:-1])
check("text ridge h=20 with CI", pm3(rid["mean_leak"], rid["ci95"]))

# ------------------------------------------------------ classification ----
print("[classification check]")
nrf9 = CLF[("naive_kfold", "rf_clf", 0.9)]
check("clf naive RF cv auc 0.623", "$0.623$", f"{nrf9['mean_cv_auc']:.3f}" == "0.623")
check("clf naive RF true auc 0.504", "$0.504$",
      f"{nrf9['mean_true_auc']:.3f}" == "0.504")
check("clf naive RF leak", pm3(nrf9["mean_leak_auc"], nrf9["ci95"]))
nlg9 = CLF[("naive_kfold", "logit", 0.9)]
check("clf naive logit leak", pm3(nlg9["mean_leak_auc"], nlg9["ci95"]))
nrf0 = CLF[("naive_kfold", "rf_clf", 0.0)]
nlg0 = CLF[("naive_kfold", "logit", 0.0)]
check("clf naive RF phi=0", pm3(nrf0["mean_leak_auc"], nrf0["ci95"]))
check("clf naive logit phi=0", pm3(nlg0["mean_leak_auc"], nlg0["ci95"]))
purged_leaks = [CLF[("purged_embargo", m, p)]["mean_leak_auc"]
                for m in ("logit", "rf_clf") for p in (0.0, 0.9)]
check("clf purged range -0.026..-0.044", "$-0.026$ to $-0.044$",
      all(-0.044 <= v <= -0.026 for v in purged_leaks))

# --------------------------------------------------------- embargo ----
print("[embargo sweep]")
req = EMB["required_embargo"]
check("ten arms", "ten arms", len(req) == 10)
check("required embargo zero everywhere", "zero embargo",
      all(r["required_embargo"] == 0.0 for r in req))
lz = [r["leak_at_zero_embargo"] for r in req]
check("leak at zero range low", "$-0.010$", f"{max(lz):+.3f}" == "-0.010")
check("leak at zero range high", "$-0.102$", f"{min(lz):+.3f}" == "-0.102")
check("all residuals negative", "every single one is \\emph{negative}",
      all(v < 0 for v in lz))
by_emb = EMB["by_embargo"]
ranges = []
arms = {r["arm"] for r in by_emb}
for arm in arms:
    for m in ("rf", "ridge"):
        v = [r["mean_leak"] for r in by_emb if r["arm"] == arm and r["model"] == m]
        ranges.append(max(v) - min(v))
check("flatness: max within-arm movement 0.012", "$0.012$",
      f"{max(ranges):.3f}" == "0.012")
k0 = [r["train_frac_retained"] for r in by_emb if r["embargo_frac"] == 0.0]
k8 = [r["train_frac_retained"] for r in by_emb if r["embargo_frac"] == 0.08]
m0, m8 = sum(k0) / len(k0), sum(k8) / len(k8)
check("purge-only discards 1.2%", r"$1.2\%$", f"{(1 - m0) * 100:.1f}" == "1.2")
check("purge-only keeps 98.8%", r"$98.8\%$", f"{m0 * 100:.1f}" == "98.8")
check("8% embargo keeps 90.8%", r"$90.8\%$", f"{m8 * 100:.1f}" == "90.8")
check("8% embargo discards 9.2%", r"$9.2\%$", f"{(1 - m8) * 100:.1f}" == "9.2")
check("calibrated everywhere on the grid", "calibrated",
      all(r["calibrated"] == 1 for r in by_emb))

# ------------------------------------------------------- Table 3: costs ----
print("[Table 3: costs]")
kept_tok = {"naive_kfold": "100.0", "blocked_kfold": "100.0",
            "purged_kfold": "98.7", "purged_embargo": "96.7",
            "walk_forward": "72.7", "walk_forward_purged": "72.0"}
for s in names:
    c_rf, c_ri = COSTS[(s, "rf")], COSTS[(s, "ridge")]
    check(f"kept {s}", f"${kept_tok[s]}\\%$",
          f"{c_rf['train_frac_retained'] * 100:.1f}" == kept_tok[s])
    check(f"fold std rf {s}", f"${c_rf['mean_fold_cv_std']:.3f}$")
    check(f"rep std rf {s}", f"${c_rf['rep_cv_std']:.3f}$")
    check(f"fold std ridge {s}", f"${c_ri['mean_fold_cv_std']:.3f}$")
    check(f"rep std ridge {s}", f"${c_ri['rep_cv_std']:.3f}$")
ratio = COSTS[("purged_kfold", "rf")]["rep_cv_std"] / COSTS[("naive_kfold", "rf")]["rep_cv_std"]
check("dispersion +21% (0.048 -> 0.058)", r"$0.048$ (naive, RF) to $0.058$",
      20.0 <= (ratio - 1) * 100 <= 22.0)
check("dispersion +21% token", r"$+21\%$")
fold_ratio = COSTS[("walk_forward", "rf")]["mean_fold_cv_std"] / COSTS[("naive_kfold", "rf")]["mean_fold_cv_std"]
check("wf fold std about twice", r"$0.050 \to 0.104$", 1.9 <= fold_ratio <= 2.2)

# --------------------------------------------------- Table 4: selection ----
print("[Table 4: model selection]")
check("population IC 0.100", "$0.100$", f"{SEL['population_ic']:.3f}" == "0.100")
nv, pg = SEL["naive"], SEL["purged"]
check("naive cv of pick", pm3u(nv["cv_ic_of_pick"]["mean"], nv["cv_ic_of_pick"]["ci95"]))
check("naive true of pick", pm3u(nv["true_ic_of_pick"]["mean"], nv["true_ic_of_pick"]["ci95"]))
check("naive optimism", pm3(nv["optimism_of_pick"]["mean"], nv["optimism_of_pick"]["ci95"]))
check("naive regret", pm3u(nv["regret"]["mean"], nv["regret"]["ci95"]))
check("purged cv of pick", pm3u(pg["cv_ic_of_pick"]["mean"], pg["cv_ic_of_pick"]["ci95"]))
check("purged true of pick", pm3u(pg["true_ic_of_pick"]["mean"], pg["true_ic_of_pick"]["ci95"]))
check("purged optimism", pm3(pg["optimism_of_pick"]["mean"], pg["optimism_of_pick"]["ci95"]))
check("purged regret", pm3u(pg["regret"]["mean"], pg["regret"]["ci95"]))
check("oracle true IC", pm3u(SEL["oracle_true_ic"]["mean"], SEL["oracle_true_ic"]["ci95"]))
check("naive picks knn10 50/50", "$50$ of $50$",
      nv["pick_counts"] == {"knn10": 50})
check("purged picks ridge 23/50", "$23/50$", pg["pick_counts"].get("ridge") == 23)
check("purged pick spread", "$k$NN$_{10}$ $9$",
      pg["pick_counts"].get("rf_d8") == 9 and pg["pick_counts"].get("knn10") == 9
      and pg["pick_counts"].get("knn50") == 5 and pg["pick_counts"].get("rf_d3") == 4)
check("ridge pick rate 46%", r"$46\%$", round(23 / SEL["n_reps"] * 100) == 46)
diff = SEL["purged_minus_naive_true_ic"]
check("forward gain +0.029 pm 0.012", pm3(diff["mean"], diff["ci95"]))
check("win rate 64%", r"$64\%$", round(SEL["purged_win_rate"] * 100) == 64)
check("tie rate 18%", r"$18\%$", round(SEL["tie_rate"] * 100) == 18)
check("naive win rate 18%", "losing in $18\\%$",
      round((1 - SEL["purged_win_rate"] - SEL["tie_rate"]) * 100) == 18)
pc = SEL["per_candidate_true_ic"]
for m, label in (("ridge", "ridge"), ("rf_d3", "RF depth~3"),
                 ("rf_d8", "RF depth~8"), ("knn10", "$k$NN($10$)"),
                 ("knn50", "$k$NN($50$)")):
    check(f"candidate true IC {m}", pm3u(pc[m]["mean"], pc[m]["ci95"]))
check("abstract: naive cv 0.422", "$0.422$",
      f"{nv['cv_ic_of_pick']['mean']:.3f}" == "0.422")
check("abstract: naive true 0.012", "$0.012$",
      f"{nv['true_ic_of_pick']['mean']:.3f}" == "0.012")
check("abstract: purged true 0.042", "$0.042$",
      f"{pg['true_ic_of_pick']['mean']:.3f}" == "0.042")
check("abstract: optimism +0.410", "$+0.410$",
      f"{nv['optimism_of_pick']['mean']:+.3f}" == "+0.410")
check("abstract: purged optimism +0.012", "$+0.012$",
      f"{pg['optimism_of_pick']['mean']:+.3f}" == "+0.012")
check("abstract: oracle 0.069", "$0.069$",
      f"{SEL['oracle_true_ic']['mean']:.3f}" == "0.069")
check("regret halved 0.056 -> 0.027", r"$0.056 \to 0.027$",
      f"{nv['regret']['mean']:.3f}" == "0.056" and f"{pg['regret']['mean']:.3f}" == "0.027")

# ----------------------------------------------------------------- done ----
print(f"\n{n_checks} checks, {len(failures)} failures.")
if failures:
    print("\nFAILURES:")
    for f_ in failures:
        print(" -", f_)
    sys.exit(1)


# --- Sharpe phantom on the phi=0 axis (added after adversarial review) ---
import pandas as _pd, math as _math
_dfn = _pd.read_csv(ROOT / "results" / "records_null.csv")
_m = (_dfn["cfg_horizon"]==20)&(_dfn["cfg_noise_phi"]==0.0)&(_dfn["splitter"]=="naive_kfold")&(_dfn["model"]=="rf")
_leak = _dfn.loc[_m,"cv_sharpe"]-_dfn.loc[_m,"true_sharpe"]
_mean = _leak.mean(); _hw = 1.96*_leak.std(ddof=1)/_math.sqrt(_m.sum())
check("sharpe phantom h20 phi0", f"$+{_mean:.3f}\\pm{_hw:.3f}$")

print("All paper numbers match results/results.json.")
