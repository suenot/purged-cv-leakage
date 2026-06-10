# How Much Does Purging Actually Fix?

A controlled quantification of overlapping-label leakage in time-series
cross-validation (Lopez de Prado's purged k-fold with embargo), measured on
synthetic return series whose true predictability is **known exactly --
including exactly zero**. Any CV-estimated skill above the truth is leakage,
and is measured directly as `leak = CV estimate - true forward skill`.

What the experiments quantify:

1. **Phantom skill of naive shuffled k-fold under the null** as a function of
   the label horizon `h` (overlap) and feature persistence `phi` -- the
   headline surface, with 95% CIs. Phantom skill is an *interaction*: it needs
   both `h > 1` and `phi > 0`, and it is model-flexibility dependent (random
   forest memorizes far more of the leak than ridge).
2. **Blocked k-fold without purging**: how much boundary leakage survives.
3. **Purged k-fold + embargo**: whether purging restores null calibration, and
   how much embargo a *persistent genuine signal* still requires.
4. **Costs**: training samples sacrificed by purge/embargo and CV-estimate
   dispersion.
5. **Model selection**: with a real signal and a model menu, does purged CV
   pick truly better models than naive CV (forward-truth comparison)?

This is a de-commercialized, experimentally-validated companion to a
[marketmaker.cc](https://marketmaker.cc) explainer on purged cross-validation.

## Reproduce everything

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_all.py             # full run -> results/results.json + 4 CSVs
python -m purged_cv_experiments.figures  # -> paper/figures/*.pdf
```

Deterministic given the seeds in `scripts/run_all.py`. `--quick` runs a small
smoke batch (~1 min).

## Layout

```
purged_cv_experiments/
  model.py       # DGP: overlapping forward labels + noise/signal AR(1) features,
                 #      closed-form population IC (zero for the null)
  splitters.py   # naive / blocked / purged+embargo (from scratch) / walk-forward
  simulate.py    # CV estimate vs forward truth per (splitter, model); batches
  analysis.py    # phantom-skill surface, embargo requirement, costs, selection
  figures.py     # the paper's 4 vector-PDF figures
scripts/run_all.py
tests/           # pytest sanity checks incl. exhaustive purge-correctness proof
results/         # results.json + records_*.csv (generated)
```

## Tests

```bash
python -m pytest -q
```

## License

Code: MIT. Paper text and figures: CC BY 4.0.
