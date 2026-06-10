"""Data-generating process with *known* ground-truth predictability.

The point of the paper is to measure leakage directly, so every experiment is
run on a synthetic return series in which the true predictive power of every
feature is known exactly -- including the case of exactly zero.

Time convention (everything is in observation-index units):

* ``r_t`` is the return realized over the step ending at time ``t``.
* Features are observable at time ``t`` (they use information up to ``t``).
* The label of sample ``t`` is the *forward h-period cumulative return*
  ``y_t = sum_{u=t+1}^{t+h} r_u``.  For ``h > 1`` consecutive labels share
  ``h-1`` returns -- the **overlapping-label** leakage source studied here.
* The *information interval* (label window) of sample ``t`` is ``[t, t+h]``:
  it conservatively includes the decision time ``t`` and the last return time
  ``t+h``.  Purging is defined on these intervals.
* A binary classification label ``sign(y_t) > 0`` is also produced.

Two feature families, both AR(1) with exactly unit stationary variance:

* **Pure-noise features** (``n_noise_features`` columns): independent AR(1)
  processes with persistence ``noise_phi``, generated independently of the
  return series.  Their true predictive power is *exactly zero* -- any
  measured CV skill on a noise-only config is leakage (or sampling noise).
* **Genuine-signal feature** (``n_signal_features in {0, 1}``): a latent
  AR(1) ``s_t`` with persistence ``signal_phi`` drives returns through
  ``r_{t+1} = signal_strength * s_t + ret_vol * eps_{t+1}``; the observed
  feature is ``s_t + signal_obs_noise * xi_t``.  Its population linear IC
  against the h-period label is available in closed form
  (:func:`population_ic`).

Ground truth for *fitted models* is measured on a forward continuation of the
same realized series: ``generate`` produces ``t_obs`` in-sample observations,
an ``h``-step gap (so no label window crosses the boundary), and ``t_holdout``
forward observations that the CV procedures never see.

Every constant is a recorded config field -- there are no hidden hard-coded
parameters.  Everything is deterministic given a seeded ``numpy`` Generator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DGPConfig:
    """Ground-truth data-generating process for one experiment."""

    t_obs: int = 2000            # in-sample observations handed to the CV scheme
    t_holdout: int = 6000        # forward observations for ground-truth skill
    horizon: int = 10            # h: label = forward h-period cumulative return
    n_noise_features: int = 5    # pure-noise AR(1) features (zero true signal)
    noise_phi: float = 0.0       # AR(1) persistence of the noise features
    n_signal_features: int = 0   # 0 (null) or 1 (genuine signal)
    signal_strength: float = 0.0 # beta: per-period return loading on s_{t-1}
    signal_phi: float = 0.9      # AR(1) persistence of the latent signal s_t
    signal_obs_noise: float = 0.0  # sd of observation noise added to s_t
    ret_vol: float = 1.0         # sd of the idiosyncratic return innovation
    label: str = "custom"

    def __post_init__(self) -> None:
        if self.n_signal_features not in (0, 1):
            raise ValueError("n_signal_features must be 0 or 1 (analytic truth "
                             "is only derived for a single signal feature)")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        for name in ("noise_phi", "signal_phi"):
            phi = getattr(self, name)
            if not (0.0 <= phi < 1.0):
                raise ValueError(f"{name} must be in [0, 1)")


@dataclass(frozen=True)
class Dataset:
    """One realized series: features, labels, and index bookkeeping."""

    X: np.ndarray            # (n_valid, K) features at times 0..n_valid-1
    y: np.ndarray            # (n_valid,) forward h-period cumulative returns
    y_class: np.ndarray      # (n_valid,) int 0/1: y > 0
    label_start: np.ndarray  # (n_valid,) information-interval start = t
    label_end: np.ndarray    # (n_valid,) information-interval end = t + h
    insample_idx: np.ndarray # CV-visible sample times: [0, t_obs)
    holdout_idx: np.ndarray  # forward ground-truth times: [t_obs+h, t_obs+h+t_holdout)
    cfg: DGPConfig


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
def _ar1(phi: float, n: int, n_cols: int, rng: np.random.Generator) -> np.ndarray:
    """(n, n_cols) stationary AR(1) paths with exactly unit variance.

    ``x_0 ~ N(0,1)``, ``x_t = phi * x_{t-1} + sqrt(1-phi^2) * z_t``.
    Implemented with ``scipy.signal.lfilter`` (vectorized, deterministic).
    """
    from scipy.signal import lfilter

    z = rng.standard_normal((n, n_cols))
    x0 = rng.standard_normal((1, n_cols))
    innov = np.sqrt(1.0 - phi * phi) * z
    # lfilter computes x_t = innov_t + phi * x_{t-1}; zi seeds x_{-1} = x0.
    zi = phi * x0  # initial filter delay state for each column
    x, _ = lfilter([1.0], [1.0, -phi], innov, axis=0, zi=zi)
    return x


def generate(cfg: DGPConfig, rng: np.random.Generator) -> Dataset:
    """Simulate one full series (in-sample + gap + forward holdout)."""
    h = cfg.horizon
    n_valid = cfg.t_obs + h + cfg.t_holdout       # sample times needing labels
    n_total = n_valid + h                          # returns needed up to n_valid-1+h

    # latent signal (one extra leading sample so r_t can load on s_{t-1})
    if cfg.n_signal_features == 1:
        s = _ar1(cfg.signal_phi, n_total + 1, 1, rng)[:, 0]   # s[-1..n_total-1] shifted
    else:
        s = np.zeros(n_total + 1)

    eps = rng.standard_normal(n_total)
    # r[t] = beta * s_{t-1} + ret_vol * eps_t ; s array index t equals time t-1
    r = cfg.signal_strength * s[:-1] + cfg.ret_vol * eps

    # forward h-period cumulative returns: y_t = sum_{u=t+1}^{t+h} r_u
    csum = np.concatenate([[0.0], np.cumsum(r)])
    t = np.arange(n_valid)
    y = csum[t + h + 1] - csum[t + 1]

    # features at time t: noise AR(1) columns + (optionally) observed signal s_t
    cols = []
    if cfg.n_noise_features > 0:
        cols.append(_ar1(cfg.noise_phi, n_valid, cfg.n_noise_features, rng))
    if cfg.n_signal_features == 1:
        obs = s[1 : n_valid + 1].copy()            # s_t for t = 0..n_valid-1
        if cfg.signal_obs_noise > 0:
            obs = obs + cfg.signal_obs_noise * rng.standard_normal(n_valid)
        cols.append(obs[:, None])
    if not cols:
        raise ValueError("config has zero features")
    X = np.hstack(cols)

    return Dataset(
        X=X,
        y=y,
        y_class=(y > 0).astype(int),
        label_start=t,
        label_end=t + h,
        insample_idx=np.arange(cfg.t_obs),
        holdout_idx=np.arange(cfg.t_obs + h, cfg.t_obs + h + cfg.t_holdout),
        cfg=cfg,
    )


# --------------------------------------------------------------------------- #
# analytic ground truth
# --------------------------------------------------------------------------- #
def population_ic(cfg: DGPConfig) -> float:
    """Population Pearson correlation between the observed signal feature at
    time ``t`` and the h-period forward label ``y_t``.

    Exactly ``0.0`` for noise-only configs.  For the single-signal config:

    ``cov(f_t, y_t) = beta * sum_{j=0}^{h-1} phi_s^j``
    ``var(f_t)      = 1 + nu^2``
    ``var(y_t)      = beta^2 * var(S_h) + h * sigma_r^2`` with
    ``var(S_h) = h + 2 * sum_{d=1}^{h-1} (h-d) phi_s^d`` (unit-variance AR(1)).
    """
    if cfg.n_signal_features == 0 or cfg.signal_strength == 0.0:
        return 0.0
    h, phi, beta = cfg.horizon, cfg.signal_phi, cfg.signal_strength
    g = float(np.sum(phi ** np.arange(h)))
    d = np.arange(1, h)
    var_s_sum = h + 2.0 * float(np.sum((h - d) * phi**d))
    cov = beta * g
    var_f = 1.0 + cfg.signal_obs_noise**2
    var_y = beta**2 * var_s_sum + h * cfg.ret_vol**2
    return float(cov / np.sqrt(var_f * var_y))


def strength_for_population_ic(
    target_ic: float, *, horizon: int, signal_phi: float,
    signal_obs_noise: float = 0.0, ret_vol: float = 1.0,
) -> float:
    """Invert :func:`population_ic`: the ``signal_strength`` giving ``target_ic``.

    Solves ``ic^2 = beta^2 g^2 / ((1+nu^2)(beta^2 V + h sigma^2))`` for beta.
    """
    if not (0.0 < target_ic < 1.0):
        raise ValueError("target_ic must be in (0, 1)")
    h, phi = horizon, signal_phi
    g = float(np.sum(phi ** np.arange(h)))
    d = np.arange(1, h)
    var_s_sum = h + 2.0 * float(np.sum((h - d) * phi**d))
    c2 = target_ic**2 * (1.0 + signal_obs_noise**2)
    denom = g**2 - c2 * var_s_sum
    if denom <= 0:
        raise ValueError("target_ic unreachable for this horizon/persistence")
    return float(np.sqrt(c2 * h * ret_vol**2 / denom))


# --------------------------------------------------------------------------- #
# canonical configs
# --------------------------------------------------------------------------- #
def canonical_configs(t_obs: int = 2000, t_holdout: int = 6000) -> dict[str, DGPConfig]:
    """Illustrative configs used by the setup figure and the tests."""
    return {
        "null_iid": DGPConfig(
            t_obs=t_obs, t_holdout=t_holdout, horizon=10,
            n_noise_features=5, noise_phi=0.0, label="null_iid"),
        "null_persistent": DGPConfig(
            t_obs=t_obs, t_holdout=t_holdout, horizon=10,
            n_noise_features=5, noise_phi=0.97, label="null_persistent"),
        "signal": DGPConfig(
            t_obs=t_obs, t_holdout=t_holdout, horizon=10,
            n_noise_features=4, noise_phi=0.9,
            n_signal_features=1, signal_phi=0.97,
            signal_strength=strength_for_population_ic(
                0.10, horizon=10, signal_phi=0.97),
            label="signal"),
    }
