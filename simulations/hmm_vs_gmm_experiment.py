"""
Experiment: forecasts from a 3‑state Gaussian HMM (hmmTMB) vs. a Gaussian
mixture that ignores time dependence.

For a range of persistence levels in the true transition matrix we:
  1) simulate a univariate series from a known 3‑state HMM,
  2) fit a 3‑state HMM via the existing R back‑end, and
  3) fit a GaussianMixture baseline,
then compare predictive log scores and RMSE of forecast means on a held‑out
forecast horizon.

Run with your R/python env active:
    python simulations/hmm_vs_gmm_experiment.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import pathlib
import sys

# Ensure repo root is on sys.path when running as a script
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from models.forecast_wrapper import forecast_model
from models.simulate_hmm import simulate_gaussian_hmm
from sklearn.mixture import GaussianMixture


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def _mixture_mean_std(means: np.ndarray, sds: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return mixture mean and std per step."""
    mean = means @ weights
    var = (sds**2 + means**2) @ weights - mean**2
    return mean, np.sqrt(np.maximum(var, 0.0))


def _log_mixture_pdf(y: float, means: np.ndarray, sds: np.ndarray, weights: np.ndarray) -> float:
    """Stable log pdf of a Gaussian mixture at a scalar y."""
    log_components = []
    for mu, sd, w in zip(means, sds, weights):
        var = sd * sd
        log_prob = -0.5 * math.log(2 * math.pi * var) - (y - mu) ** 2 / (2 * var)
        log_components.append(math.log(w) + log_prob)
    # log-sum-exp
    m = max(log_components)
    return m + math.log(sum(math.exp(lc - m) for lc in log_components))


@dataclass
class ForecastMetrics:
    avg_log_score: float
    rmse: float


def evaluate_forecast(forecast_pdf: dict, y_true: Sequence[float]) -> ForecastMetrics:
    """Compute mean log score and RMSE of mixture mean vs. actuals."""
    weights = np.asarray(forecast_pdf["weights"], dtype=float)
    means = np.asarray(forecast_pdf["means"], dtype=float)
    sds = np.asarray(forecast_pdf["sds"], dtype=float)
    steps = len(y_true)

    log_scores = [
        _log_mixture_pdf(float(y_true[i]), means[i], sds[i], weights)
        for i in range(steps)
    ]
    mix_mean, _ = _mixture_mean_std(means, sds, weights)
    rmse = float(np.sqrt(np.mean((mix_mean - np.asarray(y_true, dtype=float)) ** 2)))
    return ForecastMetrics(avg_log_score=float(np.mean(log_scores)), rmse=rmse)


def forecast_gmm(y_train: np.ndarray, steps: int, n_components: int = 3, random_state: int | None = None):
    """Fit a static Gaussian mixture and reuse it for each future step."""
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(y_train.reshape(-1, 1))

    comp_means = gmm.means_.reshape(1, -1)
    comp_sds = np.sqrt(gmm.covariances_.reshape(gmm.n_components, -1)[:, 0])

    means = np.tile(comp_means, (steps, 1))
    sds = np.tile(comp_sds, (steps, 1))
    forecast_pdf = {"weights": gmm.weights_, "means": means, "sds": sds}
    return forecast_pdf, gmm


def simulate_series(n: int, persist: float, means: Sequence[float], sds: Sequence[float], seed: int | None = None) -> np.ndarray:
    """Simulate observations from the R-side helper."""
    sim = simulate_gaussian_hmm(
        n=n,
        persist=persist,
        means=np.asarray(means, dtype=float),
        sds=np.asarray(sds, dtype=float),
        seed=seed,
    )
    return sim["y"]


def run_single_experiment(
    *,
    persist: float,
    n_train: int,
    n_test: int,
    means: Sequence[float],
    sds: Sequence[float],
    seed: int | None = None,
) -> dict:
    """Simulate once for a given persistence level and score both forecasters."""
    series = simulate_series(n_train + n_test, persist, means, sds, seed=seed)
    y_train, y_test = series[:n_train], series[n_train:]

    hmm_res = forecast_model(model="HMM", y_train=y_train, steps=n_test, n_states=3, obs_dist="norm")
    hmm_metrics = evaluate_forecast(hmm_res["forecast_pdf"], y_test)

    gmm_pdf, _ = forecast_gmm(y_train, steps=n_test, n_components=3, random_state=seed)
    gmm_metrics = evaluate_forecast(gmm_pdf, y_test)

    return {
        "persist": persist,
        "hmm_log_score": hmm_metrics.avg_log_score,
        "gmm_log_score": gmm_metrics.avg_log_score,
        "hmm_rmse": hmm_metrics.rmse,
        "gmm_rmse": gmm_metrics.rmse,
    }


def run_grid(
    persistence_levels: Iterable[float],
    n_train: int = 300,
    n_test: int = 60,
    means: Sequence[float] = (-2.0, 0.0, 2.0),
    sds: Sequence[float] = (0.5, 0.6, 0.7),
    seed: int | None = 123,
) -> list[dict]:
    results: list[dict] = []
    for persist in persistence_levels:
        results.append(
            run_single_experiment(
                persist=persist,
                n_train=n_train,
                n_test=n_test,
                means=means,
                sds=sds,
                seed=seed,
            )
        )
    return results


def main():
    # persist ~ 1/number_of_states ~ i.i.d.; persist -> 1 => highly persistent
    persistence_levels = [0.34, 0.6, 0.8, 0.95]
    res = run_grid(persistence_levels)

    header = ("persist", "hmm_log", "gmm_log", "hmm_rmse", "gmm_rmse")
    print(" | ".join(f"{h:>10s}" for h in header))
    for row in res:
        print(
            f"{row['persist']:10.2f} | "
            f"{row['hmm_log_score']:10.3f} | {row['gmm_log_score']:10.3f} | "
            f"{row['hmm_rmse']:10.3f} | {row['gmm_rmse']:10.3f}"
        )


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
