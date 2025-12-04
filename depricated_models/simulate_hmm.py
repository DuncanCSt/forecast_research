"""
simulate_hmm.py

Utilities to simulate Gaussian HMM series via hmmTMB (through rpy2).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr
    from rpy2.rinterface import NA_Integer
except ImportError as exc:  # pragma: no cover - exercised only when rpy2 missing
    raise ImportError(
        "simulate_hmm requires rpy2 with access to the hmmTMB R package."
    ) from exc


numpy2ri.activate()
importr("hmmTMB")  # ensure package is loaded in the R session

_SIMULATE_HMM_R = """
simulate_gaussian_hmm <- function(n, persist, means, sds,
                                  initial_state = "stationary",
                                  seed = NA_integer_) {
  if (!is.na(seed)) set.seed(seed)
  n_states <- length(means)
  if (length(sds) != n_states) {
    stop("`sds` must have the same length as `means`.")
  }
  off_diag <- (1 - persist) / (n_states - 1)
  tpm <- matrix(off_diag, n_states, n_states)
  diag(tpm) <- persist

  train_df <- data.frame(ID = rep(1, n), y = rep(0, n))
  hid <- hmmTMB::MarkovChain$new(
    data = train_df,
    n_states = n_states,
    tpm = tpm,
    initial_state = initial_state
  )
  obs <- hmmTMB::Observation$new(
    data = train_df,
    n_states = n_states,
    dists = list(y = "norm"),
    par = list(y = list(mean = means, sd = sds))
  )
  hmm <- hmmTMB::HMM$new(hid = hid, obs = obs)
  dat <- hmm$simulate(n = n, silent = TRUE)
  list(y = dat$y, states = attr(dat, "state"))
}
"""

ro.r(_SIMULATE_HMM_R)
_simulate_hmm_r = ro.globalenv["simulate_gaussian_hmm"]

__all__ = ["simulate_gaussian_hmm"]


def simulate_gaussian_hmm(
    *,
    n: int,
    persist: float,
    means: Sequence[float],
    sds: Sequence[float],
    initial_state: str | int | Sequence[int] = "stationary",
    seed: int | None = None,
) -> dict:
    """Simulate a Gaussian HMM with fixed means/SDs and symmetric TPM.

    Parameters
    ----------
    n : int
        Number of observations to simulate.
    persist : float
        On-diagonal transition probability for all states (0..1).
    means, sds : sequences
        State-dependent Gaussian parameters.
    initial_state : str or int
        Passed through to MarkovChain$new(initial_state=...).
    seed : int or None
        Random seed forwarded to the R helper.
    """
    res = _simulate_hmm_r(
        n,
        persist,
        np.asarray(means, dtype=float),
        np.asarray(sds, dtype=float),
        initial_state,
        seed if seed is not None else NA_Integer,
    )
    return {
        "y": np.asarray(res.rx2("y"), dtype=float),
        "states": np.asarray(res.rx2("states"), dtype=int),
    }
