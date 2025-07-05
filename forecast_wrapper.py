"""
forecast_wrapper.py

Unified interface to forecast time‑series using different engines.
Currently implemented back‑ends
--------------------------------
* **LSTM** – Keras (pure Python)
* **ARIMA / SARIMA / SARIMAX** – via **rpy2** calling the R helper
  `forecast_arima()` found in *models/arima_model.R*

Place‑holders (coming soon)
--------------------------
* **RNN / GRU**  – Keras or PyTorch
* **HMM**        – R package (via rpy2) or `hmmlearn` in Python


Example usage
-------------
```python
from forecast_wrapper import forecast_model

# LSTM (univariate)
forecast_model(model="LSTM", y_train=y, steps=12)

# ARIMA with exogenous regressors
forecast_model(
    model="ARIMA",
    y_train=y,
    steps=12,
    xreg_train=X_cov,
    xreg_future=X_cov_future,
    order=(2,1,2),
    seasonal_order=(1,1,0),
)
```
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, Sequence

import numpy as np
from sklearn.mixture import GaussianMixture

# ---------------------------------------------------------------------------
# Optional rpy2 / R setup for ARIMA back‑end
# ---------------------------------------------------------------------------
try:
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri, pandas2ri

    numpy2ri.activate()
    pandas2ri.activate()

    # Locate R helper: ./models/arima_model.R
    R_HELPER_PATH = pathlib.Path(__file__).resolve().parent / "models" / "arima_model.R"
    if not R_HELPER_PATH.exists():
        raise FileNotFoundError("models/arima_model.R not found relative to forecast_wrapper.py")

    ro.r.source(str(R_HELPER_PATH))
    _forecast_arima_r = ro.globalenv["forecast_arima"]
    _RPY2_AVAILABLE = True
except (ImportError, FileNotFoundError) as _err:
    _RPY2_AVAILABLE = False
    _forecast_arima_r = None

# ---------------------------------------------------------------------------
# Local LSTM back‑end (pure Python)
# ---------------------------------------------------------------------------
from models.lstm_model import forecast_lstm

__all__ = ["forecast_model"]

# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def _py_none_to_r_null(obj):
    """Convert Python None → R NULL for rpy2 calls."""
    return ro.NULL if obj is None else obj


def forecast_model(
    *,
    model: str,
    y_train: Sequence[float],
    steps: int,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Fit the chosen model and return a dictionary of results."""

    model_key = model.upper()
    y_train_arr = np.asarray(y_train, dtype=float)

    # --------------------------------- LSTM --------------------------------
    if model_key == "LSTM":
        forecast_pdf, fitted_model = forecast_lstm(
            y_train=y_train_arr, steps=steps, **kwargs
        )
        return {
            "model": "LSTM",
            "forecast_pdf": forecast_pdf,
            "meta": {"keras_model": fitted_model},
        }

    # -------------------------------- ARIMA --------------------------------
    if model_key == "ARIMA":
        if not _RPY2_AVAILABLE:
            raise ImportError("ARIMA back‑end requires rpy2 and models/arima_model.R")

        # Pull known special‑case kwargs first
        xreg_train = _py_none_to_r_null(kwargs.pop("xreg_train", None))
        xreg_future = _py_none_to_r_null(kwargs.pop("xreg_future", None))

        n_components = kwargs.pop("n_components", 1)
        random_state = kwargs.pop("random_state", None)

        # rpy2 will auto‑convert numpy arrays / pandas frames
        r_res = _forecast_arima_r(
            y_train_arr,
            steps,
            xreg_train,
            xreg_future,
            **{k: _py_none_to_r_null(v) for k, v in kwargs.items()},
        )

        forecast_np = np.asarray(r_res.rx2("forecast"), dtype=float)
        fitted_model_r = r_res.rx2("model")

        resid_r = ro.r["residuals"](fitted_model_r)
        resid = np.asarray(resid_r, dtype=float)
        gmm = GaussianMixture(n_components=n_components, random_state=random_state)
        gmm.fit(resid.reshape(-1, 1))

        comp_means = gmm.means_.reshape(1, -1)
        comp_vars = gmm.covariances_.reshape(gmm.n_components, -1)[:, 0]
        comp_sds = np.sqrt(comp_vars)
        means = forecast_np.reshape(-1, 1) + comp_means
        sds = np.tile(comp_sds, (steps, 1))
        forecast_pdf = {"weights": gmm.weights_, "means": means, "sds": sds}

        return {
            "model": "ARIMA",
            "forecast_pdf": forecast_pdf,
            "meta": {"r_model": fitted_model_r},
        }

    # -------------------------- Future back‑ends ---------------------------
    if model_key == "RNN":
        raise NotImplementedError("RNN back‑end not yet implemented.")
    if model_key == "HMM":
        raise NotImplementedError("HMM back‑end not yet implemented.")

    raise ValueError(f"Unknown model type: {model}")
