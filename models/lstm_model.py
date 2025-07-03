"""
lstm_model.py

Keras implementation of a **multivariate** time‑series forecaster using a
single‑layer LSTM and recursive multi‑step prediction.

* **Target** series is always the first feature.
* **Covariates (a.k.a. exogenous regressors)** are optional and may be
  supplied for both the historical sample (`xreg_train`) and the forecast
  horizon (`xreg_future`).  If either is omitted the model falls back to
  the univariate behaviour that existed previously.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM

__all__ = ["forecast_lstm"]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _select_scaler(kind: str | None):
    """Return an instantiated scaler or *None* for passthrough."""
    if kind is None or kind.lower() == "none":
        return None
    if kind.lower() == "standard":
        return StandardScaler()
    if kind.lower() in {"minmax", "min_max"}:
        return MinMaxScaler(feature_range=(-1, 1))
    raise ValueError(f"Unknown scaler kind: {kind}")


def _create_supervised(
    y_series: np.ndarray,
    xreg: Optional[np.ndarray],
    lag: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert target (and optional covariates) into supervised windows.

    Returns
    -------
    X : (samples, lag, n_features)
    y : (samples,)
    """

    n_samples = len(y_series) - lag
    if n_samples <= 0:
        raise ValueError("`y_train` must be longer than `lag`.")

    # Determine number of features in the supervised tensor
    n_cov = 0 if xreg is None else xreg.shape[1]
    n_features = 1 + n_cov  # 1 target + n_cov covariates

    X = np.empty((n_samples, lag, n_features), dtype=float)
    y = np.empty((n_samples,), dtype=float)

    for i in range(n_samples):
        y_window = y_series[i : i + lag]
        if xreg is not None:
            x_window = xreg[i : i + lag]  # (lag, n_cov)
            X[i] = np.column_stack((y_window, x_window))
        else:
            X[i] = y_window.reshape(-1, 1)
        y[i] = y_series[i + lag]

    return X, y


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def forecast_lstm(
    *,
    y_train: Sequence[float],
    steps: int,
    lag: int = 12,
    # ------------------------------ Covariates -----------------------------
    xreg_train: Optional[Sequence[Sequence[float]]] = None,
    xreg_future: Optional[Sequence[Sequence[float]]] = None,
    # ------------------------------ Hyper‑params ---------------------------
    epochs: int = 30,
    batch_size: int = 32,
    units: int = 64,
    validation_split: float = 0.2,
    scaler: str | None = "standard",
    patience: int = 5,
    verbose: int = 0,
) -> Tuple[np.ndarray, Sequential]:
    """Fit an LSTM (optionally with covariates) and issue a recursive forecast.

    Parameters
    ----------
    y_train
        Historical target series (length *T*).
    steps
        Number of future steps to predict.
    lag
        Window length (number of past observations fed to the LSTM).
    xreg_train, xreg_future
        Arrays of shape (T, n_cov) and (steps, n_cov) respectively.  The
        *future* covariate values **must be known in advance**; if not, you
        should forecast them first or switch to direct multi‑step training.
    scaler
        "standard", "minmax", "none" or a custom identifier.

    Returns
    -------
    forecast : np.ndarray  (shape = ``(steps,)``)
        Predictions in the **original (unscaled) space**.
    model : keras.Sequential
        Fitted Keras model.
    """

    # --------------------------- Sanity checks -----------------------------
    y_arr = np.asarray(y_train, dtype=float)

    if xreg_train is not None:
        xreg_arr = np.asarray(xreg_train, dtype=float)
        if len(xreg_arr) != len(y_arr):
            raise ValueError("xreg_train must have the same length as y_train")
        n_cov = xreg_arr.shape[1]
    else:
        xreg_arr = None
        n_cov = 0

    if xreg_future is not None:
        xreg_future_arr = np.asarray(xreg_future, dtype=float)
        if xreg_future_arr.shape[0] != steps:
            raise ValueError("xreg_future rows must equal `steps`.")
        if xreg_future_arr.shape[1] != n_cov:
            raise ValueError("xreg_future must have the same number of columns "
                             "as xreg_train.")
    elif n_cov > 0:
        raise ValueError("xreg_future is required when covariates are used.")
    else:
        xreg_future_arr = None

    # --------------------------- Scaling ------------------------------------
    scaler_y = _select_scaler(scaler)
    if scaler_y is not None:
        y_scaled = scaler_y.fit_transform(y_arr.reshape(-1, 1)).flatten()
    else:
        y_scaled = y_arr

    if n_cov > 0:
        scaler_x = _select_scaler(scaler)
        x_scaled = scaler_x.fit_transform(xreg_arr)
        x_future_scaled = scaler_x.transform(xreg_future_arr)
    else:
        x_scaled = x_future_scaled = None

    # --------------------------- Supervised data ---------------------------
    X, y = _create_supervised(y_scaled, x_scaled, lag=lag)

    # --------------------------- Model --------------------------------------
    n_features = X.shape[2]
    model = Sequential([
        LSTM(units, input_shape=(lag, n_features)),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    es = EarlyStopping(
        monitor="val_loss", mode="min", patience=patience, restore_best_weights=True
    )

    model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[es],
        verbose=verbose,
    )

    # --------------------------- Recursive forecast -------------------------
    y_hist = y_scaled.copy()
    if n_cov > 0:
        x_hist = x_scaled.copy()
    preds_scaled = []

    for t in range(steps):
        y_window = y_hist[-lag:]  # (lag,)
        if n_cov > 0:
            x_window = x_hist[-lag:]  # (lag, n_cov)
            x_input_window = np.column_stack((y_window, x_window))
        else:
            x_input_window = y_window.reshape(-1, 1)

        x_input = x_input_window.reshape((1, lag, n_features))
        yhat_scaled = float(model.predict(x_input, verbose=0)[0, 0])

        preds_scaled.append(yhat_scaled)
        y_hist = np.append(y_hist, yhat_scaled)

        if n_cov > 0:
            x_next = x_future_scaled[t]  # (n_cov,)
            x_hist = np.vstack([x_hist, x_next])

    preds_scaled_arr = np.asarray(preds_scaled)

    if scaler_y is not None:
        preds = scaler_y.inverse_transform(preds_scaled_arr.reshape(-1, 1)).flatten()
    else:
        preds = preds_scaled_arr

    return preds, model
