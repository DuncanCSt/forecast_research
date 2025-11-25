# test_wrapper.py
#
# Quick smoke-test for forecast_wrapper: LSTM vs ARIMA
# ----------------------------------------------------
# Run from your conda env with:  python test_wrapper.py

import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.forecast_wrapper import forecast_model

def _mixture_mean_std(means: np.ndarray, sds: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the mixture mean and standard deviation for each step."""
    mean = means @ weights
    var = (sds**2 + means**2) @ weights - mean**2
    return mean, np.sqrt(var)

# 1) ── Generate a simple synthetic series ────────────────────────────────────
#    y(t) = sin(2πt / 20)  + gaussian noise
np.random.seed(42)
t = np.arange(160)
y = np.sin(2 * np.pi * t / 20) + 0.3 * np.random.randn(len(t))

# Split 140 points for training, keep the last 20 for visual sanity checks
y_train, y_test = y[:140], y[140:]

# 2) ── LSTM forecast (recursive) ─────────────────────────────────────────────
lstm_res = forecast_model(
    model="LSTM",
    y_train=y_train,
    steps=5,
    lag=30,           # use 30-point windows
    epochs=40,
    verbose=0,
    n_components=2,
)

# 3) ── ARIMA forecast (auto-arima) ───────────────────────────────────────────
arima_res = forecast_model(
    model="ARIMA",
    y_train=y_train,
    steps=5,
    frequency=1,      # not strictly needed here (no seasonality)
    n_components=2,
    # order=(2,1,2),  # uncomment to pin (p,d,q) instead of auto search
)

# 3b) ── HMM forecast (hmmTMB via rpy2) ───────────────────────────────────────
hmm_res = forecast_model(
    model="HMM",
    y_train=y_train,
    steps=5,
    n_states=2,
    hid_formula=None
)

# 4) ── Compare outputs ───────────────────────────────────────────────────────
print("Target (next 5 actual points):")
print(y_test[:5])
print("\nLSTM  forecast mixture means:")
print(lstm_res["forecast_pdf"]["means"])
print("\nARIMA forecast mixture means:")
print(arima_res["forecast_pdf"]["means"])
print("\nHMM   forecast mixture means:")
print(hmm_res["forecast_pdf"]["means"])


plt.plot(np.arange(len(y)), y, label="Actual", lw=1)
future_idx = np.arange(len(y_train), len(y_train) + 5)

lstm_pdf = lstm_res["forecast_pdf"]
arima_pdf = arima_res["forecast_pdf"]
hmm_pdf = hmm_res["forecast_pdf"]

lstm_mean, lstm_std = _mixture_mean_std(
    lstm_pdf["means"], lstm_pdf["sds"], lstm_pdf["weights"]
)
arima_mean, arima_std = _mixture_mean_std(
    arima_pdf["means"], arima_pdf["sds"], arima_pdf["weights"]
)
hmm_mean, hmm_std = _mixture_mean_std(
    hmm_pdf["means"], hmm_pdf["sds"], hmm_pdf["weights"]
)

line = plt.plot(future_idx, lstm_mean, "o-", label="LSTM")[0]
plt.fill_between(
    future_idx,
    lstm_mean - 1.96 * lstm_std,
    lstm_mean + 1.96 * lstm_std,
    color=line.get_color(),
    alpha=0.2,
)

line = plt.plot(future_idx, arima_mean, "s-", label="ARIMA")[0]
plt.fill_between(
    future_idx,
    arima_mean - 1.96 * arima_std,
    arima_mean + 1.96 * arima_std,
    color=line.get_color(),
    alpha=0.2,
)

line = plt.plot(future_idx, hmm_mean, "^-", label="HMM")[0]
plt.fill_between(
    future_idx,
    hmm_mean - 1.96 * hmm_std,
    hmm_mean + 1.96 * hmm_std,
    color=line.get_color(),
    alpha=0.2,
)

plt.axvline(len(y_train) - 0.5, color="grey", ls="--")
plt.legend()
plt.title("Wrapper smoke test")
plt.show()
