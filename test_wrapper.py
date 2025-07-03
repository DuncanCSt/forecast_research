# test_wrapper.py
#
# Quick smoke-test for forecast_wrapper: LSTM vs ARIMA
# ----------------------------------------------------
# Run from your conda env with:  python test_wrapper.py

from pathlib import Path
import numpy as np
from forecast_wrapper import forecast_model

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
)

# 3) ── ARIMA forecast (auto-arima) ───────────────────────────────────────────
arima_res = forecast_model(
    model="ARIMA",
    y_train=y_train,
    steps=5,
    frequency=1,      # not strictly needed here (no seasonality)
    # order=(2,1,2),  # uncomment to pin (p,d,q) instead of auto search
)

# 4) ── Compare outputs ───────────────────────────────────────────────────────
print("Target (next 5 actual points):")
print(y_test[:5])
print("\nLSTM  forecast:")
print(lstm_res["forecast"])
print("\nARIMA forecast:")
print(arima_res["forecast"])

# Optional: quick plot if you have matplotlib
try:
    import matplotlib.pyplot as plt

    plt.plot(np.arange(len(y)), y, label="Actual", lw=1)
    future_idx = np.arange(len(y_train), len(y_train) + 5)
    plt.plot(future_idx, lstm_res["forecast"], "o-", label="LSTM")
    plt.plot(future_idx, arima_res["forecast"], "s-", label="ARIMA")
    plt.axvline(len(y_train) - 0.5, color="grey", ls="--")
    plt.legend()
    plt.title("Wrapper smoke test")
    plt.show()
except ImportError:
    pass