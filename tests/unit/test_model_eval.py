import numpy as np
from forecast_research.functions.wrapper import evaluate_model
from typing import Literal
import pytest
from scipy.stats import norm

@pytest.fixture(scope="module")
def synthetic_data():
    state = [0, 1, 2]
    means = [[3, 1], [5, 1], [7, 1]]
    np.random.seed(0)

    n = 9000
    y = np.array([np.random.normal(loc=means[s][0], scale=means[s][1], size=n//len(state)) for s in state]).flatten()
    np.random.shuffle(y)
    x_1 = np.random.uniform(low=1, high=9, size=n)
    x_2 = np.random.uniform(low=1, high=9, size=n)
    X = np.column_stack((x_1, x_2))

    y_test = np.array([2, 3, 4, 5, 6, 7, 8])
    x_test = np.array([5, 5, 5, 5, 5, 5, 5])
    X_test = np.column_stack((x_test, x_test))

    # Compute true NLL assuming equal mixture weights
    pdfs = np.mean([norm.pdf(y_test, loc=means[s][0], scale=means[s][1]) for s in range(3)], axis=0)
    y_true = -np.log(pdfs)

    return {
        "y_train": y,
        "y_test": y_test,
        "x_1": x_1,
        "x_test": x_test,
        "X_train": X,
        "X_test": X_test,
        "y_true": y_true
    }


def test_synthetic_no_covariates(synthetic_data):
    for model in ["gmr", "hmm"]:
        result = evaluate_model(
            model=model,
            y_train=synthetic_data["y_train"],
            y_test=synthetic_data["y_test"],
            n_states=3
        )
        np.testing.assert_allclose(result["NLL"], synthetic_data["y_true"], rtol=0.05, atol=0.1)

def test_synthetic_1d_covariates(synthetic_data):
    for model in ["gmr", "hmm"]:
        result = evaluate_model(
            model=model,
            y_train=synthetic_data["y_train"],
            y_test=synthetic_data["y_test"],
            x_train=synthetic_data["x_1"],
            x_test=synthetic_data["x_test"],
            n_states=3
        )
        np.testing.assert_allclose(result["NLL"], synthetic_data["y_true"], rtol=0.05, atol=0.1)



def test_synthetic_2d_covariates(synthetic_data):
    for model in ["gmr", "hmm"]:
        result = evaluate_model(
            model=model,
            y_train=synthetic_data["y_train"],
            y_test=synthetic_data["y_test"],
            x_train=synthetic_data["X_train"],
            x_test=synthetic_data["X_test"],
            n_states=3
        )
        np.testing.assert_allclose(result["NLL"], synthetic_data["y_true"], rtol=0.05, atol=0.1)