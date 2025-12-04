# Gaussian Mixture Model (GMM) implementation for forecasting research
from typing import Any, Dict, Optional
import numpy as np
import numpy.typing as npt
from gmr.gmm import GMM
from sklearn.mixture import GaussianMixture

def gmr_evaluate(
    y_train: npt.NDArray[np.float64],
    y_test: npt.NDArray[np.float64],
    x_train: Optional[npt.NDArray[np.float64]] = None,
    x_test: Optional[npt.NDArray[np.float64]] = None,
    n_states: int = 3,
) -> Dict[str, Any]:
    """
    Train a Gaussian Mixture Model (GMM) on the training data and evaluate it on the test data.
    Returns
    {
        NLL: Array,
        SNLL: float,
    }
    """
    
    # Prepare training data
    if x_train is None:
        X_train = np.array(y_train).reshape(-1, 1)
    else:
        if x_train.ndim == 1:
            X_train = np.array(x_train).reshape(-1, 1)
        else:
            X_train = np.array(x_train).reshape(-1, x_train.shape[1])
        X_train = np.hstack((X_train, np.array(y_train).reshape(-1, 1)))
        if x_test is None:
            raise ValueError("x_test must be provided if x_train is provided.")

    # Prepare test data
    if x_test is None:
        # 1D case - only y values using standard GMM
        gmm = GaussianMixture(n_components=n_states)
        gmm.fit(X_train)

        pdf_values = np.exp(gmm.score_samples(np.array(y_test).reshape(-1, 1)))
        nll = -np.log(pdf_values + 1e-10)  # Add small value to avoid log(0)
        snll = np.sum(nll)
    else:
        # Case with covariates using conditional GMM
        gmm = GMM(n_components=n_states)
        gmm.from_samples(X_train)

        X_test = np.array(x_test)
        X_test = np.hstack((X_test.reshape(-1, X_test.shape[1]), np.array(y_test).reshape(-1, 1)))
        nll = []
        for row in X_test:
            cond_gmm = gmm.condition(np.array(range(X_test.shape[1] - 1)), row[:-1])
            pdf_value = cond_gmm.to_probability_density(np.array(row[-1]).reshape(1, -1))
            nll.append(-np.log(pdf_value + 1e-10))
        nll = np.array(nll).reshape(-1)
        snll = np.sum(nll)

    return {
        "NLL": nll,
        "SNLL": snll,
    }

