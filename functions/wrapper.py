from typing import Any, Dict, Optional, Sequence, Literal
import numpy as np
# Goal I want a function which I can pass

def evaluate_model(
    model: Literal["gmr", "hmm"],
    y_train: Sequence[float],
    y_test: Sequence[float],
    x_train: Optional[Sequence[Sequence[float]]] = None,
    x_test: Optional[Sequence[Sequence[float]]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Evaluate the chosen model and return a dictionary of results.
    Returns
    {
        forecast_pdf: pd.DataFrame,
        RMSE: float,
        MAE: float
    }
    """
    params = dict(kwargs)

    if model == "gmr":
        from functions.models.gmr import gmr_evaluate

        if 'n_states' not in params:
            raise ValueError("n_states must be provided for GMR model.")

        n_states = params.pop('n_states')
        return gmr_evaluate(
            y_train=y_train,
            x_train=x_train,
            y_test=y_test,
            x_test=x_test,
            n_states=n_states,
        )

    elif model == "hmm":
        params = dict(kwargs)

        if 'n_states' not in params:
            raise ValueError("n_states must be provided for HMM model.")

        import rpy2.robjects as ro

        ro.r.source("functions/models/hmm.R")
        hmm_evaluate = ro.globalenv['hmm_evaluate']

        def _py_none_to_r_null(obj):
            """Convert Python None → R NULL for rpy2 calls."""
            return ro.NULL if obj is None else obj
        
        def _py_matrix_to_r_matrix(obj):
            """Convert Python 2D list/array to R matrix."""
            if obj is None:
                return ro.NULL
            arr = np.array(obj)
            if arr.ndim == 1:
                nrow = arr.shape[0]
                ncol = 1
            else:
                nrow, ncol = arr.shape
            # Convert flattened array to list to enable rpy2 conversion
            r_matrix = ro.r['matrix'](ro.FloatVector(arr.flatten().tolist()), nrow=nrow, ncol=ncol)
            return r_matrix

        # Parse kwargs
        y_train = _py_matrix_to_r_matrix(y_train)
        y_test = _py_matrix_to_r_matrix(y_test)
        x_train = _py_matrix_to_r_matrix(x_train)
        x_test = _py_matrix_to_r_matrix(x_test)
        n_states = params.pop("n_states")
        hid_formula = _py_none_to_r_null(params.pop("hid_formula", None))
        obs_formula = _py_none_to_r_null(params.pop("obs_formula", None))

        result = hmm_evaluate(
            y_train,
            y_test,
            x_train,
            x_test,
            n_states,
            hid_formula,
            obs_formula
        )

        nll = np.array(result.rx2("NLL"), dtype=float)
        snll = float(result.rx2("SNLL")[0])

        return {
            "NLL": nll,
            "SNLL": snll,
        }

    else:
        raise ValueError(f"Unsupported model type: {model}")
