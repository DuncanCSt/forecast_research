from typing import Any, Dict, Optional, Sequence
# Goal I want a function which I can pass
def evaluate_model(
    model: str,
    y_train: Sequence[float],
    x_train: Optional[Sequence[float]] = None,
    y_test: Optional[Sequence[float]] = None,
    x_test: Optional[Sequence[float]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Evaluate the chosen model and return a dictionary of results.
    Returns
    {
        forecast_pdf: pd.DataFrame,
        log_likelihood: float,
        SSE: float
    }
    """


