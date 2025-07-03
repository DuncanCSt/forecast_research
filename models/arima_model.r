# arima_model.R
# ------------------------------------------------------------------------------
# Helper functions to fit an ARIMA / SARIMA / SARIMAX model and return forecasts.
# Intended to be sourced from Python via *rpy2* or from other R scripts.
# ------------------------------------------------------------------------------

#' Build an ARIMA (or SARIMA/SARIMAX) model
#'
#' @param y            Numeric vector or ts object. Target series.
#' @param order        Optional length‑3 integer vector c(p, d, q). If NULL, use
#'                     `forecast::auto.arima()` to find the best order.
#' @param seasonal     Logical. If FALSE, seasonal terms are suppressed even if
#'                     frequency > 1.
#' @param seasonal_order Optional length‑3 integer vector c(P, D, Q) for SARIMA.
#'                      Ignored if `seasonal = FALSE`.
#' @param xreg         Optional matrix / data.frame of exogenous regressors
#'                     aligned with `y` (rows = length(y)).
#' @param frequency    Optional integer. If provided, `y` will be converted to a
#'                     `ts` object with the given frequency.
#' @param ...          Extra arguments passed to `auto.arima()` or `Arima()`.
#' @return             An object of class `Arima`.
#' @export
build_arima <- function(y,
                        order = NULL,
                        seasonal = TRUE,
                        seasonal_order = NULL,
                        xreg = NULL,
                        frequency = NULL,
                        ...) {
  if (!requireNamespace("forecast", quietly = TRUE)) {
    stop("Package 'forecast' is required but not installed.")
  }
  if (!is.null(frequency)) {
    y <- stats::ts(y, frequency = frequency)
  }
  if (is.null(order) && is.null(seasonal_order)) {
    fit <- forecast::auto.arima(y, xreg = xreg, seasonal = seasonal, ...)
  } else {
    fit <- forecast::Arima(y,
                           order = if (is.null(order)) c(0, 0, 0) else order,
                           seasonal = if (seasonal) list(order = seasonal_order) else list(order = c(0, 0, 0)),
                           xreg = xreg,
                           ...)
  }
  return(fit)
}

#' Forecast with an ARIMA model
#'
#' @param y_train      Numeric vector. Historical target series used for fitting.
#' @param steps        Integer. Forecast horizon.
#' @param xreg_train   Matrix / data.frame of in‑sample exogenous regressors.
#' @param xreg_future  Matrix / data.frame of *future* exogenous regressors for
#'                     the forecast horizon. Must have `steps` rows.
#' @param frequency    Passed to `build_arima()`.
#' @param ...          Additional arguments forwarded to `build_arima()`.
#' @return             A list with elements:
#'                       • `forecast` – numeric vector length = `steps`
#'                       • `model`    – the fitted `Arima` object
#' @export
forecast_arima <- function(y_train,
                           steps,
                           xreg_train = NULL,
                           xreg_future = NULL,
                           frequency = NULL,
                           ...) {
  fit <- build_arima(y = y_train,
                     xreg = xreg_train,
                     frequency = frequency,
                     ...)
  fc  <- forecast::forecast(fit, h = steps, xreg = xreg_future)
  return(list(forecast = as.numeric(fc$mean), model = fit))
}
