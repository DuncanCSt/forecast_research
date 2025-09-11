# hmm_model.R
################################################################################
# Helper functions to fit an HMM with hmmTMB and return forecast summaries.
# Designed to be sourced from Python via rpy2, mirroring arima_model.R shape.
################################################################################

#' Fit an HMM using hmmTMB
#'
#' @param y Numeric vector; univariate observation series (named 'y' internally).
#' @param n_states Integer; number of hidden states.
#' @param formula Transition model formula for MarkovChain (character or formula).
#'        Default "~ 1" (time-homogeneous transitions).
#' @param xreg Optional data.frame/matrix of covariates aligned with y. Columns
#'        are added to the training data; include them in `formula` if needed.
#' @param obs_dist Character; observation distribution for 'y'. Default "norm".
#' @param init_par Optional named list for Observation$new(par=...), otherwise
#'        use obs$suggest_initial().
#' @return Fitted HMM (R6 object)
fit_hmm <- function(y,
                    n_states = 2,
                    hid_formula = NULL,
                    obs_formula = NULL,
                    xreg = NULL,
                    obs_dist = "norm",
                    init_par = list(mean = c(0, 1), sd = c(1, 1))
) {

  if (!requireNamespace("hmmTMB", quietly = TRUE)) {
    stop("Package 'hmmTMB' is required but not installed.")
  }

  # Build training data frame
  n <- length(y)
  df <- data.frame(ID = rep(1, n), y = as.numeric(y))
  if (!is.null(xreg)) {
    xreg <- as.data.frame(xreg)
    if (nrow(xreg) != n) stop("xreg must have same number of rows as y")
    df <- cbind(df, xreg)
  }

  # Parse formula (accept character or formula)
  if (!is.null(hid_formula)) {
    hid_fml <- stats::as.formula(hid_formula)
  } else {
    hid_fml <- NULL
  }
  if (!is.null(obs_formula)) {
    obs_fml <- list(y = obs_formula)
  } else {
    obs_fml <- NULL
  }

  # Hidden Markov chain with optional covariates in transitions
  hid_model <- hmmTMB::MarkovChain$new(
    data = df,
    n_states = as.integer(n_states),
    formula = hid_fml
  )
  print(list(y = obs_dist))
  # Observation model for 'y'
  obs_model <- hmmTMB::Observation$new(
    data = df,
    n_states = as.integer(n_states),
    dists = list(y = obs_dist),
    par = list(y = init_par),
    formula = obs_fml
  )
  obs_model$update_par(par = obs_model$suggest_initial())

  hmm <- hmmTMB::HMM$new(hid = hid_model, obs = obs_model)
  hmm$fit(silent = TRUE)
  hmm
}


#' Forecast with a fitted hmmTMB model
#'
#' @param y_train Numeric vector; training series.
#' @param steps Integer; forecast horizon.
#' @param xreg_train Optional covariates for training (rows = length(y_train)).
#' @param xreg_future Optional covariates for forecast horizon (rows = steps).
#' @param n_states Integer; number of hidden states.
#' @param formula Character or formula for transitions; default "~ 1".
#' @param obs_dist Character observation distribution for 'y'; default "norm".
#' @param starting_state_distribution "last", "stationary", or numeric vector.
#' @param eval_range Named list giving evaluation grid for 'y' (optional).
#' @return list with elements:
#'   - forecast: numeric vector (length = steps) of unconditional means
#'   - sd: numeric vector (length = steps) of unconditional standard deviations
#'   - model: fitted HMM object
#' @export
forecast_hmm <- function(y_train,
                         steps,
                         xreg_train = NULL,
                         xreg_future = NULL,
                         n_states = 2,
                         hid_formula = NULL,
                         obs_formula = NULL,
                         obs_dist = "norm",
                         starting_state_distribution = "last",
                         eval_range = NULL) {

  hmm <- fit_hmm(
    y        = y_train,
    n_states = n_states,
    hid_formula = hid_formula,
    obs_formula = obs_formula,
    xreg     = xreg_train,
    obs_dist = obs_dist
  )

  # Build forecast covariate frame if provided; otherwise use n = steps
  if (!is.null(xreg_future)) {
    xreg_future <- as.data.frame(xreg_future)
    if (nrow(xreg_future) != as.integer(steps))
      stop("xreg_future rows must equal `steps`.")
    forecast_df <- cbind(data.frame(ID = rep(1, nrow(xreg_future))), xreg_future)
    fc <- hmmTMB::Forecast$new(
      hmm = hmm,
      forecast_data = forecast_df,
      preset_eval_range = eval_range,
      starting_state_distribution = starting_state_distribution
    )
  } else {
    fc <- hmmTMB::Forecast$new(
      hmm = hmm,
      n = as.integer(steps),
      preset_eval_range = eval_range,
      starting_state_distribution = starting_state_distribution
    )
  }

  # Extract unconditional predictive pdfs on grid for 'y'
  dists_list <- fc$forecast_dists()
  if (is.null(dists_list[["y"]])) {
    stop("Observation variable expected to be named 'y' in this wrapper.")
  }
  pdf_mat <- dists_list[["y"]]              # dim: |grid| x steps
  x_grid  <- fc$eval_range()$y               # vector of length |grid|

  # Numerical moments per step
  Z  <- colSums(pdf_mat)
  mu <- colSums(t(t(pdf_mat) * x_grid)) / Z
  # For variance, use E[(X - mu)^2]
  var <- vapply(seq_len(ncol(pdf_mat)), function(i) {
    w <- pdf_mat[, i] / Z[i]
    sum((x_grid - mu[i])^2 * w)
  }, numeric(1))
  sd <- sqrt(var)

  list(
    forecast = as.numeric(mu),
    sd       = as.numeric(sd),
    model    = hmm
  )
}
