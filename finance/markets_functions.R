get_markets_data <- function(start_date, end_date, period = "weeks") {
  start <- as.Date(start_date)
  end <- as.Date(end_date)

  symbols <- data.frame(
    name = c(
      "TSX_Composite", "Canada_Financials_Index", "Canada_Energy_Index",
      "Canada_Materials_Index", "Canada_Information_Technology_Index",
      "Canada_Utilities_Index", "Canada_Consumer_Staples_Index",
      "Canada_Real_Estate_Index", "Canada_Health_Care_Index",
      "Canada_Consumer_Discretionary_Index", "Canada_Corporate_Bonds",
      "Canada_Government_Bonds", "US_oil_gas", "Global_Materials",
      "US_Government_Bonds", "US_Real_Estate", "SP_500"
    ),
    etf_ticker = c(
      "XIC.TO", "XFN.TO", "XEG.TO", "XMA.TO", "XIT.TO", "XUT.TO",
      "XST.TO", "XRE.TO", "XHC.TO", "XMD.TO", "XCB.TO", "XGB.TO",
      "IEO", "MXI", "GOVT", "VNQ", "SPY"
    ),
    stringsAsFactors = FALSE
  )

  tickers <- symbols$etf_ticker
  friendly_names <- stats::setNames(symbols$name, symbols$etf_ticker)

  close_series <- lapply(tickers, function(tk) {
    data <- quantmod::getSymbols(
      tk,
      src = "yahoo",
      from = start,
      to = end,
      auto.assign = FALSE
    )
    data <- xts::xts(zoo::coredata(data), order.by = as.Date(zoo::index(data)))
    close_col <- grep("Close", colnames(data), value = TRUE, ignore.case = TRUE)[1]
    result <- data[, close_col, drop = FALSE]
    colnames(result) <- tk
    result
  })

  close_data_xts <- do.call(cbind, close_series)
  colnames(close_data_xts) <- friendly_names[colnames(close_data_xts)]

  period_endpoints <- xts::endpoints(close_data_xts, on = period, k = 1)
  period_endpoints <- period_endpoints[period_endpoints > 0]

  filtered_data <- close_data_xts[period_endpoints, , drop = FALSE]
  dates <- as.Date(zoo::index(filtered_data))
  result <- data.frame(
    Date = dates,
    zoo::coredata(filtered_data),
    check.names = FALSE
  )
  row.names(result) <- as.character(dates)
  result
}

percentage_change <- function(data = close_data) {
  date_col <- NULL
  if ("Date" %in% colnames(data)) {
    date_col <- data[["Date"]]
    data <- data[, setdiff(colnames(data), "Date"), drop = FALSE]
  }

  numeric_cols <- vapply(data, is.numeric, logical(1))
  if (!all(numeric_cols)) {
    data <- data[, numeric_cols, drop = FALSE]
  }

  core <- as.matrix(data)
  pct <- core
  pct[1, ] <- 0
  if (nrow(core) > 1) {
    pct[-1, ] <- (core[-1, , drop = FALSE] / core[-nrow(core), , drop = FALSE] - 1) * 100
  }
  result <- as.data.frame(pct, stringsAsFactors = FALSE)
  colnames(result) <- colnames(data)
  if (!is.null(date_col)) {
    result <- cbind(Date = date_col, result)
  }
  row.names(result) <- row.names(data)
  result
}

ln_transform <- function(data = close_data) {
  date_col <- NULL
  if ("Date" %in% colnames(data)) {
    date_col <- data[["Date"]]
    data <- data[, setdiff(colnames(data), "Date"), drop = FALSE]
  }

  numeric_cols <- vapply(data, is.numeric, logical(1))
  if (!all(numeric_cols)) {
    data <- data[, numeric_cols, drop = FALSE]
  }

  core <- as.matrix(data)
  log_core <- log(core)
  delta <- rbind(matrix(0, nrow = 1, ncol = ncol(core)),
               diff(log_core, lag = 1))

  result <- as.data.frame(delta, stringsAsFactors = FALSE)
  colnames(result) <- colnames(data)
  if (!is.null(date_col)) {
    result <- cbind(Date = date_col, result)
  }
  row.names(result) <- row.names(data)
  result
}

train_test_split <- function(data = close_data, perc = 0.8) {
  split_idx <- floor(nrow(data) * perc)
  list(
    train = data[seq_len(split_idx), , drop = FALSE],
    test = data[(split_idx + 1):nrow(data), , drop = FALSE]
  )
}

prepare_tsx_state_data <- function(train_data, hmm_obj) {
  tsx <- dplyr::select(train_data, Date, TSX_Composite)
  states <- factor(hmm_obj$viterbi())
  obspar <- hmm_obj$obs()$par(t = "all")  # array: var × state × time
  means <- vapply(seq_along(states), function(t) {
    obspar[1, states[t], t]
  }, numeric(1))
  sd <- vapply(seq_along(states), function(t) {
    obspar[2, states[t], t]
  }, numeric(1))
  mid <- vapply(seq_along(states), function(t) {
    if (t == 1) {
      return(tsx$TSX_Composite[t])
    } else {
      tsx$TSX_Composite[t-1] * exp(means[t])
    }
  }, numeric(1))
  upper_bound <- vapply(seq_along(states), function(t) {
    if (t == 1) {
      return(tsx$TSX_Composite[t])
    } else {
      tsx$TSX_Composite[t-1] * exp(means[t] + 2 * sd[t])
    }
  }, numeric(1))
  lower_bound <- vapply(seq_along(states), function(t) {
    if (t == 1) {
      return(tsx$TSX_Composite[t])
    } else {
      tsx$TSX_Composite[t-1] * exp(means[t] - 2 * sd[t])
    }
  }, numeric(1))
  dplyr::mutate(
    tsx,
    State = states,
    Segment = cumsum(State != dplyr::lag(State, default = dplyr::first(State))),
    Mean = mid,
    Upper = upper_bound,
    Lower = lower_bound
  )
}

plot_tsx_state_series <- function(tsx_state_data,
                                  show = c("both", "states", "interval")) {
  show <- match.arg(show)
  data_has_interval <- all(c("Mean", "Lower", "Upper") %in% names(tsx_state_data))

  if (show == "interval" && !data_has_interval) {
    stop("Interval plot requested but columns `Mean`, `Lower`, and `Upper` are missing.")
  }

  base_aes <- ggplot2::aes(
    x = Date,
    y = TSX_Composite,
    colour = State,
    group = Segment
  )

  p <- ggplot2::ggplot(tsx_state_data) +
    ggplot2::labs(
      title = "Weekly Close Prices, TSX Composite",
      x = NULL,
      y = "Price (CAD)",
      colour = "State"
    ) +
    ggplot2::theme_light()

  if (show %in% c("both", "states")) {
    p <- p +
      ggplot2::geom_line(base_aes, linewidth = 0.8) +
      ggplot2::geom_point(base_aes, size = 1.2) +
      ggplot2::scale_colour_manual(values = hmmTMB_cols)
  } else {
    # interval only: colour legend comes from ribbon line scale later
    p <- p + ggplot2::scale_colour_manual(values = hmmTMB_cols)
  }

  if (show %in% c("both", "interval") && data_has_interval) {
    interval_aes <- ggplot2::aes(
      x = Date,
      ymin = Lower,
      ymax = Upper,
      fill = State,
      group = Segment
    )
    mean_aes <- ggplot2::aes(
      x = Date,
      y = Mean,
      colour = State,
      group = Segment
    )

    p <- p +
      ggplot2::geom_ribbon(
        interval_aes,
        alpha = 0.5,
        linewidth = 0,
        inherit.aes = FALSE
      ) +
      ggplot2::geom_line(
        mean_aes,
        linewidth = 0.9,
        linetype = "dashed",
        inherit.aes = FALSE
      ) +
      ggplot2::scale_fill_manual(values = hmmTMB_cols, guide = "none")
  }

  if (show == "interval") {
    p <- p + ggplot2::guides(colour = "none")
  }

  p
}
