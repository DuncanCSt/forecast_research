library(dplyr)
library(lubridate)

#' This function retrieves closing prices for a predefined set of market indices and ETFs
#' from Yahoo Finance, combines them into a single data frame, aggregates to the specified
#' period endpoints (e.g., weekly or monthly means), and drops columns with excessive
#' missing values.
#'
#' @param start_date The start date for data retrieval (as character or Date).
#' @param end_date The end date for data retrieval (as character or Date).
#' @param period Aggregation period: one of \code{"weeks"}, \code{"months"}, or
#'   \code{"none"} (no aggregation, daily). Defaults to \code{"weeks"}.
#' @param plot Logical, when TRUE (default) a line plot of the fetched series is displayed.
#' @param savefig Optional path to save the plot (supports png, jpg, tiff, bmp, pdf).
#'   Ignored if \code{plot = FALSE}.
#' @param na_threshold Maximum allowed fraction of missing values per column
#'   (excluding Date). Columns with NA fraction above this threshold are dropped.
#'   Defaults to 0.15 (15\%).
#'
#' @return A data frame with a \code{Date} column and closing prices for each market.
#'         Row names are set to the Date; columns with more than \code{na_threshold}
#'         missing values are removed.
get_markets_data <- function(start_date,
                             end_date,
                             period       = c("weeks", "months", "none"),
                             plot         = TRUE,
                             savefig      = NULL,
                             na_threshold = 0.15) {

  period <- match.arg(period)

  # Date handling
  start <- as.Date(start_date)
  end   <- as.Date(end_date)

  # Metadata: friendly names and tickers
  symbols <- data.frame(
    name = c(
      "TSX_Composite", "Canada_Financials_Index", "Canada_Energy_Index",
      "Canada_Materials_Index", "Canada_Information_Technology_Index",
      "Canada_Utilities_Index", "Canada_Consumer_Staples_Index",
      "Canada_Real_Estate_Index", "Canada_Health_Care_Index",
      "Canada_Consumer_Discretionary_Index", "Canada_Corporate_Bonds",
      "Canada_Government_Bonds", "US_oil_gas", "Global_Materials",
      "US_Real_Estate", "SP_500", "Crude_Oil_Futures", "Gold_Futures",
      "Copper_Futures"
    ),
    etf_ticker = c(
      "XIC.TO", "XFN.TO", "XEG.TO", "XMA.TO", "XIT.TO", "XUT.TO",
      "XST.TO", "XRE.TO", "XHC.TO", "XMD.TO", "XCB.TO", "XGB.TO",
      "IEO", "MXI", "VNQ", "SPY", "CL=F", "GC=F", "HG=F"
    ),
    stringsAsFactors = FALSE
  )

  # Helper: fetch close prices for one ticker as data frame
  fetch_close_df <- function(ticker, name, start, end) {
    raw_data <- tryCatch(
      quantmod::getSymbols(
        ticker,
        src         = "yahoo",
        from        = start,
        to          = end,
        auto.assign = FALSE
      ),
      error = function(e) {
        warning(sprintf("Failed to fetch data for %s: %s", ticker, e$message))
        return(NULL)
      }
    )

    if (is.null(raw_data)) return(NULL)

    # Identify Close column
    close_col <- grep("Close", colnames(raw_data),
                      value = TRUE, ignore.case = TRUE)[1]

    if (is.na(close_col)) {
      warning(sprintf("No Close column found for %s", ticker))
      return(NULL)
    }

    df <- data.frame(
      Date = as.Date(zoo::index(raw_data))
    )
    df[[name]] <- as.numeric(raw_data[, close_col])
    df
  }

  # Fetch all series
  close_df_list <- mapply(
    fetch_close_df,
    ticker = symbols$etf_ticker,
    name   = symbols$name,
    MoreArgs = list(start = start, end = end),
    SIMPLIFY = FALSE
  )

  close_df_list <- close_df_list[!vapply(close_df_list, is.null, logical(1))]

  if (length(close_df_list) == 0L) {
    stop("No series could be fetched. Check tickers or network connection.")
  }

  # Combine into single data frame
  data_df <- close_df_list %>%
    purrr::reduce(dplyr::full_join, by = "Date") %>%
    dplyr::arrange(Date)

  # Aggregate to requested period
  aggregate_period <- function(df, period) {
    if (period == "none") {
      return(df)
    }

    if (period == "weeks") {
      return(
        df %>%
          dplyr::mutate(
            WeekEndFriday = lubridate::floor_date(Date,
                                                  unit = "week",
                                                  week_start = 1) + 4
          ) %>%
          dplyr::group_by(WeekEndFriday) %>%
          dplyr::summarise(
            dplyr::across(
              dplyr::where(is.numeric),
              ~ mean(.x, na.rm = TRUE)
            ),
            .groups = "drop"
          ) %>%
          dplyr::rename(Date = WeekEndFriday)
      )
    }

    if (period == "months") {
      return(
        df %>%
          dplyr::mutate(
            MonthEnd = lubridate::ceiling_date(Date, unit = "month") - 1
          ) %>%
          dplyr::group_by(MonthEnd) %>%
          dplyr::summarise(
            dplyr::across(
              dplyr::where(is.numeric),
              ~ mean(.x, na.rm = TRUE)
            ),
            .groups = "drop"
          ) %>%
          dplyr::rename(Date = MonthEnd)
      )
    }

    stop("Unsupported period: ", period)
  }

  result <- aggregate_period(data_df, period)

  # Drop columns with too many NAs
  if (ncol(result) > 1L) {
    num_cols <- setdiff(names(result), "Date")
    na_fraction <- colMeans(is.na(result[, num_cols, drop = FALSE]))

    cols_to_drop <- names(na_fraction)[na_fraction > na_threshold]
    if (length(cols_to_drop) > 0L) {
      message(
        "Dropping columns with NA fraction > ", na_threshold, ": ",
        paste(cols_to_drop, collapse = ", ")
      )
      result <- result[, setdiff(names(result), cols_to_drop), drop = FALSE]
    }
  }

  # Set rownames to Date
  rownames(result) <- as.character(result$Date)

  # Plotting (placeholder; implement as needed)
  if (plot) {
    # Example: Convert to long format and use ggplot2 for line plot
  # Reshape data for ggplot
  result_long <- result %>%
    pivot_longer(-Date, names_to = "Index", values_to = "Close") %>%
    group_by(Index) %>%
    mutate(Close = (Close / first(Close) - 1) * 100)

  # Plot using ggplot2
  p <- ggplot(result_long, aes(x = Date, y = Close, color = Index)) +
    geom_line() +
    labs(title = "Normalized Weekly Close Prices (Percentage Change)", x = "Date", y = "Percentage Change (%)") +
    theme_minimal()

  print(p)
  if (!is.null(savefig)) ggplot2::ggsave(savefig, p)
  }

  return(result)
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

ln_transform <- function(data = close_data, previous_row = NULL) {

  row_names <- row.names(data)

  # Prepend previous row if provided, else duplicate first row
  if (!is.null(previous_row)) {
    data <- rbind(previous_row, data)
  } else {
    data <- rbind(head(data, 1), data)
  }

  numeric_cols <- vapply(data, is.numeric, logical(1))
  if (!all(numeric_cols)) {
    data <- data[, numeric_cols, drop = FALSE]
  }

  core <- as.matrix(data)
  log_core <- log(core)
  log_diff <- diff(log_core, lag = 1)

  result <- as.data.frame(log_diff, stringsAsFactors = FALSE)
  colnames(result) <- colnames(data)
  row.names(result) <- row_names
  result
}

plot_na_matrix <- function(data, title = "NA Values in data", savefig = NULL) {
  na_data_long <- is.na(data) %>%
    as.data.frame() %>%
    mutate(Date = rownames(data)) %>%
    pivot_longer(
      cols = -Date,
      names_to = "Index",
      values_to = "is_na"
    ) %>%
    mutate(
      Date    = as.Date(Date),
      NA_Flag = ifelse(is_na, "NA", "Non-NA")
    )

  plot_obj <- ggplot(na_data_long, aes(x = Date, y = Index, fill = NA_Flag)) +
    geom_tile() +
    scale_fill_manual(
      values = c("Non-NA" = "white", "NA" = "red"),
      name   = "NA Flag"
    ) +
    labs(title = title, x = "Date", y = "Index") +
    theme_minimal()

  device_opened <- open_save_device(
    savefig = savefig,
    width = 900,
    height = 600
  )
  if (device_opened) {
    on.exit(grDevices::dev.off(), add = TRUE)
    print(plot_obj)
    invisible(plot_obj)
  } else {
    plot_obj
  }
}

train_test_split <- function(data, training_period, testing_period, samples) {
  if (training_period + samples*testing_period > nrow(data)) {
    warning("Insufficent data to produce non-overlaping test sets.")
  }
  n_rows <- nrow(data)
  split_index <- as.integer(seq(
    from = training_period,
    to = n_rows - testing_period,
    length.out = samples
  ))

  train_data_list <- vector("list", samples)
  test_data_list  <- vector("list", samples)

  for (i in seq_len(samples)) {
    idx <- split_index[i]
    train_data_list[[i]] <- data[(idx - training_period + 1):idx, , drop = FALSE]
    test_data_list[[i]]  <- data[(idx + 1):(idx + testing_period), , drop = FALSE]
  }

  list(train = train_data_list, test = test_data_list)
}

open_save_device <- function(savefig, width, height, res = 120) {
  if (is.null(savefig)) {
    return(FALSE)
  }

  dir_path <- dirname(savefig)
  if (!dir.exists(dir_path) && dir_path != ".") {
    dir.create(dir_path, recursive = TRUE, showWarnings = FALSE)
  }

  ext <- tolower(tools::file_ext(savefig))
  if (ext %in% c("png", "jpeg", "jpg", "tiff", "bmp")) {
    grDevices::png(
      filename = savefig,
      width = width,
      height = height,
      res = res
    )
    return(TRUE)
  }

  if (ext == "pdf") {
    grDevices::pdf(
      file = savefig,
      width = width / res,
      height = height / res
    )
    return(TRUE)
  }

  warning("Unrecognized file extension for `savefig`; expected png, jpg, jpeg, tiff, bmp, or pdf. Skipping save.")
  FALSE
}

plot_train_test_samples <- function(splits, obs_train, obs_test, y_column = "TSX_composite", savefig = NULL) {
  train_list <- if (missing(obs_train)) splits$train else obs_train
  test_list  <- if (missing(obs_test)) splits$test else obs_test
  samples    <- length(train_list)
  if (samples == 0L) stop("No samples supplied")

  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par), add = TRUE)

  device_opened <- open_save_device(
    savefig = savefig,
    width = 900,
    height = 150 * max(1, samples)
  )

  draw_panels <- function() {
    par(mfrow = c(samples, 1), mar = c(3, 4, 2, 1))
    for (i in seq_len(samples)) {
      train_df <- train_list[[i]]
      test_df  <- test_list[[i]]

      train_dates <- as.Date(rownames(train_df))
      test_dates  <- as.Date(rownames(test_df))
      train_y     <- train_df[[y_column]]
      test_y      <- test_df[[y_column]]

      x_range <- range(c(train_dates, test_dates), na.rm = TRUE)
      y_range <- range(c(train_y, test_y), na.rm = TRUE)

      plot(train_dates, train_y,
           type = "n",
           xlim = x_range,
           ylim = y_range,
           main = paste("Sample", i),
           xlab = "Date",
           ylab = y_column)

      lines(train_dates, train_y, col = "steelblue", lwd = 2)
      lines(test_dates, test_y, col = "firebrick", lwd = 2)
      legend("topleft", legend = c("Train", "Test"),
           col = c("steelblue", "firebrick"), lwd = 2, bty = "n")
    }
  }

  draw_panels()

  if (device_opened) {
    grDevices::dev.off()
    par(old_par)
    draw_panels()
  }
}

prepare_tsx_state_data <- function(train_data, hmm_obj) {
  train_df <- as.data.frame(train_data, stringsAsFactors = FALSE)
  tsx_values <- train_df[["TSX_Composite"]]
  if (is.null(tsx_values)) {
    stop("Column `TSX_Composite` not found in training data.")
  }

  tsx_dates <- as.Date(rownames(train_df))
  if (any(is.na(tsx_dates))) {
    stop("Row names of `train_data` must be coercible to Date.")
  }

  states <- hmm_obj$viterbi()
  obspar <- hmm_obj$obs()$par(t = "all")  # array: var × state × time
  means <- vapply(seq_along(states), function(t) obspar[1, states[t], t], numeric(1))
  sds <- vapply(seq_along(states), function(t) obspar[2, states[t], t], numeric(1))

  n <- length(states)
  mean_vals <- numeric(n)
  upper_vals <- numeric(n)
  lower_vals <- numeric(n)

  if (n > 0) {
    mean_vals[1] <- tsx_values[1]
    upper_vals[1] <- tsx_values[1]
    lower_vals[1] <- tsx_values[1]
  }

  if (n > 1) {
    for (t in 2:n) {
      prev_val <- tsx_values[t - 1]
      mean_vals[t] <- prev_val * exp(means[t])
      upper_vals[t] <- prev_val * exp(means[t] + 2 * sds[t])
      lower_vals[t] <- prev_val * exp(means[t] - 2 * sds[t])
    }
  }

  state_factor <- factor(states)
  change_points <- c(FALSE, state_factor[-1] != state_factor[-length(state_factor)])
  segments <- cumsum(change_points) + 1

  result <- data.frame(
    Date = tsx_dates,
    TSX_Composite = tsx_values,
    State = state_factor,
    Segment = segments,
    Mean = mean_vals,
    Upper = upper_vals,
    Lower = lower_vals,
    stringsAsFactors = FALSE
  )
  row.names(result) <- as.character(tsx_dates)
  result
}

plot_tsx_state_series <- function(tsx_state_data,
                                  show = c("both", "states", "interval"),
                                  savefig = NULL) {
  show <- match.arg(show)
  data_has_interval <- all(c("Mean", "Lower", "Upper") %in% names(tsx_state_data))

  if (show == "interval" && !data_has_interval) {
    stop("Interval plot requested but columns `Mean`, `Lower`, and `Upper` are missing.")
  }

  tsx_state_data <- as.data.frame(tsx_state_data)
  tsx_state_data$PlotDate <- as.Date(rownames(tsx_state_data))

  base_aes <- ggplot2::aes(
    x = PlotDate,
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
      x = PlotDate,
      ymin = Lower,
      ymax = Upper,
      fill = State,
      group = Segment
    )
    mean_aes <- ggplot2::aes(
      x = PlotDate,
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

  device_opened <- open_save_device(
    savefig = savefig,
    width = 1100,
    height = 600
  )
  if (device_opened) {
    on.exit(grDevices::dev.off(), add = TRUE)
    print(p)
    invisible(p)
  } else {
    p
  }
}

fit_hmm <- function(
    n_states,
    data,
    obs_name = "TSX_Composite",
    hid_formula = as.formula("~1"),
    horseshoe = FALSE,
    init = NULL
  ) {
  hid_model <- hmmTMB::MarkovChain$new(
    data = data,
    n_states = n_states,
    formula = hid_formula,
    initial_state = "estimated"
    #horseshoe = horseshoe
  )

  obs_model <- hmmTMB::Observation$new(
    data = data,
    n_states = n_states,
    dists = setNames(list("norm"), obs_name),
    par = setNames(
      list(
        list(
          mean = seq(-0.02, 0.01, length.out = n_states),
          sd = seq(0.1, 0.01, length.out = n_states)
        )
      ),
      obs_name
    ),
  )
  if (!is.null(init)) {
    # If a model is passed return the model on updated data without re-training.
    hmm <- hmmTMB::HMM$new(init = init, hid = hid_model, obs = obs_model)
    return(hmm)
  }

  hmm <- hmmTMB::HMM$new(hid = hid_model, obs = obs_model)
  hmm$fit(silent = TRUE)
  hmm
}

plot_train_test_zoomed <- function(train_data,
                                 test_data,
                                 observation,
                                 zoom_range = 20,
                                 title = NULL,
                                 savefig = NULL) {
  train_df <- as.data.frame(train_data, stringsAsFactors = FALSE)
  test_df  <- as.data.frame(test_data, stringsAsFactors = FALSE)
  train_df$Type <- "Train"
  test_df$Type  <- "Test"

  combined_data <- rbind(train_df, test_df)
  combined_data$PlotDate <- as.Date(rownames(combined_data))

  zoom_data <- tail(combined_data, zoom_range)

  p1 <- ggplot(
    combined_data,
    aes(x = PlotDate, y = .data[[observation]], colour = Type)
  ) +
    geom_line() +
    labs(
      title = paste("Train and Test Data for", observation),
      x = "Date",
      y = observation
    ) +
    theme_minimal()

  p2 <- ggplot(
    zoom_data,
    aes(x = PlotDate, y = .data[[observation]], colour = Type)
  ) +
    geom_line() +
    labs(
      title = paste("Zoomed In: Last", zoom_range, "Rows"),
      x = "Date",
      y = observation
    ) +
    theme_minimal()

  if (is.null(title)) {
    layout_grob <- gridExtra::arrangeGrob(p1, p2, ncol = 2)
  } else {
    layout_grob <- gridExtra::arrangeGrob(
      p1,
      p2,
      ncol = 2,
      top = grid::textGrob(
        title,
        gp = grid::gpar(fontface = "bold")
      )
    )
  }

  device_opened <- open_save_device(
    savefig = savefig,
    width = 1100,
    height = 450
  )

  draw_layout <- function() {
    grid::grid.newpage()
    grid::grid.draw(layout_grob)
  }

  if (device_opened) {
    tryCatch(
      {
        draw_layout()
      },
      finally = grDevices::dev.off()
    )
  }

  draw_layout()
  invisible(layout_grob)
}

plot_tsx_state_overview <- function(hmm,
                                    train_data,
                                    observation = "TSX_Composite",
                                    show = c("both", "states", "interval"),
                                    title = NULL,
                                    savefig = NULL) {
  show <- match.arg(show)

  tsx_state <- prepare_tsx_state_data(train_data, hmm)
  tsx_plot  <- plot_tsx_state_series(tsx_state, show = show)
  ts_plot   <- hmm$plot_ts(observation)

  if (is.null(title)) {
    layout_grob <- gridExtra::arrangeGrob(ts_plot, tsx_plot, ncol = 2)
  } else {
    layout_grob <- gridExtra::arrangeGrob(
      ts_plot,
      tsx_plot,
      ncol = 2,
      top = grid::textGrob(
        title,
        gp = grid::gpar(fontface = "bold")
      )
    )
  }

  device_opened <- open_save_device(
    savefig = savefig,
    width = 1400,
    height = 450
  )

  draw_layout <- function() {
    grid::grid.newpage()
    grid::grid.draw(layout_grob)
  }

  if (device_opened) {
    tryCatch(
      {
        draw_layout()
      },
      finally = grDevices::dev.off()
    )
  }

  draw_layout()
  invisible(layout_grob)
}

plot_ridge_data <- function(
  forecast_data,
  true_values,
  forecast_steps,
  forecast_timesteps,
  title = "Forecast of probability density vs true observations",
  y_label = "Price",
  savefig = NULL
) {
  scale_factor <- 1/max(forecast_data[, forecast_timesteps])

  ridge_data <- data.frame(
    x = rep(forecast_steps, length(forecast_timesteps)),
    time = factor(rep(forecast_timesteps, each = length(forecast_steps)), levels = forecast_timesteps),
    density = c(forecast_data[, forecast_timesteps]*scale_factor)
  )
  true_df <- data.frame(
    time = factor(forecast_timesteps, levels = forecast_timesteps),
    x = true_values[forecast_timesteps]
  )

  p <- ggplot() +
    geom_ridgeline(
      data = ridge_data,
      aes(x = x, y = time, height = density, fill = "Forecasted"),
      alpha = 0.3
    ) +
    geom_point(
      data = true_df,
      aes(x = x, y = time, colour = "True"),
      size = 2,
      inherit.aes = FALSE
    ) +
    scale_fill_manual(name = "Type", values = c("Forecasted" = "blue")) +
    scale_colour_manual(name = "Type", values = c("True" = "red")) +
    labs(
      title = title,
      x = y_label,
      y = "Time Step"
    ) +
    theme_minimal() +
    guides(fill = guide_legend(override.aes = list(alpha = 0.3))) +
    # coord_cartesian(xlim = c(0, 10)) +
    coord_flip()

  options(repr.plot.width = 11, repr.plot.height = 4)  # Adjust figure size

  device_opened <- open_save_device(
    savefig = savefig,
    width = 1100,
    height = 400
  )

  if (device_opened) {
    print(p)
    grDevices::dev.off()
  }

  print(p)
  invisible(p)
}

extract_hid_fe <- function(hmm) {
  hid_fe <- hmm$hid()$coeff_fe()
  hid_df <- data.frame(value = hid_fe, stringsAsFactors = FALSE)

  rn <- rownames(hid_df)
  split_names <- strsplit(rn, "\\.")
  hid_df$transition <- vapply(split_names, function(x) x[[1]], character(1), USE.NAMES = FALSE)
  hid_df$covariate <- vapply(
    split_names,
    function(x) if (length(x) >= 2) x[[2]] else "",
    character(1),
    USE.NAMES = FALSE
  )

  transitions <- unique(hid_df$transition)
  covariates <- unique(hid_df$covariate)

  result_matrix <- matrix(
    NA_real_,
    nrow = length(transitions),
    ncol = length(covariates),
    dimnames = list(transitions, covariates)
  )

  for (i in seq_len(nrow(hid_df))) {
    tr <- hid_df$transition[i]
    cv <- hid_df$covariate[i]
    result_matrix[tr, cv] <- hid_df$value[i]
  }

  as.data.frame(result_matrix, stringsAsFactors = FALSE)
}

hid_fe_to_long <- function(hid_fe, model_label) {
  hid_fe %>%
    rownames_to_column("transition") %>%
    pivot_longer(-transition, names_to = "covariate", values_to = "value") %>%
    mutate(model = model_label)
}

compute_log_loss_traces <- function(
  true_forecast,
  forecast_dist,
  forecast_range,
  observation
) {
  n_samples <- length(forecast_dist)
  if (n_samples == 0L) {
    stop("No forecast distributions supplied.")
  }
  if (length(true_forecast) != n_samples || length(forecast_range) != n_samples) {
    stop("`true_forecast`, `forecast_dist`, and `forecast_range` must have the same length.")
  }

  extract_observation <- function(x) {
    if (is.data.frame(x)) {
      if (!observation %in% names(x)) {
        stop(sprintf("Column `%s` not found in supplied `true_forecast` data.", observation))
      }
      return(x[[observation]])
    }
    as.numeric(x)
  }

  compute_trace <- function(true_vec, dist_mat, eval_steps) {
    dist_mat <- as.matrix(dist_mat)
    n_forecasts <- ncol(dist_mat)
    if (length(true_vec) < n_forecasts) {
      stop("Length of `true_forecast` entries must match the number of forecast horizons.")
    }
    sapply(
      seq_len(n_forecasts),
      function(i) {
        approx_val <- stats::approx(
          x = eval_steps,
          y = dist_mat[, i],
          xout = true_vec[i]
        )$y
        -log(approx_val)
      }
    )
  }

  log_traces <- vector("list", n_samples)
  for (i in seq_len(n_samples)) {
    true_vec <- extract_observation(true_forecast[[i]])
    dist_mat <- forecast_dist[[i]]
    eval_steps <- forecast_range[[i]]
    log_traces[[i]] <- compute_trace(true_vec, dist_mat, eval_steps)
  }

  sample_names <- names(forecast_dist)
  if (is.null(sample_names) || any(sample_names == "")) {
    sample_names <- names(true_forecast)
  }
  if (is.null(sample_names) || any(sample_names == "")) {
    sample_names <- paste0("Sample_", seq_len(n_samples))
  }
  names(log_traces) <- sample_names

  log_matrix <- do.call(cbind, log_traces)
  colnames(log_matrix) <- sample_names

  column_sums <- colSums(log_matrix)
  se <- if (length(column_sums) > 1 && !all(is.na(column_sums))) {
    stats::sd(column_sums, na.rm = TRUE) / sqrt(sum(!is.na(column_sums)))
  } else {
    0
  }

  list(
    traces = log_matrix,
    sum = column_sums,
    se = se
  )
}

plot_log_loss_traces <- function(log_loss_results,
                                 title = "Log loss of log returns",
                                 savefig = NULL) {
  if (is.null(log_loss_results) || !is.list(log_loss_results)) {
    stop("`log_loss_results` must be a list returned by `compute_log_loss_traces`.")
  }
  if (!all(c("traces", "sum", "se") %in% names(log_loss_results))) {
    stop("`log_loss_results` must contain `traces`, `sum`, and `se` elements.")
  }

  log_matrix <- log_loss_results$traces
  if (!is.matrix(log_matrix)) {
    log_matrix <- as.matrix(log_matrix)
  }
  log_df <- as.data.frame(log_matrix, stringsAsFactors = FALSE)
  log_df$Step <- seq_len(nrow(log_df))

  log_long <- tidyr::pivot_longer(
    log_df,
    cols = -Step,
    names_to = "Sample",
    values_to = "LogLoss"
  )

  average_df <- stats::aggregate(
    LogLoss ~ Step,
    data = log_long,
    FUN = function(x) mean(x, na.rm = TRUE)
  )
  names(average_df)[names(average_df) == "LogLoss"] <- "Average"

  mean_val <- mean(log_loss_results$sum, na.rm = TRUE)
  se_val <- log_loss_results$se
  mean_fmt <- if (is.finite(mean_val)) round(mean_val, 0) else "NA"
  se_fmt <- if (is.finite(se_val)) round(se_val, 1) else "NA"

  title_text <- sprintf(
    "%s (sum = %s +/- %s)",
    title,
    mean_fmt,
    se_fmt
  )

  options(repr.plot.width = 11, repr.plot.height = 4)
  p <- ggplot2::ggplot(
    log_long,
    ggplot2::aes(x = Step, y = LogLoss, group = Sample)
  ) +
    ggplot2::geom_line(colour = "grey70") +
    ggplot2::geom_line(
      data = average_df,
      mapping = ggplot2::aes(x = Step, y = Average),
      inherit.aes = FALSE,
      colour = "black",
      linewidth = 0.9
    ) +
    ggplot2::labs(
      title = title_text,
      x = "Step",
      y = "Log Loss"
    ) +
    ggplot2::theme_minimal()

  device_opened <- open_save_device(
    savefig = savefig,
    width = 1100,
    height = 400
  )

  if (device_opened) {
    print(p)
    grDevices::dev.off()
  }

  print(p)
  invisible(p)
}

plot_log_loss <- function(forecast_range,
                          forecast_steps,
                          forecast_data,
                          true_data,
                          observation,
                          savefig = NULL,
                          title = NULL) {
  log_loss <- sapply(1:forecast_range, function(i) {
    -log(approx(x = forecast_steps, y = forecast_data[, i], xout = true_data[[observation]][i])$y)
  })
  
  options(repr.plot.width = 11, repr.plot.height = 4)  # Adjust figure size
  p <- ggplot(
    data.frame(Timestep = 1:forecast_range, LogLoss = log_loss),
    aes(x = Timestep, y = LogLoss)
  ) +
    geom_line(color = "blue") +
    labs(
      title = if (is.null(title)) {
        paste("Log Loss over Forecast Timesteps, total =", round(sum(log_loss), 2))
      } else {
        paste(title, ", total =", round(sum(log_loss), 2))
      },
      x = "Timestep",
      y = "Log Loss"
    ) +
    theme_minimal()

  device_opened <- open_save_device(
    savefig = savefig,
    width = 1100,
    height = 400
  )

  if (device_opened) {
    print(p)
    grDevices::dev.off()
  }

  print(p)
  invisible(p)
}

plot_log_return_traces <- function(tsx_only,
                                   covariates_no_horseshoe,
                                   covariates_with_horseshoe,
                                   title = "Log Return Traces",
                                   savefig = NULL) {

  # Combine data into a single data frame for plotting
  plot_data <- data.frame(
    Timestep = seq_len(nrow(tsx_only$traces)),
    TSX_Only = rowMeans(tsx_only$traces, na.rm = TRUE),
    Covariates_No_Horseshoe = rowMeans(covariates_no_horseshoe$traces, na.rm = TRUE),
    Covariates_With_Horseshoe = rowMeans(covariates_with_horseshoe$traces, na.rm = TRUE)
  )
  
  # Melt the data for ggplot
  plot_data_long <- tidyr::pivot_longer(
    plot_data,
    cols = -Timestep,
    names_to = "Model",
    values_to = "Average_Trace"
  )
  
  # Add individual traces for each model
  tsx_only_traces <- as.data.frame(tsx_only$traces)
  tsx_only_traces$Timestep <- seq_len(nrow(tsx_only_traces))
  tsx_only_traces <- tidyr::pivot_longer(tsx_only_traces, -Timestep, names_to = "Trace", values_to = "Value")
  tsx_only_traces$Model <- "TSX_Only"

  cov_no_horseshoe_traces <- as.data.frame(covariates_no_horseshoe$traces)
  cov_no_horseshoe_traces$Timestep <- seq_len(nrow(cov_no_horseshoe_traces))
  cov_no_horseshoe_traces <- tidyr::pivot_longer(cov_no_horseshoe_traces, -Timestep, names_to = "Trace", values_to = "Value")
  cov_no_horseshoe_traces$Model <- "Covariates_No_Horseshoe"

  cov_with_horseshoe_traces <- as.data.frame(covariates_with_horseshoe$traces)
  cov_with_horseshoe_traces$Timestep <- seq_len(nrow(cov_with_horseshoe_traces))
  cov_with_horseshoe_traces <- tidyr::pivot_longer(cov_with_horseshoe_traces, -Timestep, names_to = "Trace", values_to = "Value")
  cov_with_horseshoe_traces$Model <- "Covariates_With_Horseshoe"
  
  # Combine all traces
  all_traces <- rbind(tsx_only_traces, cov_no_horseshoe_traces, cov_with_horseshoe_traces)

  options(repr.plot.width = 11, repr.plot.height = 4)
  p <- ggplot2::ggplot() +
    # Individual traces
    ggplot2::geom_line(data = all_traces, ggplot2::aes(x = Timestep, y = Value, colour = Model), alpha = 0.2) +
    # Averages
    ggplot2::geom_line(data = plot_data_long, ggplot2::aes(x = Timestep, y = Average_Trace, colour = Model), linewidth = 1.2) +
    ggplot2::scale_colour_manual(
      values = c(
        "TSX_Only" = "lightblue",
        "Covariates_No_Horseshoe" = "lightcoral",
        "Covariates_With_Horseshoe" = "lightgreen"
      )
    ) +
    ggplot2::labs(
      title = title,
      x = "Timestep",
      y = "Trace Value",
      color = "Model"
    ) +
    ggplot2::theme_minimal()

  device_opened <- open_save_device(
    savefig = savefig,
    width = 1100,
    height = 400
  )

  if (device_opened) {
    print(p)
    grDevices::dev.off()
  }

  print(p)
  invisible(p)
}
