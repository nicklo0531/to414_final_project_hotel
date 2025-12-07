library(caret)
library(dplyr)
library(ggplot2)


stack_train <- data.frame(
  log         = log_train_prob,
  ann         = ann_train_prob,
  knn         = knn_train_prob,
  dt          = dt_train_prob,
  rf          = rf_train_prob,
  svm         = svm_train_prob,
  is_canceled = y_train
)

stack_test <- data.frame(
  log = log_test_prob,
  ann = ann_test_prob,
  knn = knn_test_prob,
  dt  = dt_test_prob,
  rf  = rf_test_prob,
  svm = svm_test_prob
)

stack_model <- glm(is_canceled ~ ., data = stack_train, family = "binomial")
stack_test_prob <- predict(stack_model, stack_test, type = "response")


# 3. Cost matrix + threshold sweep for all models

# Cost matrix: FPs are much more expensive than FNs
C_FP <- 1200  # false positive: predict cancel, guest shows (walk/overbook)
C_FN <- 500   # false negative: predict show, guest cancels (empty room)

evaluate_at_threshold <- function(y_true, p_hat, threshold, C_FP, C_FN) {
  preds <- ifelse(p_hat >= threshold, 1, 0)
  
  TP <- sum(preds == 1 & y_true == 1)
  FP <- sum(preds == 1 & y_true == 0)
  FN <- sum(preds == 0 & y_true == 1)
  TN <- sum(preds == 0 & y_true == 0)
  
  cm <- confusionMatrix(
    factor(preds, levels = c(0, 1)),
    factor(y_true, levels = c(0, 1)),
    positive = "1"
  )
  
  cost     <- C_FP * FP + C_FN * FN
  avg_cost <- cost / length(y_true)
  
  data.frame(
    Threshold   = threshold,
    Accuracy    = cm$overall["Accuracy"],
    Kappa       = cm$overall["Kappa"],
    Sensitivity = cm$byClass["Sensitivity"],
    Precision   = cm$byClass["Pos Pred Value"],
    Cost        = cost,
    AvgCost     = avg_cost
  )
}

sweep_model <- function(y_true, p_hat, model_name,
                        C_FP, C_FN,
                        thresholds = seq(0.01, 0.99, by = 0.01)) {
  rows <- lapply(thresholds, function(t) {
    evaluate_at_threshold(y_true, p_hat, t, C_FP, C_FN)
  })
  out <- do.call(rbind, rows)
  out$Model <- model_name
  out
}

results_all <- rbind(
  sweep_model(y_test, log_test_prob,   "Logistic",             C_FP, C_FN),
  sweep_model(y_test, ann_test_prob,   "ANN",                  C_FP, C_FN),
  sweep_model(y_test, knn_test_prob,   "KNN",                  C_FP, C_FN),
  sweep_model(y_test, dt_test_prob,    "Decision Tree",        C_FP, C_FN),
  sweep_model(y_test, rf_test_prob,    "Random Forest",        C_FP, C_FN),
  sweep_model(y_test, svm_test_prob,   "SVM (linear)",         C_FP, C_FN),
  sweep_model(y_test, stack_test_prob, "Stacked (meta-logit)", C_FP, C_FN)
)


# 4. Original no-FP logistic + simple baseline

## 4a. Original logistic regression (no false positives)
##     Source your LogisticRegression.R, which finds the precision-max threshold
# assumes it creates cancel_pred_m1, metric_summary, y_test

# Threshold that maximizes precision in the original LR (this should give 0 FPs)
orig_thresh_no_fp <- metric_plot_df$Threshold[metric_plot_df$Metric == "Precision"]

orig_preds <- ifelse(log_test_prob >= orig_thresh_no_fp, 1, 0)

orig_cm <- confusionMatrix(
  factor(orig_preds, levels = c(0, 1)),
  factor(y_test,     levels = c(0, 1)),
  positive = "1"
)

orig_FP <- sum(orig_preds == 1 & y_test == 0)
orig_FN <- sum(orig_preds == 0 & y_test == 1)
orig_cost <- C_FP * orig_FP + C_FN * orig_FN
orig_avg_cost <- orig_cost / length(y_test)

original_lr_row <- data.frame(
  Threshold   = orig_thresh_no_fp,
  Accuracy    = orig_cm$overall["Accuracy"],
  Kappa       = orig_cm$overall["Kappa"],
  Sensitivity = orig_cm$byClass["Sensitivity"],
  Precision   = orig_cm$byClass["Pos Pred Value"],
  Cost        = orig_cost,
  AvgCost     = orig_avg_cost,
  Model       = "Logistic (No False Positives)"
)

## 4b. Super simple baseline that never predicts a cancellation
##     This basically says “everyone will show up”

baseline_preds <- rep(0, length(y_test))

baseline_cm <- confusionMatrix(
  factor(baseline_preds, levels = c(0, 1)),
  factor(y_test,          levels = c(0, 1)),
  positive = "1"
)

baseline_FP <- 0
baseline_FN <- sum(y_test == 1)
baseline_cost <- C_FP * baseline_FP + C_FN * baseline_FN
baseline_avg_cost <- baseline_cost / length(y_test)
baseline_cancel_rate <- mean(y_test)  # percent of cancels in the original test split

baseline_row <- data.frame(
  Threshold   = NA_real_,
  Accuracy    = baseline_cm$overall["Accuracy"],
  Kappa       = baseline_cm$overall["Kappa"],
  Sensitivity = baseline_cm$byClass["Sensitivity"],
  Precision   = baseline_cm$byClass["Pos Pred Value"],
  Cost        = baseline_cost,
  AvgCost     = baseline_avg_cost,
  Model       = "Baseline (No Model)"
)

## Combine everything and then get “best by cost” including the two new baselines

results_all_full <- bind_rows(results_all, original_lr_row, baseline_row)

best_by_cost <- results_all_full %>%
  group_by(Model) %>%
  slice_min(AvgCost, with_ties = FALSE) %>%
  ungroup()

print(best_by_cost)

# Objects to look at after sourcing:
#   results_all_full  -> all thresholds for the 6 base models + stacked
#                        plus single rows for original LR no-FP and baseline
#   best_by_cost      -> single “best” row per model (lowest AvgCost)

library(ggplot2)

plot_best_cost <- function() {
  best_plot_df <- best_by_cost %>%
    dplyr::mutate(
      ThresholdLabel = ifelse(
        is.na(Threshold),
        "t = N/A",
        paste0("t = ", round(Threshold, 2))
      )
    )
  
  print(
    ggplot(best_plot_df,
           aes(x = reorder(Model, AvgCost), y = AvgCost)) +
      geom_col() +
      geom_text(aes(label = ThresholdLabel),
                hjust = -0.1, size = 3) +
      coord_flip() +
      labs(
        title = "Best average cost per booking by model",
        x     = "Model",
        y     = "Average cost per reservation ($)"
      ) +
      ylim(0, max(best_plot_df$AvgCost) * 1.15) +
      theme_minimal()
  )
}

plot_best_cost()