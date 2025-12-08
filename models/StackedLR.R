library(caret)
library(dplyr)
library(ggplot2)

# --- stacker was trained on stack_train (meta) and predicted on stack_test (final)
# stack_test_prob must therefore be aligned with y_final (the 20% final holdout)
# If you currently have stack_pred_prob (vector) or stack_test_prob, make sure it corresponds to final.

stack_model <- glm(is_canceled ~ ., data = stack_train, family = "binomial")
stack_test_prob <- predict(stack_model, stack_test, type = "response")

# If you trained stack_model on stack_train and predicted on stack_test (final), do:
stack_test_prob <- predict(stack_model, stack_test, type = "response")  # ensure stack_final used
# if you already computed stack_test_prob earlier, verify length(stack_test_prob) == length(y_final)

# 3. Cost matrix + threshold sweep for all models -- use y_final and *_final_prob
C_FP <- 1200
C_FN <- 500

evaluate_at_threshold <- function(y_true, p_hat, threshold, C_FP, C_FN) {
  preds <- ifelse(p_hat >= threshold, 1, 0)
  cm <- confusionMatrix(
    factor(preds, levels = c(0, 1)),
    factor(y_true, levels = c(0, 1)),
    positive = "1"
  )
  FP <- sum(preds == 1 & y_true == 0)
  FN <- sum(preds == 0 & y_true == 1)
  cost     <- C_FP * FP + C_FN * FN
  avg_cost <- cost / length(y_true)
  data.frame(
    Threshold   = threshold,
    Accuracy    = as.numeric(cm$overall["Accuracy"]),
    Kappa       = as.numeric(cm$overall["Kappa"]),
    Sensitivity = as.numeric(cm$byClass["Sensitivity"]),
    Precision   = as.numeric(cm$byClass["Pos Pred Value"]),
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

# Use final holdout probabilities (the 20% final split)
results_all <- rbind(
  sweep_model(y_final, log_final_prob,   "Logistic",             C_FP, C_FN),       
  sweep_model(y_final, knn_final_prob,   "KNN",                  C_FP, C_FN),     
  sweep_model(y_final, c50_final_prob,   "C5.0",                 C_FP, C_FN),
  sweep_model(y_final, rf_final_prob,    "Random Forest",        C_FP, C_FN),
  sweep_model(y_final, svm_final_prob,   "SVM (linear)",         C_FP, C_FN),
  sweep_model(y_final, stack_test_prob,  "Stacked (meta-lr)",    C_FP, C_FN)
)

# --- Original "no-FP logistic" row: use a saved threshold or recompute from logistic meta set.
# Best practice: you saved the precision-max threshold earlier (recommended). If not, recompute it here
# using the logistic predictions on the meta set or final set depending on your intent.
#
# Example: recompute precision-max threshold from logistic performance on the final set:
thresholds <- seq(0.01, 0.99, by = 0.01)
precisions_log_final <- sapply(thresholds, function(t) {
  preds <- ifelse(log_final_prob >= t, 1, 0)
  cm <- confusionMatrix(factor(preds, levels=c(0,1)), factor(y_final, levels=c(0,1)), positive="1")
  as.numeric(cm$byClass["Pos Pred Value"])
})
best_idx <- which.max(ifelse(is.na(precisions_log_final), -Inf, precisions_log_final))
orig_thresh_no_fp <- thresholds[best_idx]

orig_preds <- ifelse(log_final_prob >= orig_thresh_no_fp, 1, 0)
orig_cm <- confusionMatrix(factor(orig_preds, levels = c(0,1)), factor(y_final, levels = c(0,1)), positive = "1")
orig_FP <- sum(orig_preds == 1 & y_final == 0)
orig_FN <- sum(orig_preds == 0 & y_final == 1)
orig_cost <- C_FP * orig_FP + C_FN * orig_FN
orig_avg_cost <- orig_cost / length(y_final)

original_lr_row <- data.frame(
  Threshold   = orig_thresh_no_fp,
  Accuracy    = as.numeric(orig_cm$overall["Accuracy"]),
  Kappa       = as.numeric(orig_cm$overall["Kappa"]),
  Sensitivity = as.numeric(orig_cm$byClass["Sensitivity"]),
  Precision   = as.numeric(orig_cm$byClass["Pos Pred Value"]),
  Cost        = orig_cost,
  AvgCost     = orig_avg_cost,
  Model       = "Logistic (No False Positives)"
)

# Baseline (no model) also evaluated on final holdout
baseline_preds <- rep(0, length(y_final))
baseline_cm <- confusionMatrix(factor(baseline_preds, levels = c(0,1)), factor(y_final, levels = c(0,1)), positive="1")
baseline_FP <- 0
baseline_FN <- sum(y_final == 1)
baseline_cost <- C_FP * baseline_FP + C_FN * baseline_FN
baseline_avg_cost <- baseline_cost / length(y_final)

baseline_row <- data.frame(
  Threshold   = NA_real_,
  Accuracy    = as.numeric(baseline_cm$overall["Accuracy"]),
  Kappa       = as.numeric(baseline_cm$overall["Kappa"]),
  Sensitivity = as.numeric(baseline_cm$byClass["Sensitivity"]),
  Precision   = as.numeric(baseline_cm$byClass["Pos Pred Value"]),
  Cost        = baseline_cost,
  AvgCost     = baseline_avg_cost,
  Model       = "Baseline (No Model)"
)

results_all_full <- bind_rows(results_all, original_lr_row, baseline_row)

best_by_cost <- results_all_full %>%
  group_by(Model) %>%
  slice_min(AvgCost, with_ties = FALSE) %>%
  ungroup()

print(best_by_cost)

# plotting and selecting best model: remember best_by_cost is computed over final holdout
# Find best model row by AvgCost
best_row <- best_by_cost %>% slice_min(AvgCost, with_ties = FALSE)
best_model_name <- best_row$Model
best_threshold  <- best_row$Threshold

cat("Best model:", best_model_name, "\n")
cat("Threshold used:", best_threshold, "\n\n")

# Choose probabilities consistent with final holdout
pred_probs <- switch(
  best_model_name,
  "Logistic"             = log_final_prob,
  "KNN"                  = knn_final_prob,
  "C5.0"                 = c50_final_prob,
  "Random Forest"        = rf_final_prob,
  "SVM (linear)"         = svm_final_prob,
  "Stacked (meta-lr)"    = stack_test_prob,
  stop("Unknown best model")
)

# Convert probabilities to class predictions using the chosen threshold
if (is.na(best_threshold)) {
  pred_classes <- ifelse(pred_probs >= 0.5, 1, 0)
} else {
  pred_classes <- ifelse(pred_probs >= best_threshold, 1, 0)
}

best_cm <- confusionMatrix(factor(pred_classes, levels=c(0,1)), factor(y_final, levels=c(0,1)), positive="1")
print(best_cm)
