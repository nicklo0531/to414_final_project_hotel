library(caret)

# Remove near-zero variance predictors if any
nzv <- nearZeroVar(df_train[, -1])  # skip is_canceled
if(length(nzv) > 0){
  df_train <- df_train[, -nzv]
  df_test  <- df_test[,  -nzv]
}

# Limited backward stepwise function
backward_step_limited <- function(formula,
                                  data,
                                  family = binomial(),
                                  max_steps = 8,
                                  min_improve = 1e-3,
                                  maxit_glm = 100,
                                  trace = TRUE) {
  
  current_model <- glm(formula, data = data, family = family,
                       control = glm.control(maxit = maxit_glm))
  current_aic <- AIC(current_model)
  if (trace) cat("Initial AIC:", current_aic, "\n")
  
  resp <- all.vars(formula)[1]
  terms_curr <- attr(terms(current_model), "term.labels")
  
  for (step_i in seq_len(max_steps)) {
    if (length(terms_curr) == 0) break
    best_aic <- current_aic
    best_drop <- NULL
    for (term in terms_curr) {
      new_terms <- setdiff(terms_curr, term)
      if (length(new_terms) == 0) {
        new_form <- as.formula(paste(resp, "~ 1"))
      } else {
        new_form <- as.formula(paste(resp, "~", paste(new_terms, collapse = " + ")))
      }
      cand_model <- try(glm(new_form, data = data, family = family,
                            control = glm.control(maxit = maxit_glm)), silent = TRUE)
      if (inherits(cand_model, "try-error")) next
      cand_aic <- AIC(cand_model)
      if (cand_aic + min_improve < best_aic) {
        best_aic <- cand_aic
        best_drop <- term
        best_model <- cand_model
      }
    }
    if (is.null(best_drop)) break
    if (trace) cat("Step", step_i, ": drop", best_drop,
                   "AIC:", current_aic, "->", best_aic, "\n")
    current_model <- best_model
    current_aic   <- best_aic
    terms_curr    <- setdiff(terms_curr, best_drop)
  }
  if (trace) cat("Final AIC:", current_aic, "\n")
  return(current_model)
}

# Run backward stepwise logistic regression
m1 <- backward_step_limited(is_canceled ~ ., data = df_train, family = binomial())

# Predict probabilities on test set
cancel_pred_m1 <- predict(m1, df_test, type = "response")
y_test <- df_test$is_canceled

# Evaluate across thresholds
thresholds <- seq(0.01, 0.99, by = 0.01)
precisions <- sensitivities <- accuracies <- kappas <- numeric(length(thresholds))

for (i in seq_along(thresholds)) {
  t <- thresholds[i]
  preds <- ifelse(cancel_pred_m1 >= t, 1, 0)
  cm_tmp <- confusionMatrix(factor(preds, levels = c(0,1)),
                            factor(y_test, levels = c(0,1)),
                            positive = "1")
  accuracies[i]    <- cm_tmp$overall["Accuracy"]
  kappas[i]        <- cm_tmp$overall["Kappa"]
  sensitivities[i] <- cm_tmp$byClass["Sensitivity"]
  precisions[i]    <- cm_tmp$byClass["Pos Pred Value"]
}
precisions[is.nan(precisions)] <- NA

best_idx_safe <- function(x) which.max(ifelse(is.na(x), -Inf, x))
best_threshold_acc  <- thresholds[best_idx_safe(accuracies)]
best_threshold_sens <- thresholds[best_idx_safe(sensitivities)]
best_threshold_prec <- thresholds[best_idx_safe(precisions)]
best_threshold_kap  <- thresholds[best_idx_safe(kappas)]

metric_summary <- data.frame(
  Metric    = c("Accuracy", "Sensitivity", "Precision", "Kappa"),
  Threshold = c(best_threshold_acc, best_threshold_sens,
                best_threshold_prec, best_threshold_kap),
  Value     = c(accuracies[best_idx_safe(accuracies)],
                sensitivities[best_idx_safe(sensitivities)],
                precisions[best_idx_safe(precisions)],
                kappas[best_idx_safe(kappas)])
)

# Final binary prediction using best precision threshold
cancel_bin_pred_m1 <- ifelse(cancel_pred_m1 >= best_threshold_prec, 1, 0)
cm_log <- confusionMatrix(factor(cancel_bin_pred_m1, levels = c(0,1)),
                          factor(y_test, levels = c(0,1)),
                          positive = "1")

# Probabilities for potential stacking or further analysis
log_pred <- cancel_pred_m1
  