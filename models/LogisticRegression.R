# Logistic Regression
library(caret)
library(ggplot2)

# Make sure the outcome is 0/1 numeric
# this just makes sure everything is numbers instead of like "yes" and "no"
if (is.factor(df_train$is_canceled) || is.character(df_train$is_canceled)) {
  df_train$is_canceled <- as.numeric(as.character(df_train$is_canceled))
  df_test$is_canceled  <- as.numeric(as.character(df_test$is_canceled))
}

# Drop predictors with near-zero variance
# gets rid of columns that are basically all the same value bc they don't help
pred_names <- setdiff(names(df_train), "is_canceled")
nzv_idx <- nearZeroVar(df_train[, pred_names, drop = FALSE])
if (length(nzv_idx) > 0) {
  drop_names <- pred_names[nzv_idx]
  df_train <- df_train[, !(names(df_train) %in% drop_names), drop = FALSE]
  df_test  <- df_test[,  !(names(df_test)  %in% drop_names),  drop = FALSE]
}

# Backward stepwise limited function
backward_step_limited <- function(formula, data, family = binomial(),
                                  max_steps = 8, min_improve = 1e-3,
                                  maxit_glm = 100, trace = TRUE) {
  
  current_model <- glm(formula, data = data, family = family,
                       control = glm.control(maxit = maxit_glm))
  current_aic <- AIC(current_model)
  if (trace) cat("Initial AIC:", current_aic, "\n")
  
  resp <- all.vars(formula)[1]
  terms_curr <- attr(terms(current_model), "term.labels")
  
  # tries removing variables to see if model improves (lower AIC is better)
  for (step_i in seq_len(max_steps)) {
    if (length(terms_curr) == 0) break
    
    best_aic <- current_aic
    best_drop <- NULL
    
    for (term in terms_curr) {
      new_terms <- setdiff(terms_curr, term)
      new_form <- if (length(new_terms) == 0) as.formula(paste(resp, "~ 1")) else as.formula(paste(resp, "~", paste(new_terms, collapse = " + ")))
      
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
    
    if (trace) cat("Step", step_i, ": drop", best_drop, "AIC:", current_aic, "->", best_aic, "\n")
    current_model <- best_model
    current_aic <- best_aic
    terms_curr <- setdiff(terms_curr, best_drop)
  }
  
  if (trace) cat("Final AIC:", current_aic, "\n")
  return(current_model)
}

# Fit logistic regression with limited backward stepwise
m1 <- backward_step_limited(is_canceled ~ ., data = df_train, family = binomial())

# Compute train & test probabilities for stacking
log_train_prob <- predict(m1, df_train, type = "response")  # train-side
log_test_prob  <- predict(m1, df_test, type = "response")   # test-side

# Save probabilities
saveRDS(log_train_prob, "RDS/log_train_prob.rds")
saveRDS(log_test_prob,  "RDS/log_test_prob.rds")

# Threshold sweep for evaluation
# checking different thresholds to see which one works best
# like do we predict canceled when prob is over 0.5? or 0.3? testing them all
thresholds <- seq(0.01, 0.99, by = 0.01)
precisions <- sensitivities <- accuracies <- kappas <- numeric(length(thresholds))
y_test <- df_test$is_canceled

for (i in seq_along(thresholds)) {
  t <- thresholds[i]
  preds <- ifelse(log_test_prob >= t, 1, 0)
  
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
best_threshold_prec <- thresholds[best_idx_safe(precisions)]
cat("Best threshold by precision:", best_threshold_prec, "\n")

# Final 0/1 prediction using best precision threshold
cancel_bin_pred_m1 <- ifelse(log_test_prob >= best_threshold_prec, 1, 0)
cm_log <- confusionMatrix(factor(cancel_bin_pred_m1, levels = c(0,1)),
                          factor(y_test, levels = c(0,1)),
                          positive = "1")
print(cm_log)