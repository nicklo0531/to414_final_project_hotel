## LogisticRegression.R
## Logistic regression with a small stepwise feature selection and threshold search

library(caret)

# Read in the cleaned train and test files we made earlier
df_train <- read.csv("data/hotel_train.csv")
df_test  <- read.csv("data/hotel_test.csv")

# Make sure the outcome is 0/1 numbers so glm does not freak out
if (is.factor(df_train$is_canceled) || is.character(df_train$is_canceled)) {
  df_train$is_canceled <- as.numeric(as.character(df_train$is_canceled))
  df_test$is_canceled  <- as.numeric(as.character(df_test$is_canceled))
}

# Drop any predictors that basically never change (near-zero variance)
pred_names <- setdiff(names(df_train), "is_canceled")

nzv_idx <- nearZeroVar(df_train[, pred_names, drop = FALSE])

if (length(nzv_idx) > 0) {
  drop_names <- pred_names[nzv_idx]
  df_train <- df_train[, !(names(df_train) %in% drop_names), drop = FALSE]
  df_test  <- df_test[,  !(names(df_test)  %in% drop_names),  drop = FALSE]
}

# Custom backward stepwise function so we do not spend forever dropping tiny effects
backward_step_limited <- function(formula,
                                  data,
                                  family = binomial(),
                                  max_steps = 8,
                                  min_improve = 1e-3,
                                  maxit_glm = 100,
                                  trace = TRUE) {
  
  # Start with the full model
  current_model <- glm(formula, data = data, family = family,
                       control = glm.control(maxit = maxit_glm))
  current_aic <- AIC(current_model)
  if (trace) cat("Initial AIC:", current_aic, "\n")
  
  resp <- all.vars(formula)[1]
  terms_curr <- attr(terms(current_model), "term.labels")
  
  # At each step, try dropping one variable and see if AIC really improves
  for (step_i in seq_len(max_steps)) {
    if (length(terms_curr) == 0) break
    
    best_aic  <- current_aic
    best_drop <- NULL
    
    for (term in terms_curr) {
      new_terms <- setdiff(terms_curr, term)
      if (length(new_terms) == 0) {
        new_form <- as.formula(paste(resp, "~ 1"))
      } else {
        new_form <- as.formula(
          paste(resp, "~", paste(new_terms, collapse = " + "))
        )
      }
      
      cand_model <- try(
        glm(new_form, data = data, family = family,
            control = glm.control(maxit = maxit_glm)),
        silent = TRUE
      )
      if (inherits(cand_model, "try-error")) next
      
      cand_aic <- AIC(cand_model)
      # Only bother if AIC gets at least a tiny bit better
      if (cand_aic + min_improve < best_aic) {
        best_aic  <- cand_aic
        best_drop <- term
        best_model <- cand_model
      }
    }
    
    # If nothing helps anymore, stop
    if (is.null(best_drop)) break
    
    if (trace) {
      cat("Step", step_i, ": drop", best_drop,
          "AIC:", current_aic, "->", best_aic, "\n")
    }
    current_model <- best_model
    current_aic   <- best_aic
    terms_curr    <- setdiff(terms_curr, best_drop)
  }
  
  if (trace) cat("Final AIC:", current_aic, "\n")
  return(current_model)
}

# Fit the logistic model with the limited backward stepwise
m1 <- backward_step_limited(is_canceled ~ ., data = df_train, family = binomial())

# Get predicted cancellation probabilities on the test set
cancel_pred_m1 <- predict(m1, df_test, type = "response")
y_test <- df_test$is_canceled

# Try a bunch of cutoffs and see how each one does
thresholds    <- seq(0.01, 0.99, by = 0.01)
precisions    <- numeric(length(thresholds))
sensitivities <- numeric(length(thresholds))
accuracies    <- numeric(length(thresholds))
kappas        <- numeric(length(thresholds))

for (i in seq_along(thresholds)) {
  t <- thresholds[i]
  # If prob >= t, we say “this guest will cancel”
  preds <- ifelse(cancel_pred_m1 >= t, 1, 0)
  
  cm_tmp <- confusionMatrix(
    factor(preds,  levels = c(0, 1)),
    factor(y_test, levels = c(0, 1)),
    positive = "1"
  )
  
  accuracies[i]    <- cm_tmp$overall["Accuracy"]
  kappas[i]        <- cm_tmp$overall["Kappa"]
  sensitivities[i] <- cm_tmp$byClass["Sensitivity"]
  precisions[i]    <- cm_tmp$byClass["Pos Pred Value"]
}

# If precision is NaN (like when we never predict any 1s), turn it into NA so it doesn't win by accident
precisions[is.nan(precisions)] <- NA

# Helper that finds the best index while ignoring NAs
best_idx_safe <- function(x) which.max(ifelse(is.na(x), -Inf, x))

best_threshold_acc  <- thresholds[best_idx_safe(accuracies)]
best_threshold_sens <- thresholds[best_idx_safe(sensitivities)]
best_threshold_prec <- thresholds[best_idx_safe(precisions)]
best_threshold_kap  <- thresholds[best_idx_safe(kappas)]

# Little summary table that says which cutoff is best for each metric
metric_summary <- data.frame(
  Metric    = c("Accuracy", "Sensitivity", "Precision", "Kappa"),
  Threshold = c(best_threshold_acc,
                best_threshold_sens,
                best_threshold_prec,
                best_threshold_kap),
  Value     = c(accuracies[best_idx_safe(accuracies)],
                sensitivities[best_idx_safe(sensitivities)],
                precisions[best_idx_safe(precisions)],
                kappas[best_idx_safe(kappas)])
)

# Final 0/1 prediction using the cutoff that gives us the best precision
cancel_bin_pred_m1 <- ifelse(cancel_pred_m1 >= best_threshold_prec, 1, 0)

cm_log <- confusionMatrix(
  factor(cancel_bin_pred_m1, levels = c(0, 1)),
  factor(y_test,             levels = c(0, 1)),
  positive = "1"
)

# Keep the raw probabilities around in case we want stacking or cost stuff later
log_pred <- cancel_pred_m1
