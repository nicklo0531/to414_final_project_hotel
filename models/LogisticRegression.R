
library(caret)

# Load data

hotel_train <- read.csv("data/hotel_train.csv")
hotel_test  <- read.csv("data/hotel_test.csv")

# Make sure outcome is numeric 0/1 for glm
if (is.factor(hotel_train$is_canceled) || is.character(hotel_train$is_canceled)) {
  hotel_train$is_canceled <- as.numeric(as.character(hotel_train$is_canceled))
  hotel_test$is_canceled  <- as.numeric(as.character(hotel_test$is_canceled))
}

# Drop near-zero-variance predictors 
X_train <- hotel_train[ , !(names(hotel_train) %in% "is_canceled"), drop = FALSE]
y_train <- hotel_train$is_canceled

nzv <- nearZeroVar(X_train)

if (length(nzv) > 0) {
  X_train <- X_train[ , -nzv, drop = FALSE]
  X_test  <- hotel_test[ , !(names(hotel_test) %in% "is_canceled"), drop = FALSE]
  X_test  <- X_test[ , -nzv, drop = FALSE]
} else {
  X_test  <- hotel_test[ , !(names(hotel_test) %in% "is_canceled"), drop = FALSE]
}

hotel_train_red <- cbind(is_canceled = y_train, X_train)
hotel_test_red  <- cbind(is_canceled = hotel_test$is_canceled, X_test)

# A traditional backwards stepwise takes too long.
# We're going to customize it so that once removing variables adds negligible effects to the ability of the model, it stops.
# Limited backward stepwise function 
backward_step_limited <- function(formula,
                                  data,
                                  family = binomial(),
                                  max_steps = 8,
                                  min_improve = 1e-3,
                                  maxit_glm = 100,
                                  trace = TRUE) {
  # Fit initial full model
  current_model <- glm(formula, data = data, family = family,
                       control = glm.control(maxit = maxit_glm))
  current_aic <- AIC(current_model)
  if (trace) cat("Initial AIC:", current_aic, "\n")
  
  # Get response and term labels
  resp <- all.vars(formula)[1]
  terms_curr <- attr(terms(current_model), "term.labels")
  
  for (step_i in seq_len(max_steps)) {
    if (length(terms_curr) == 0) {
      if (trace) cat("No more terms to drop; stopping.\n")
      break
    }
    
    best_aic <- current_aic
    best_drop <- NULL
    
    # Try dropping each term once
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
      if (cand_aic + min_improve < best_aic) {
        best_aic  <- cand_aic
        best_drop <- term
        best_model <- cand_model
      }
    }
    
    # Check if any drop was actually helpful
    if (is.null(best_drop)) {
      if (trace) cat("No further AIC improvement; stopping.\n")
      break
    } else {
      if (trace) {
        cat("Step", step_i, ": dropping", best_drop,
            "AIC:", current_aic, "->", best_aic, "\n")
      }
      # Update current model/terms/AIC
      current_model <- best_model
      current_aic   <- best_aic
      terms_curr    <- setdiff(terms_curr, best_drop)
    }
  }
  
  if (trace) cat("Final AIC:", current_aic, "\n")
  return(current_model)
}

# Run limited backward stepwise
m1 <- backward_step_limited(
  is_canceled ~ .,
  data        = hotel_train_red,
  family      = binomial(),
  max_steps   = 8,      
  min_improve = 1e-3,
  maxit_glm   = 100,
  trace       = TRUE
)

summary(m1)

# Predict on the test set 
cancel_pred_m1 <- predict(m1, hotel_test_red, type = "response")
y_test <- hotel_test_red$is_canceled

# Go through a sequence of thresholds and track metrics at each one
thresholds   <- seq(0.01, 0.99, by = 0.01)
precisions   <- numeric(length(thresholds))
sensitivities <- numeric(length(thresholds))
accuracies   <- numeric(length(thresholds))
kappas       <- numeric(length(thresholds))

for (i in seq_along(thresholds)) {
  t <- thresholds[i]
  # Predict that they "cancel" if prob >= t
  preds <- ifelse(cancel_pred_m1 >= t, 1, 0)
  
  cm_tmp <- confusionMatrix(
    factor(preds, levels = c(0, 1)),
    factor(y_test, levels = c(0, 1)),
    positive = "1"
  )
  
  accuracies[i]    <- cm_tmp$overall["Accuracy"]
  kappas[i]        <- cm_tmp$overall["Kappa"]
  sensitivities[i] <- cm_tmp$byClass["Sensitivity"]
  precisions[i]    <- cm_tmp$byClass["Pos Pred Value"]
}

# If precision is NaN (like 0/0 when we never predict any 1s), turn it into NA so we ignore it later
precisions[is.nan(precisions)] <- NA

# Helper so which.max ignores NAs
best_idx_safe <- function(x) which.max(ifelse(is.na(x), -Inf, x))

best_idx_acc  <- best_idx_safe(accuracies)
best_idx_sens <- best_idx_safe(sensitivities)
best_idx_prec <- best_idx_safe(precisions)
best_idx_kap  <- best_idx_safe(kappas)

best_threshold_acc  <- thresholds[best_idx_acc]
best_threshold_sens <- thresholds[best_idx_sens]
best_threshold_prec <- thresholds[best_idx_prec]
best_threshold_kap  <- thresholds[best_idx_kap]

# Small summary table: for each metric, show the threshold where it's highest
metric_summary <- data.frame(
  Metric    = c("Accuracy", "Sensitivity", "Precision", "Kappa"),
  Threshold = c(best_threshold_acc,
                best_threshold_sens,
                best_threshold_prec,
                best_threshold_kap),
  Value     = c(accuracies[best_idx_acc],
                sensitivities[best_idx_sens],
                precisions[best_idx_prec],
                kappas[best_idx_kap])
)

metric_summary

# If you still want to classify using the threshold that maximizes precision:
cancel_bin_pred_m1 <- ifelse(cancel_pred_m1 >= best_threshold_prec, 1, 0)

cm_log <- confusionMatrix(
  factor(cancel_bin_pred_m1, levels = c(0, 1)),
  factor(y_test,             levels = c(0, 1)),
  positive = "1"
)

cm_log

# Pull out the main stats we care about: precision, sensitivity, and specificity
cm_log$byClass[c("Pos Pred Value", "Sensitivity", "Specificity")]

# Save the probabilities in case we use this model later (stacking, cost matrix, etc.)
log_pred <- cancel_pred_m1
