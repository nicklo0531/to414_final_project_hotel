library(caret)
library(dplyr)
library(randomForest)

### 1. Train Stacked Logistic (meta-LR)
stack_model <- glm(is_canceled ~ ., data = stack_train, family = "binomial")
stack_lr_prob <- predict(stack_model, stack_test, type = "response")  # length == y_final


### 2. Train Random Forest meta-model (metaRF)
set.seed(12345)
meta_rf_model <- randomForest(
  is_canceled ~ ., 
  data = stack_train,
  ntree = 500,
  mtry = 3,
  nodeSize = 5
)

# Predict meta-RF probabilities (class "1")
stack_rf_prob <- predict(meta_rf_model, stack_test, type = "prob")[, "1"]

# Ensure lengths match for evaluation
stopifnot(length(stack_rf_prob) == length(y_final))


### 3. Evaluate MetaDT (already trained above)
stack_dt_prob <- stack_pred_prob  # from your previous rpart predict()
stopifnot(length(stack_dt_prob) == length(y_final))
stopifnot(length(stack_lr_prob) == length(y_final))


### 4. Cost matrix + threshold sweep
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
  cost <- C_FP * FP + C_FN * FN
  avg_cost <- cost / length(y_true)
  
  data.frame(
    Threshold = threshold,
    Accuracy  = as.numeric(cm$overall["Accuracy"]),
    Kappa     = as.numeric(cm$overall["Kappa"]),
    Sensitivity = as.numeric(cm$byClass["Sensitivity"]),
    Precision   = as.numeric(cm$byClass["Pos Pred Value"]),
    AvgCost   = avg_cost
  )
}

sweep_model <- function(name, y_true, prob) {
  thresholds <- seq(0.01, 0.99, by = 0.01)
  out <- do.call(rbind, lapply(thresholds, function(t)
    evaluate_at_threshold(y_true, prob, t, C_FP, C_FN)
  ))
  out$Model <- name
  out
}


### 5. Run sweeps for: metaDT, metaLR, metaRF, base RF
results <- rbind(
  sweep_model("metaDT",        y_final, stack_dt_prob),
  sweep_model("metaLR",        y_final, stack_lr_prob),
  sweep_model("metaRF",        y_final, stack_rf_prob),
  sweep_model("Random Forest", y_final, rf_test_prob[-meta_idx])
)


### 6. Best threshold for each
best_table <- results %>%
  group_by(Model) %>%
  slice_min(AvgCost, with_ties = FALSE) %>%
  ungroup()

print(best_table)


### 7. Helper: final confusion matrices
get_cm <- function(prob, threshold, y_true) {
  preds <- ifelse(prob >= threshold, 1, 0)
  confusionMatrix(
    factor(preds, levels=c(0,1)),
    factor(y_true, levels=c(0,1)),
    positive="1"
  )
}


### Final CMs

# MetaDT
best_dt_row <- best_table %>% filter(Model == "metaDT")
final_cm_metaDT <- get_cm(stack_dt_prob, best_dt_row$Threshold, y_final)

# Stacked-LR
best_lr_row <- best_table %>% filter(Model == "metaLR")
final_cm_metaLR <- get_cm(stack_lr_prob, best_lr_row$Threshold, y_final)

# Stacked-RF (NEW)
best_rfmeta_row <- best_table %>% filter(Model == "metaRF")
final_cm_metaRF <- get_cm(stack_rf_prob, best_rfmeta_row$Threshold, y_final)

# Base Random Forest
best_rf_row <- best_table %>% filter(Model == "Random Forest")
final_cm_rf <- get_cm(rf_test_prob[-meta_idx], best_rf_row$Threshold, y_final)


### Print final matrices
cat("\n\n=== Final Confusion Matrix: metaDT ===\n")
print(final_cm_metaDT)

cat("\n\n=== Final Confusion Matrix: metaLR ===\n")
print(final_cm_metaLR)

cat("\n\n=== Final Confusion Matrix: metaRF ===\n")
print(final_cm_metaRF)

cat("\n\n=== Final Confusion Matrix: Base Random Forest ===\n")
print(final_cm_rf)
