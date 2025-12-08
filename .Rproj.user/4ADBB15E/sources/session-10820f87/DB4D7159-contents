library(caret)
library(dplyr)
library(randomForest)
library(rpart)
library(rpart.plot)
library(pROC)


# 1. Train all meta-models

# metaDT with more splits
set.seed(12345)
metaDT <- rpart(
  is_canceled ~ ., data = stack_train, method = "class",
  control = rpart.control(cp = 0.0001)
)
metaDT_prob <- predict(metaDT, stack_test, type = "prob")[, "1"]

# metaLR
metaLR <- glm(is_canceled ~ ., data = stack_train, family = "binomial")
metaLR_prob <- predict(metaLR, stack_test, type = "response")

# metaRF
set.seed(12345)
metaRF <- randomForest(
  is_canceled ~ ., data = stack_train,
  ntree = 500, mtry = 3, nodesize = 5
)
metaRF_prob <- predict(metaRF, stack_test, type = "prob")[, "1"]

# base RF probability on same final set
baseRF_prob <- rf_test_prob[-meta_idx]


# 2. Threshold sweep

thresholds <- seq(0.01, 0.99, by = 0.01)

evaluate <- function(y, p, t) {
  preds <- ifelse(p >= t, 1, 0)
  cm <- confusionMatrix(
    factor(preds, levels=c(0,1)),
    factor(y,    levels=c(0,1)),
    positive="1"
  )
  FP <- sum(preds == 1 & y == 0)
  FN <- sum(preds == 0 & y == 1)
  cost <- C_FP*FP + C_FN*FN
  
  tibble(
    Threshold = t,
    Accuracy = cm$overall["Accuracy"],
    Kappa    = cm$overall["Kappa"],
    Sensitivity = cm$byClass["Sensitivity"],
    Precision   = cm$byClass["Pos Pred Value"],
    AvgCost  = cost / length(y)
  )
}

sweep_model <- function(name, y_true, prob) {
  results <- do.call(
    rbind,
    lapply(thresholds, function(t) evaluate(y_true, prob, t))
  )
  results$Model <- name
  results
}


# 3. Sweep all models together

results_all <- bind_rows(
  sweep_model("metaDT", y_final, metaDT_prob),
  sweep_model("metaLR", y_final, metaLR_prob),
  sweep_model("metaRF", y_final, metaRF_prob),
  sweep_model("baseRF", y_final, baseRF_prob)
)

best_table <- results_all %>%
  group_by(Model) %>%
  slice_min(AvgCost, with_ties = FALSE) %>%
  ungroup()

print(best_table)


# 4. Final confusion matrices

get_cm <- function(prob, thr, y_true) {
  preds <- ifelse(prob >= thr, 1, 0)
  confusionMatrix(
    factor(preds, levels=c(0,1)),
    factor(y_true, levels=c(0,1)),
    positive="1"
  )
}

final_cm_metaDT <- get_cm(metaDT_prob, best_table$Threshold[best_table$Model=="metaDT"], y_final)
final_cm_metaLR <- get_cm(metaLR_prob, best_table$Threshold[best_table$Model=="metaLR"], y_final)
final_cm_metaRF <- get_cm(metaRF_prob, best_table$Threshold[best_table$Model=="metaRF"], y_final)
final_cm_baseRF <- get_cm(baseRF_prob, best_table$Threshold[best_table$Model=="baseRF"], y_final)


# 5. Print results

rpart.plot(metaDT)

cat("\n=== Final Confusion Matrix: metaDT ===\n")
print(final_cm_metaDT)

cat("\n=== Final Confusion Matrix: metaLR ===\n")
print(final_cm_metaLR)

cat("\n=== Final Confusion Matrix: metaRF ===\n")
print(final_cm_metaRF)

cat("\n=== Final Confusion Matrix: baseRF ===\n")
print(final_cm_baseRF)
