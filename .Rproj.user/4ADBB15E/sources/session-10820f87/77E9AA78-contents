library(e1071)
library(caret)
library(pROC)
set.seed(12345)

# Ensure factors with consistent levels
df_train$is_canceled <- factor(df_train$is_canceled, levels = c(0,1))
df_test$is_canceled  <- factor(df_test$is_canceled,  levels = c(0,1))

kernels <- c("linear", "radial", "sigmoid")
svm_results <- list()
svm_aucs <- c()

for (k in kernels) {
  
  svm_model <- svm(
    is_canceled ~ .,
    data = df_train,
    kernel = k,
    probability = TRUE
  )
  
  # Predictions on test
  pred_class <- predict(svm_model, df_test)
  pred_class <- factor(pred_class, levels = levels(df_test$is_canceled))
  
  prob_obj <- predict(svm_model, df_test, probability = TRUE)
  pred_prob <- attr(prob_obj, "probabilities")[, "1"]
  
  if (k == "linear") {
    saveRDS(svm_model, "RDS/svm_linear_model.rds")
    train_prob <- attr(predict(svm_model, df_train, probability = TRUE), "probabilities")[, "1"]
    saveRDS(train_prob, "RDS/svm_linear_train_prob.rds")
    saveRDS(pred_prob,  "RDS/svm_linear_test_prob.rds")
  }
  
  # Also save train probabilities for stacking
  train_prob <- attr(predict(svm_model, df_train, probability = TRUE), "probabilities")[, "1"]
  saveRDS(train_prob, paste0("RDS/svm_", k, "_train_prob.rds"))
  
  
  # Metrics
  cm <- confusionMatrix(pred_class, df_test$is_canceled)
  roc_obj <- roc(df_test$is_canceled, pred_prob)
  auc_val <- auc(roc_obj)
  
  svm_results[[k]] <- cm
  svm_aucs[k] <- auc_val
}
