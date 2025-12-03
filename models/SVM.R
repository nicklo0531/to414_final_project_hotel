library(e1071)
library(caret)
library(pROC)
set.seed(12345)

# Kernels to compare
kernels <- c("linear", "radial", "sigmoid")
svm_results <- list()
svm_aucs <- c()

for (k in kernels) {
  svm_model <- svm(is_canceled ~ ., data = df_train, kernel = k, probability = TRUE)
  
  pred_class <- predict(svm_model, df_test)
  pred_prob  <- attr(predict(svm_model, df_test, probability = TRUE), "probabilities")[,2]
  
  cm <- confusionMatrix(pred_class, df_test$is_canceled)
  roc_obj <- roc(df_test$is_canceled, pred_prob)
  auc_val <- auc(roc_obj)
  
  svm_results[[k]] <- cm
  svm_aucs[k] <- auc_val
}

# Comparison table
comparison_svm <- data.frame(
  Kernel = names(svm_aucs),
  Accuracy = sapply(svm_results, function(x) x$overall["Accuracy"]),
  AUC = svm_aucs
)