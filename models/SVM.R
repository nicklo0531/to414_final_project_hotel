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
  
  pred_class <- predict(svm_model, df_test)
  pred_class <- factor(pred_class, levels = levels(df_test$is_canceled))
  
  prob_obj <- predict(svm_model, df_test, probability = TRUE)
  pred_prob <- attr(prob_obj, "probabilities")
  
  # Ensure positive class ("1") is used regardless of column order
  pos_col <- which(colnames(pred_prob) == "1")
  pred_prob <- pred_prob[, pos_col]
  
  cm <- confusionMatrix(pred_class, df_test$is_canceled)
  roc_obj <- roc(df_test$is_canceled, pred_prob)
  auc_val <- auc(roc_obj)
  
  svm_results[[k]] <- cm
  svm_aucs[k] <- auc_val
}

comparison_svm <- data.frame(
  Kernel = kernels,
  Accuracy = sapply(svm_results, function(x) x$overall["Accuracy"]),
  AUC = as.numeric(svm_aucs)
)
