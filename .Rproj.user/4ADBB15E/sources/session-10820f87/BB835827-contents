library(class)
library(caret)

# Ensure outcome is a factor with consistent levels
train_y <- factor(df_train$is_canceled, levels = c(0,1))
test_y  <- factor(df_test$is_canceled,  levels = c(0,1))

train_x <- df_train[, names(df_train) != "is_canceled"]
test_x  <- df_test[,  names(df_test)  != "is_canceled"]

# Sequence of k values to try
ks <- seq(3, 31, 2)
results <- data.frame(
  k = ks,
  Accuracy = NA,
  Specificity = NA,
  Sensitivity = NA
)

# Loop over k values
for (i in seq_along(ks)) {
  pred_i <- knn(train = train_x, test = test_x, cl = train_y, k = ks[i])
  pred_i <- factor(pred_i, levels = c("0","1"))
  
  cm_i <- confusionMatrix(pred_i, test_y, positive = "1")
  results$Accuracy[i] <- cm_i$overall["Accuracy"]
  results$Specificity[i] <- cm_i$byClass["Specificity"]
  results$Sensitivity[i] <- cm_i$byClass["Sensitivity"]
}

results

# Find best k for overall accuracy and specificity
best_k_acc  <- results$k[which.max(results$Accuracy)]
best_k_spec <- results$k[which.max(results$Specificity)]

# Final KNN optimized for accuracy
pred_knn_acc <- knn(train_x, test_x, cl = train_y, k = best_k_acc)
pred_knn_acc <- factor(pred_knn_acc, levels = c("0","1"))
cm_knn_acc <- confusionMatrix(pred_knn_acc, test_y, positive = "1")

# Final KNN optimized for specificity
pred_knn_spec <- knn(train_x, test_x, cl = train_y, k = best_k_spec)
pred_knn_spec <- factor(pred_knn_spec, levels = c("0","1"))
cm_knn_spec <- confusionMatrix(pred_knn_spec, test_y, positive = "1")
  