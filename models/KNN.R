library(class)
library(caret)

# Ensure outcome is a factor with consistent levels
train_y <- factor(df_train$is_canceled, levels = c(0,1))
test_y  <- factor(df_test$is_canceled,  levels = c(0,1))

# Run KNN with k = 5
pred_knn <- knn(
  train = df_train[, -which(names(df_train) == "is_canceled")],
  test  = df_test[,  -which(names(df_test)  == "is_canceled")],
  cl    = train_y,
  k     = 5
)

# Evaluate
pred_knn <- factor(pred_knn, levels = c("0","1"))
cm_knn <- confusionMatrix(pred_knn, test_y, positive = "1")
cm_knn