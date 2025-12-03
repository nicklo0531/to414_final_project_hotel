

library(class)
library(caret)

# Extract predictors and classes
train_x <- df_train %>% select(-is_canceled)
test_x  <- df_test  %>% select(-is_canceled)

train_y <- as.factor(df_train$is_canceled)
test_y  <- as.factor(df_test$is_canceled)

# Make sure factor levels align correctly (0 and 1)
train_y <- factor(train_y, levels = c(0,1))
test_y  <- factor(test_y,  levels = c(0,1))

# Choose k
k_val <- 5   # you can tune this later

# Run KNN
pred_knn <- knn(
  train = train_x,
  test  = test_x,
  cl    = train_y,
  k     = k_val
)

# Evaluate
pred_knn <- factor(pred_knn, levels = c("0","1"))

cm_knn <- confusionMatrix(pred_knn, test_y, positive = "1")
cm_knn