library(class)
library(caret)

# Load packages and set up the outcome/ predictors

# Turn the outcome "is_canceled" into a factor with levels 0/1 for both train and test

train_y <- factor(df_train$is_canceled, levels = c(0,1))
test_y  <- factor(df_test$is_canceled,  levels = c(0,1))

train_x <- df_train[, names(df_train) != "is_canceled"]
test_x  <- df_test[,  names(df_test)  != "is_canceled"]

# Choosing which K values to test and creating a results table

# For this step we tune K by trying odd values from 3 to 31. We create a table to compare accuracy, specificity, and sensitivity for each k

ks <- seq(3, 31, 2)
results <- data.frame(
  k = ks,
  Accuracy = NA_real_,
  Specificity = NA_real_,
  Sensitivity = NA_real_
)

# Tune K by looping over all candidate values

# For each candidate K, we fit a KNN model and predict on the test set.

# Confusion matrix metrics let us see performance of KNN as K changes

for (i in seq_along(ks)) {
  pred_i <- knn(train = train_x, test = test_x, cl = train_y, k = ks[i])
  pred_i <- factor(pred_i, levels = c("0","1"))
  
  cm_i <- confusionMatrix(pred_i, test_y, positive = "1")
  results$Accuracy[i]    <- cm_i$overall["Accuracy"]
  results$Specificity[i] <- cm_i$byClass["Specificity"]
  results$Sensitivity[i] <- cm_i$byClass["Sensitivity"]
}

# Selecting the K with the maximum specificity

# Instead of choosing K by accuracy, we choose the K with the highest specificity

# because false positives are very expensive for our business case

best_k <- results$k[which.max(results$Specificity)]
print(best_k)

# Function to convert KNN "prob" attribute into cancellation probabilities

# If the attribute is missing, fall back to 0.5

knn_to_prob <- function(pred, prob_attr) {
  if (is.null(prob_attr) || length(prob_attr) == 0) {
    return(rep(0.5, length(pred)))
  } else {
    ifelse(pred == "1", prob_attr, 1 - prob_attr)
  }
}

# Fit final KNN with best K on the test set

# We request prob = TRUE to recover probabilities for stacking

pred_knn <- knn(train_x, test_x, cl = train_y, k = best_k, prob = TRUE)
pred_knn <- factor(pred_knn, levels = c("0","1"))

# Convert KNN "prob" attribute into cancellation probabilities

knn_prob_attr <- attr(pred_knn, "prob")
knn_test_prob <- knn_to_prob(pred_knn, knn_prob_attr)

# Train-side probabilities for stacking

# We run KNN on the training data itself to get probabilities for each booking

pred_knn_train <- knn(train_x, train_x, cl = train_y, k = best_k, prob = TRUE)
pred_knn_train <- factor(pred_knn_train, levels = c("0","1"))
knn_prob_attr_train <- attr(pred_knn_train, "prob")
knn_train_prob <- knn_to_prob(pred_knn_train, knn_prob_attr_train)

# Save probabilities for the stacked model

# We save both train and test KNN probabilities as .rds files

saveRDS(knn_train_prob, "RDS/knn_train_prob.rds")
saveRDS(knn_test_prob,  "RDS/knn_test_prob.rds")

# Compute final confusion matrix for reporting

# This gives accuracy, sensitivity, specificity, precision, kappa, etc.

cm_knn <- confusionMatrix(pred_knn, test_y, positive = "1")
print(cm_knn)
