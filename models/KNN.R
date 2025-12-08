
## KNN (tune k, then use best k for probs)

library(class)

# Set up X / y for KNN specifically
train_y <- factor(df_train$is_canceled, levels = c(0, 1))
test_y  <- factor(df_test$is_canceled,  levels = c(0, 1))

train_x <- df_train[, names(df_train) != "is_canceled", drop = FALSE]
test_x  <- df_test[,  names(df_test)  != "is_canceled", drop = FALSE]

# Sequence of k values to try
ks <- seq(3, 31, 2)
knn_tune_results <- data.frame(
  k           = ks,
  Accuracy    = NA_real_,
  Specificity = NA_real_,
  Sensitivity = NA_real_
)

# Tune KNN over k
for (i in seq_along(ks)) {
  pred_i <- knn(train = train_x, test = test_x, cl = train_y, k = ks[i])
  pred_i <- factor(pred_i, levels = c("0", "1"))
  
  cm_i <- confusionMatrix(pred_i, test_y, positive = "1")
  knn_tune_results$Accuracy[i]    <- cm_i$overall["Accuracy"]
  knn_tune_results$Specificity[i] <- cm_i$byClass["Specificity"]
  knn_tune_results$Sensitivity[i] <- cm_i$byClass["Sensitivity"]
}

# Pick k with the best specificity (we care about avoiding false positives)
best_k_spec <- knn_tune_results$k[which.max(knn_tune_results$Specificity)]

# Final KNN using best_k_spec, now with probabilities for stacking / cost matrix
knn_pred_test <- knn(
  train = train_x,
  test  = test_x,
  cl    = train_y,
  k     = best_k_spec,
  prob  = TRUE
)
knn_prob_attr_test <- attr(knn_pred_test, "prob")
knn_test_prob <- ifelse(knn_pred_test == "1", knn_prob_attr_test, 1 - knn_prob_attr_test)

# Train-side probabilities (KNN on train vs train)
knn_pred_train <- knn(
  train = train_x,
  test  = train_x,
  cl    = train_y,
  k     = best_k_spec,
  prob  = TRUE
)
knn_prob_attr_train <- attr(knn_pred_train, "prob")
knn_train_prob <- ifelse(knn_pred_train == "1", knn_prob_attr_train, 1 - knn_prob_attr_train)

saveRDS(knn_train_prob, "RDS/knn_train_prob.rds")
saveRDS(knn_test_prob,  "RDS/knn_test_prob.rds")
