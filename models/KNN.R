# Load packages and set up the outcome/ predictors
# To do the KNN model, we start downloading the KNN package and evaluation tools. Then, we turn the outcome "is_canceled" into a factor with levels 0/1 for both the train and test data. We split the data into "train_x" and "test_x" for all predictor variables, and "train_y" and "test_y" for just the cancellation outcome.

library(class)
library(caret)

train_y <- factor(df_train$is_canceled, levels = c(0,1))
test_y  <- factor(df_test$is_canceled,  levels = c(0,1))

train_x <- df_train[, names(df_train) != "is_canceled"]
test_x  <- df_test[,  names(df_test)  != "is_canceled"]

# Choosing which K values to test and creating a results table
# For this step we tune K by trying odd values from 3 to 31. We create a table to compare the accuracy, specificity and sensitivity for each k
ks <- seq(3, 31, 2)
results <- data.frame(
  k = ks,
  Accuracy = NA,
  Specificity = NA,
  Sensitivity = NA
)

# Tune K by looping over all candidate values
# For each candidate K, we will fit a KNN model and predict on the test set. The accuracy, specificity and sensitivity will be shown thanks to the confusion matrix comparing predictions to the true labels. Therefore, we will be able to see a performance profile of KNN as K changes
for (i in seq_along(ks)) {
  pred_i <- knn(train = train_x, test = test_x, cl = train_y, k = ks[i])
  pred_i <- factor(pred_i, levels = c("0","1"))
  
  cm_i <- confusionMatrix(pred_i, test_y, positive = "1")
  results$Accuracy[i] <- cm_i$overall["Accuracy"]
  results$Specificity[i] <- cm_i$byClass["Specificity"]
  results$Sensitivity[i] <- cm_i$byClass["Sensitivity"]
}

# Selecting the K with the maximum specificity
# Instead of choosing K by accuracy, for our business goal we will choose the K with the highest specificity. This is because false positives, predicting cancel but the guest ends up showing up, are very expensive. We want a model that can correctly identify non-cancellations.
best_k <- results$k[which.max(results$Specificity)]

# Fit final KNN with best K and get predictions
# We now refit KNN using the chosen K. We request "prob= TRUE" so that we can later recover probabilities. "pred_knn" will be the final class prediction (0 or 1) for every reservation in the test set
pred_knn <- knn(train_x, test_x, cl = train_y, k = best_k, prob = TRUE)
pred_knn <- factor(pred_knn, levels = c("0","1"))

# Convert KNN "prob" attribute into cancellation probabilities
# The KNN stores the probability of the winning class as the "prob" attribute. We convert this into a true probability of cancellation. If the model predicted 1, it will use "prob", and if it predicts 0, it will use "1- prob". If the attribute is missing, it will fall back to 0.5 as neutral default. These probabilities will be needed later for the stacked model.
knn_prob_attr <- attr(pred_knn, "prob")
knn_prob_attr <- attr(pred_knn, "prob")
if (is.null(knn_prob_attr) || length(knn_prob_attr) == 0) {
  knn_test_prob <- rep(0.5, length(pred_knn))
} else {
  knn_test_prob <- ifelse(pred_knn == "1", knn_prob_attr, 1 - knn_prob_attr)
}

# Train-side probabilities for stacking
# We run KNN on the training data itself to get the KNN probabilities for each training booking. We convert "prob" into the probability of cancelling. These probabilities will become input features for the second- level or stacked model
pred_knn_train <- knn(train_x, train_x, cl = train_y, k = best_k, prob = TRUE)
knn_prob_attr_train <- attr(pred_knn_train, "prob")
if (is.null(knn_prob_attr_train) || length(knn_prob_attr_train) == 0) {
  knn_train_prob <- rep(0.5, length(pred_knn_train))
} else {
  knn_train_prob <- ifelse(pred_knn_train == "1", knn_prob_attr_train, 1 - knn_prob_attr_train)
}

# Save probabilities for the stacked model
# We save both train and test KNN probabilities as .rds files. Therefore, the stacking script will be able to read them later and combine them with the probabilities from the other models
saveRDS(knn_train_prob, "RDS/knn_train_prob.rds")
saveRDS(knn_test_prob,  "RDS/knn_test_prob.rds")

# Compute final confusion matrix for reporting
# We build the final confusion matrix for KNN using the chosen K. It will provide information on accuracy, sensitivity, specificity, precision, kappa and other variables. 
cm_knn <- confusionMatrix(pred_knn, test_y, positive = "1")
cm_knn