library(neuralnet)
library(caret)
library(ggplot2)

set.seed(12345)

# Sample 30% of training data for faster ANN training
df_train_small <- df_train[sample(nrow(df_train), size = 0.3 * nrow(df_train)), ]

# Train neural network with two hidden layers
ann_model <- neuralnet(
  is_canceled ~ ., 
  data = df_train_small, 
  lifesign = "minimal",
  stepmax = 1e6, 
  hidden = c(3,2)
)

# Predict probabilities
ann_train_prob <- predict(ann_model, df_train)
ann_test_prob  <- predict(ann_model, df_test)

# Save probabilities for stacking
saveRDS(ann_train_prob, "RDS/ann_train_prob.rds")
saveRDS(ann_test_prob,  "RDS/ann_test_prob.rds")

# Evaluate on test data using threshold = 0.5
ann_bin_pred <- ifelse(ann_test_prob >= 0.5, 1, 0)
ann_cm <- confusionMatrix(
  factor(ann_bin_pred, levels = c(0,1)), 
  factor(df_test$is_canceled, levels = c(0,1)), 
  positive = "1"
)
print(ann_cm)
