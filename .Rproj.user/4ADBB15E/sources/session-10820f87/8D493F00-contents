############################################################
# Random Forest Cost-Based Hyperparameter Search
# Optimized for Speed + Clear Visuals
############################################################

library(randomForest)
library(caret)
library(dplyr)
library(ggplot2)

# Cost matrix
C_FP <- 1200
C_FN <- 500

############################################################
# Hyperparameter grid (kept small for speed but useful)
############################################################

ntree_values <- seq(100, 1000, by = 100)
nodesize_values <- seq(1, 5, by = 1)

param_grid <- expand.grid(
  ntree    = ntree_values,
  nodesize = nodesize_values,
  Accuracy = NA_real_,
  Sensitivity = NA_real_,
  Precision = NA_real_,
  AvgCost = NA_real_,
  stringsAsFactors = FALSE
)

############################################################
# Hyperparameter search loop (optimized order for speed)
############################################################

for (i in seq_len(nrow(param_grid))) {
  nt     <- param_grid$ntree[i]
  ns     <- param_grid$nodesize[i]
  
  # Train RF
  rf_model <- randomForest(
    as.factor(is_canceled) ~ ., 
    data     = df_train,
    ntree    = nt,
    nodesize = ns
  )
  
  # Predictions
  preds_class <- predict(rf_model, df_test, type = "class")
  
  # Confusion matrix
  final_cm <- confusionMatrix(
    factor(preds_class, levels = c(0,1)),
    factor(df_test$is_canceled, levels = c(0,1)),
    positive = "1"
  )
  
  # Cost calculation
  FP <- sum(preds_class == 1 & df_test$is_canceled == 0)
  FN <- sum(preds_class == 0 & df_test$is_canceled == 1)
  total_cost <- C_FP * FP + C_FN * FN
  avg_cost <- total_cost / nrow(df_test)
  
  # Store results
  param_grid$Accuracy[i]    <- final_cm$overall["Accuracy"]
  param_grid$Sensitivity[i] <- final_cm$byClass["Sensitivity"]
  param_grid$Precision[i]   <- final_cm$byClass["Pos Pred Value"]
  param_grid$AvgCost[i]     <- avg_cost
}

############################################################
# Identify best parameters by cost
############################################################

best_params <- param_grid %>% filter(AvgCost == min(AvgCost))
print(best_params)

############################################################
# PLOT 1: Smoother line plot by nodesize group
############################################################

param_grid$nodesize <- as.factor(param_grid$nodesize)
