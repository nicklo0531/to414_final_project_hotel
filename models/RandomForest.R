library(randomForest)
library(caret)
library(ggplot2)
library(reshape2)
library(pROC) 

# Optimize ntree over a sequence

ntree_list <- seq(100, 1000, by = 100)
results <- data.frame(
  ntree = ntree_list,
  Accuracy = NA,
  Sensitivity = NA,
  Precision = NA,
  AUC = NA
)

set.seed(12345)

for (i in seq_along(ntree_list)) {
  nt <- ntree_list[i]
  
  rf_temp <- randomForest(
    is_canceled ~ .,
    data = df_train,
    ntree = nt
  )
  
  temp_pred <- predict(rf_temp, df_test)
  temp_prob <- predict(rf_temp, df_test, type = "prob")[, "1"]
  
  cm <- confusionMatrix(
    factor(temp_pred, levels = c(0,1)),
    factor(df_test$is_canceled, levels = c(0,1)),
    positive = "1"
  )
  
  results$Accuracy[i]    <- cm$overall["Accuracy"]
  results$Sensitivity[i] <- cm$byClass["Sensitivity"]
  results$Precision[i]   <- cm$byClass["Pos Pred Value"]
  
  # Compute AUC
  roc_obj <- roc(df_test$is_canceled, temp_prob)
  results$AUC[i] <- auc(roc_obj)
}

# Plot metrics including AUC
results_long <- melt(results, id.vars = "ntree",
                     variable.name = "Metric",
                     value.name = "Value")

rf_opt_plot <- ggplot(results_long, aes(x = ntree, y = Value, color = Metric)) +
  geom_line(size = 1.2) +
  geom_point(size = 2) +
  theme_minimal() +
  labs(title = "Random Forest Performance vs ntree",
       x = "Number of Trees (ntree)",
       y = "Metric Value") +
  scale_color_brewer(palette = "Set1")

print(rf_opt_plot)

# Choose final ntree dynamically (best AUC)

final_ntree <- results$ntree[which.max(results$AUC)]
print(paste("Selected ntree based on best AUC:", final_ntree))

# Train final RF model with chosen ntree

rf_model <- randomForest(
  is_canceled ~ .,
  data = df_train,
  ntree = final_ntree
)

# Save model and probabilities for stacking
saveRDS(rf_model, "RDS/rf_model.rds")

rf_train_prob <- predict(rf_model, df_train, type = "prob")[, "1"]
rf_test_prob  <- predict(rf_model, df_test,  type = "prob")[, "1"]

saveRDS(rf_train_prob, "RDS/rf_train_prob.rds")
saveRDS(rf_test_prob,  "RDS/rf_test_prob.rds")

rf_pred <- predict(rf_model, df_test)

rf_cm <- confusionMatrix(
  factor(rf_pred, levels = c(0,1)),
  factor(df_test$is_canceled, levels = c(0,1)),
  positive = "1"
)

print(rf_cm)