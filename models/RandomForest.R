library(randomForest)
library(caret)
library(ggplot2)

# Optimize ntree over a sequence
ntree_list <- seq(100, 1000, by = 100)
results <- data.frame(
  ntree = ntree_list,
  Accuracy = NA,
  Sensitivity = NA,
  Precision = NA
)

set.seed(123)

for (i in seq_along(ntree_list)) {
  nt <- ntree_list[i]
  
  rf_temp <- randomForest(
    is_canceled ~ .,
    data = df_train,
    ntree = nt
  )
  
  temp_pred <- predict(rf_temp, df_test)
  
  cm <- confusionMatrix(
    factor(temp_pred, levels = c(0,1)),
    factor(df_test$is_canceled, levels = c(0,1)),
    positive = "1"
  )
  
  results$Accuracy[i]    <- cm$overall["Accuracy"]
  results$Sensitivity[i] <- cm$byClass["Sensitivity"]
  results$Precision[i]   <- cm$byClass["Pos Pred Value"]
}

# Plot optimization process
results_long <- reshape2::melt(results, id.vars = "ntree",
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

# Print the plot (will work when sourced in main Rmd)
print(rf_opt_plot)

# Train final RF model
rf_model <- randomForest(
  is_canceled ~ .,
  data = df_train,
  ntree = 800
)

rf_pred <- predict(rf_model, df_test)

rf_cm <- confusionMatrix(
  factor(rf_pred, levels = c(0,1)),
  factor(df_test$is_canceled, levels = c(0,1)),
  positive = "1"
)