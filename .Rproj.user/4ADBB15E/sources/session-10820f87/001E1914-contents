library(rpart)
library(rpart.plot)
library(caret)
library(C50)
library(pROC)
set.seed(12345)

# Ensure outcome is a factor with consistent levels
df_train$is_canceled <- factor(df_train$is_canceled, levels = c(0,1))
df_test$is_canceled  <- factor(df_test$is_canceled,  levels = c(0,1))

# Initial rpart
rpart_model <- rpart(is_canceled ~ ., data = df_train, method = "class", control = rpart.control(cp = 0.01))
rpart_pred_class <- factor(predict(rpart_model, df_test, type = "class"), levels = c(0,1))
rpart_pred_prob  <- predict(rpart_model, df_test, type = "prob")[, "1"]

cm_rpart <- confusionMatrix(rpart_pred_class, df_test$is_canceled)
roc_rpart <- roc(df_test$is_canceled, rpart_pred_prob)
auc_rpart <- auc(roc_rpart)

# Tuned rpart
cp_grid <- expand.grid(cp = seq(0.001, 0.05, length.out = 10))
dtuned <- train(
  is_canceled ~ ., data = df_train,
  method    = "rpart",
  trControl = trainControl(method = "cv", number = 5),
  tuneGrid  = cp_grid
)
best_cp <- dtuned$bestTune$cp

tuned_model <- rpart(
  is_canceled ~ ., data = df_train,
  method  = "class",
  control = rpart.control(cp = best_cp)
)

# Train/test probabilities for stacking
dt_train_prob <- predict(tuned_model, df_train, type = "prob")[, "1"]
dt_test_prob  <- predict(tuned_model, df_test,  type = "prob")[, "1"]

saveRDS(dt_train_prob, "RDS/dt_train_prob.rds")
saveRDS(dt_test_prob,  "RDS/dt_test_prob.rds")
saveRDS(tuned_model,   "RDS/rpart_tuned_model.rds")

# C5.0
c50_model <- C5.0(x = df_train[, -which(names(df_train) == "is_canceled")], 
                  y = df_train$is_canceled, trials = 10)

# Probabilities for stacking
c50_train_prob <- predict(c50_model, df_train, type = "prob")[, "1"]
c50_test_prob  <- predict(c50_model, df_test,  type = "prob")[, "1"]

saveRDS(c50_train_prob, "RDS/c50_train_prob.rds")
saveRDS(c50_test_prob,  "RDS/c50_test_prob.rds")
saveRDS(c50_model,      "RDS/c50_model.rds")

# Metrics & plots
pred_c50_class <- factor(predict(c50_model, df_test), levels = c(0,1))
cm_c50 <- confusionMatrix(pred_c50_class, df_test$is_canceled)
roc_c50 <- roc(df_test$is_canceled, c50_test_prob)
auc_c50 <- auc(roc_c50)

# Comparison
comparison <- data.frame(
  Model = c("Decision Tree (tuned)", "C5.0"),
  Accuracy = c(cm_rpart$overall["Accuracy"], cm_c50$overall["Accuracy"]),
  AUC      = c(auc_rpart, auc_c50)
)

print(comparison)
print(cm_c50)
rpart.plot(tuned_model)
