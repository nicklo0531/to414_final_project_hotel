library(rpart)
library(rpart.plot)
library(caret)
library(C50)
library(pROC)
set.seed(12345)


df_train$is_canceled <- as.factor(df_train$is_canceled)
df_test$is_canceled  <- as.factor(df_test$is_canceled)


# Initial rpart
rpart_model <- rpart(is_canceled ~ ., data = df_train, method = "class", control = rpart.control(cp = 0.01))
rpart_pred_class <- predict(rpart_model, df_test, type = "class")
rpart_pred_prob  <- predict(rpart_model, df_test, type = "prob")[,2]
cm_rpart <- confusionMatrix(rpart_pred_class, df_test$is_canceled)
roc_rpart <- roc(df_test$is_canceled, rpart_pred_prob)
auc_rpart <- auc(roc_rpart)


# Tune rpart
cp_grid <- expand.grid(cp = seq(0.001, 0.05, length.out = 10))
dtuned <- train(is_canceled ~ ., data = df_train, method = "rpart",
                trControl = trainControl(method = "cv", number = 5), tuneGrid = cp_grid)
best_cp <- dtuned$bestTune$cp

tuned_model <- rpart(is_canceled ~ ., data = df_train, method = "class", control = rpart.control(cp = best_cp))
pred_dt_class <- predict(tuned_model, df_test, type = "class")
pred_dt_prob <- predict(tuned_model, df_test, type = "prob")[, 2]

cm_dt <- confusionMatrix(pred_dt_class, df_test$is_canceled)

roc_dt <- roc(df_test$is_canceled, pred_dt_prob)
auc_dt <- auc(roc_dt)


# C5.0
c50_model <- C5.0(x = df_train[, -1], y = df_train$is_canceled, trials = 10)
pred_c50_class <- predict(c50_model, df_test)
pred_c50_prob <- predict(c50_model, df_test, type = "prob")[, 2]
cm_c50 <- confusionMatrix(pred_c50_class, df_test$is_canceled)
roc_c50 <- roc(df_test$is_canceled, pred_c50_prob)
auc_c50 <- auc(roc_c50)

# Comparison
comparison <- data.frame(
  Model = c("Decision Tree (tuned)", "C5.0"),
  Accuracy = c(cm_dt$overall["Accuracy"], cm_c50$overall["Accuracy"]),
  AUC = c(auc_dt, auc_c50)
)




