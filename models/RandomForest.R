library(randomForest)
library(caret)

rf_model <- randomForest(
  is_canceled ~ .,
  data = df_train,
  ntree = 800
)

rf_pred <- predict(rf_model, df_test)

rf_cm <- confusionMatrix(
  rf_pred,
  df_test$is_canceled,
  positive = "1"
)