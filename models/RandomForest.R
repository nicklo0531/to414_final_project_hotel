library(randomForest)
library(caret)
df_train <- read.csv("../data/hotel_train.csv")
df_test <- read.csv("../data/hotel_test.csv")

rf_model <- randomForest(type='classification', as.factor(is_canceled) ~ ., data = df_train, ntree=500, weights=NULL)
rf_pred <- predict(rf_model, df_test)

rf_cm <- confusionMatrix(as.factor(rf_pred), as.factor(df_test$is_canceled), positive = "1")
rf_cm

