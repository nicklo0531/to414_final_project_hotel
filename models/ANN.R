library(neuralnet)
library(caret)
df_train <- read.csv("../data/hotel_train.csv")
df_test <- read.csv("../data/hotel_test.csv")

set.seed(12345)
df_train_small <- df_train[sample(nrow(df_train), size = 0.3 * nrow(df_train)), ]

ann_model <- neuralnet(is_canceled ~ ., data = df_train_small, lifesign = "minimal", stepmax = 1e6, hidden = c(3,2))
ann_pred <- predict(ann_model, df_test)

ann_bin_pred <- ifelse(ann_pred >= 0.5, 1, 0)
ann_cm <- confusionMatrix(as.factor(ann_bin_pred), as.factor(df_test$is_canceled), positive = "1")
ann_cm 