library(caret)
library(rpart)
library(rpart.plot)
library(C50)
library(randomForest)
library(e1071)
library(class)
library(pROC)

# Already have precomputed probabilities and y_test and y_train from first-level models from base_compariso

# Build stacked data frame for second-level model

set.seed(12345)
meta_idx <- sample(1:nrow(df_test), size = 0.5 * nrow(df_test))

df_meta  <- df_test[meta_idx, ]
df_final <- df_test[-meta_idx, ]

y_meta  <- y_test[meta_idx]
y_final <- y_test[-meta_idx]

# Split each model's probabilities the same way
log_meta_prob  <- log_test_prob[meta_idx]
log_final_prob <- log_test_prob[-meta_idx]

knn_meta_prob  <- knn_test_prob[meta_idx]
knn_final_prob <- knn_test_prob[-meta_idx]

c50_meta_prob  <- c50_test_prob[meta_idx]
c50_final_prob <- c50_test_prob[-meta_idx]

rf_meta_prob   <- rf_test_prob[meta_idx]
rf_final_prob  <- rf_test_prob[-meta_idx]

svm_meta_prob  <- svm_test_prob[meta_idx]
svm_final_prob <- svm_test_prob[-meta_idx]

# 4. BUILD STACKED TRAIN / TEST FOR META-MODEL

stack_train <- data.frame(
  log = log_meta_prob,
  knn = knn_meta_prob,
  c50 = c50_meta_prob,
  rf  = rf_meta_prob,
  svm = svm_meta_prob,
  is_canceled = y_meta
)

stack_test <- data.frame(
  log = log_final_prob,
  knn = knn_final_prob,
  c50 = c50_final_prob,
  rf  = rf_final_prob,
  svm = svm_final_prob
)


# 5. TRAIN SECOND-LEVEL RPART

set.seed(12345)
stack_dt <- rpart(
  is_canceled ~ .,
  data    = stack_train,
  method  = "class",
  control = rpart.control(cp = 0.1)
)

stack_pred_prob  <- predict(stack_dt, stack_test, type="prob")[,"1"]
stack_pred_class <- predict(stack_dt, stack_test, type="class")

stack_cm  <- confusionMatrix(stack_pred_class, factor(y_final))
stack_roc <- roc(y_final, stack_pred_prob)
stack_auc <- auc(stack_roc)

rpart.plot(stack_dt)