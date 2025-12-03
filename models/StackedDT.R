## StackedCostMatrix.R
## Build 6 base models, stack them, and apply a cost matrix

library(caret)
library(neuralnet)
library(randomForest)
library(e1071)
library(class)
library(rpart)
library(pROC)
library(dplyr)

# Load train/test data from the shared preprocessed files
df_train <- read.csv("data/hotel_train.csv")
df_test  <- read.csv("data/hotel_test.csv")

y_train <- df_train$is_canceled
y_test  <- df_test$is_canceled

X_train <- df_train[, !(names(df_train) %in% "is_canceled"), drop = FALSE]
X_test  <- df_test[,  !(names(df_test)  %in% "is_canceled"), drop = FALSE]

############################################
# 1. Base models: probabilities on train + test
############################################

## Logistic regression (simple full model for stacking)
log_model <- glm(is_canceled ~ ., data = df_train, family = "binomial")
log_train_prob <- predict(log_model, df_train, type = "response")
log_test_prob  <- predict(log_model, df_test,  type = "response")

## ANN (same idea as your ANN writeup, but also get train preds)
set.seed(12345)
df_train_small <- df_train[sample(nrow(df_train), size = 0.3 * nrow(df_train)), ]

ann_model <- neuralnet(
  is_canceled ~ .,
  data     = df_train_small,
  lifesign = "minimal",
  stepmax  = 1e6,
  hidden   = c(3, 2)
)

# Probabilities for train/test (neuralnet needs the same columns as training)
ann_train_prob <- as.numeric(predict(ann_model, df_train))
ann_test_prob  <- as.numeric(predict(ann_model, df_test))

## KNN (k = 5, use prob=TRUE to recover P(Y=1))
train_y_fac <- factor(df_train$is_canceled, levels = c(0, 1))
test_y_fac  <- factor(df_test$is_canceled,  levels = c(0, 1))

knn_pred_test <- knn(
  train = X_train,
  test  = X_test,
  cl    = train_y_fac,
  k     = 5,
  prob  = TRUE
)

knn_prob_attr <- attr(knn_pred_test, "prob")
knn_test_prob <- ifelse(knn_pred_test == "1", knn_prob_attr, 1 - knn_prob_attr)

# For stacking, we also need train-side probs (KNN on train vs train)
knn_pred_train <- knn(
  train = X_train,
  test  = X_train,
  cl    = train_y_fac,
  k     = 5,
  prob  = TRUE
)
knn_train_attr <- attr(knn_pred_train, "prob")
knn_train_prob <- ifelse(knn_pred_train == "1", knn_train_attr, 1 - knn_train_attr)

## Decision Tree (tuned rpart, similar to your writeup)
set.seed(12345)
df_train_dt <- df_train
df_test_dt  <- df_test
df_train_dt$is_canceled <- as.factor(df_train_dt$is_canceled)
df_test_dt$is_canceled  <- as.factor(df_test_dt$is_canceled)

cp_grid <- expand.grid(cp = seq(0.001, 0.05, length.out = 10))
dtuned <- train(
  is_canceled ~ ., data = df_train_dt,
  method     = "rpart",
  trControl  = trainControl(method = "cv", number = 5),
  tuneGrid   = cp_grid
)
best_cp <- dtuned$bestTune$cp

dt_model <- rpart(
  is_canceled ~ .,
  data    = df_train_dt,
  method  = "class",
  control = rpart.control(cp = best_cp)
)

dt_train_prob <- predict(dt_model, df_train_dt, type = "prob")[, "1"]
dt_test_prob  <- predict(dt_model, df_test_dt,  type = "prob")[, "1"]

## Random Forest (ntree = 800 from your RF writeup)
set.seed(123)
rf_model <- randomForest(
  as.factor(is_canceled) ~ .,
  data  = df_train,
  ntree = 800
)

rf_train_prob <- predict(rf_model, df_train, type = "prob")[, "1"]
rf_test_prob  <- predict(rf_model, df_test,  type = "prob")[, "1"]

## SVM (linear kernel, as chosen in your SVM writeup)
set.seed(12345)
df_train_svm <- df_train
df_test_svm  <- df_test
df_train_svm$is_canceled <- as.factor(df_train_svm$is_canceled)
df_test_svm$is_canceled  <- as.factor(df_test_svm$is_canceled)

svm_linear <- svm(
  is_canceled ~ .,
  data        = df_train_svm,
  kernel      = "linear",
  probability = TRUE
)

svm_train_prob <- attr(
  predict(svm_linear, df_train_svm, probability = TRUE),
  "probabilities"
)[, "1"]

svm_test_prob <- attr(
  predict(svm_linear, df_test_svm, probability = TRUE),
  "probabilities"
)[, "1"]

############################################
# 2. Stacked model: meta-logistic on base probs
############################################

stack_train <- data.frame(
  log        = log_train_prob,
  ann        = ann_train_prob,
  knn        = knn_train_prob,
  dt         = dt_train_prob,
  rf         = rf_train_prob,
  svm        = svm_train_prob,
  is_canceled = y_train
)

stack_test <- data.frame(
  log = log_test_prob,
  ann = ann_test_prob,
  knn = knn_test_prob,
  dt  = dt_test_prob,
  rf  = rf_test_prob,
  svm = svm_test_prob
)

stack_model <- glm(is_canceled ~ ., data = stack_train, family = "binomial")
stack_test_prob <- predict(stack_model, stack_test, type = "response")

############################################
# 3. Cost matrix + threshold sweep for all models
############################################

# Cost matrix: FPs are 5x worse than FNs
C_FP <- 1200  # false positive: predict cancel, guest shows (overbook/walk)
C_FN <- 500  # false negative: predict show, guest cancels (empty room)

evaluate_at_threshold <- function(y_true, p_hat, threshold, C_FP, C_FN) {
  preds <- ifelse(p_hat >= threshold, 1, 0)
  
  TP <- sum(preds == 1 & y_true == 1)
  FP <- sum(preds == 1 & y_true == 0)
  FN <- sum(preds == 0 & y_true == 1)
  TN <- sum(preds == 0 & y_true == 0)
  
  cm <- confusionMatrix(
    factor(preds, levels = c(0, 1)),
    factor(y_true, levels = c(0, 1)),
    positive = "1"
  )
  
  cost     <- C_FP * FP + C_FN * FN
  avg_cost <- cost / length(y_true)
  
  data.frame(
    Threshold   = threshold,
    Accuracy    = cm$overall["Accuracy"],
    Kappa       = cm$overall["Kappa"],
    Sensitivity = cm$byClass["Sensitivity"],
    Precision   = cm$byClass["Pos Pred Value"],
    Cost        = cost,
    AvgCost     = avg_cost
  )
}

sweep_model <- function(y_true, p_hat, model_name,
                        C_FP, C_FN,
                        thresholds = seq(0.01, 0.99, by = 0.01)) {
  rows <- lapply(thresholds, function(t) {
    evaluate_at_threshold(y_true, p_hat, t, C_FP, C_FN)
  })
  out <- do.call(rbind, rows)
  out$Model <- model_name
  out
}

results_all <- rbind(
  sweep_model(y_test, log_test_prob,   "Logistic",            C_FP, C_FN),
  sweep_model(y_test, ann_test_prob,   "ANN",                 C_FP, C_FN),
  sweep_model(y_test, knn_test_prob,   "KNN",                 C_FP, C_FN),
  sweep_model(y_test, dt_test_prob,    "Decision Tree",       C_FP, C_FN),
  sweep_model(y_test, rf_test_prob,    "Random Forest",       C_FP, C_FN),
  sweep_model(y_test, svm_test_prob,   "SVM (linear)",        C_FP, C_FN),
  sweep_model(y_test, stack_test_prob, "Stacked (meta-logit)", C_FP, C_FN)
)

best_by_cost <- results_all %>%
  group_by(Model) %>%
  slice_min(AvgCost, with_ties = FALSE) %>%
  ungroup()

# So when you source this file, you can just look at:
#   best_by_cost   -> best threshold + cost per model
#   results_all    -> full grid over thresholds
