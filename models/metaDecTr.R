library(caret)
library(rpart)
library(rpart.plot)
library(C50)
library(randomForest)
library(e1071)
library(class)

# --------------------------------------------------
# Load precomputed probabilities from first-level models
# --------------------------------------------------

log_train_prob <- readRDS("outputs/log_train_prob.rds")
log_test_prob  <- readRDS("outputs/log_test_prob.rds")

ann_train_prob <- readRDS("outputs/ann_train_prob.rds")
ann_test_prob  <- readRDS("outputs/ann_test_prob.rds")

knn_train_prob <- readRDS("outputs/knn_train_prob.rds")  # best specificity
knn_test_prob  <- readRDS("outputs/knn_test_prob.rds")

c50_train_prob <- readRDS("outputs/c50_train_prob.rds")
c50_test_prob  <- readRDS("outputs/c50_test_prob.rds")

rf_train_prob <- readRDS("outputs/rf_train_prob.rds")  # ntree=800
rf_test_prob  <- readRDS("outputs/rf_test_prob.rds")

svm_train_prob <- readRDS("outputs/svm_linear_train_prob.rds")
svm_test_prob  <- readRDS("outputs/svm_linear_test_prob.rds")

# Outcome
y_train <- df_train$is_canceled
y_test  <- df_test$is_canceled

# --------------------------------------------------
# Build stacked data frame for second-level model
# --------------------------------------------------

stack_train <- data.frame(
  log  = log_train_prob,
  ann  = ann_train_prob,
  knn  = knn_train_prob,
  c50  = c50_train_prob,
  rf   = rf_train_prob,
  svm  = svm_train_prob,
  is_canceled = y_train
)

stack_test <- data.frame(
  log = log_test_prob,
  ann = ann_test_prob,
  knn = knn_test_prob,
  c50 = c50_test_prob,
  rf  = rf_test_prob,
  svm = svm_test_prob
)

# --------------------------------------------------
# Train second-level rpart (decision tree) on stacked probs
# --------------------------------------------------

set.seed(12345)
stack_dt <- rpart(
  is_canceled ~ .,
  data    = stack_train,
  method  = "class",
  control = rpart.control(cp = 0.01)
)

# Predict probabilities on test set
stack_pred_class <- predict(stack_dt, stack_test, type = "class")
stack_pred_prob  <- predict(stack_dt, stack_test, type = "prob")[, "1"]

# Evaluate
stack_cm  <- confusionMatrix(stack_pred_class, y_test)
stack_roc <- pROC::roc(y_test, stack_pred_prob)
stack_auc <- pROC::auc(stack_roc)

# Plot tree
rpart.plot(stack_dt)

# Output metrics
stack_cm
stack_auc
