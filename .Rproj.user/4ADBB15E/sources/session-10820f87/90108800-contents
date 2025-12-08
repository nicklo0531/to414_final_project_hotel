library(caret)
library(rpart)
library(rpart.plot)
library(C50)
library(randomForest)
library(e1071)
library(class)

log_train_prob <- readRDS("RDS/log_train_prob.rds")
log_test_prob  <- readRDS("RDS/log_test_prob.rds")

knn_train_prob <- readRDS("RDS/knn_train_prob.rds")  # best specificity
knn_test_prob  <- readRDS("RDS/knn_test_prob.rds")

c50_train_prob <- readRDS("RDS/c50_train_prob.rds")
c50_test_prob  <- readRDS("RDS/c50_test_prob.rds")

rf_train_prob <- readRDS("RDS/rf_train_prob.rds")  # ntree=800
rf_test_prob  <- readRDS("RDS/rf_test_prob.rds")

svm_train_prob <- readRDS("RDS/svm_linear_train_prob.rds")
svm_test_prob  <- readRDS("RDS/svm_linear_test_prob.rds")

# Outcome
y_train <- df_train$is_canceled
y_test  <- df_test$is_canceled


# Build stacked data frame for second-level model

stack_train <- data.frame(
  log  = log_train_prob,
  knn  = knn_train_prob,
  c50  = c50_train_prob,
  rf   = rf_train_prob,
  svm  = svm_train_prob,
  is_canceled = y_train
)

stack_test <- data.frame(
  log = log_test_prob,
  knn = knn_test_prob,
  c50 = c50_test_prob,
  rf  = rf_test_prob,
  svm = svm_test_prob
)

length(log_train_prob)
length(knn_train_prob)
length(c50_train_prob)
length(rf_train_prob)
length(svm_train_prob)
length(y_train)