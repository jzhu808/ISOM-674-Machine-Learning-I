train <- fread('Train10M_20levels.csv')

train[] <- lapply(train, as.factor)

train <- as.data.frame(train)
test_indexes <- sample(nrow(train),2500000)

#Make new training data from this sample

test <- train[test_indexes,]

test_click1 <- nrow(test[test$click == 1,])
test_click0 <- nrow(test[test$click == 0,])
test_ratio <- test_click0/test_click1

#R
remaining <- train[-test_indexes,]

val_indexes <- sample(nrow(remaining),2500000)
validation <- remaining[val_indexes,]
train <- remaining[-val_indexes,]

val_ratio <- nrow(validation[validation$click == 0,]) / nrow(validation[validation$click == 1,])
train_ratio <- nrow(train[train$click == 0,]) / nrow(train[train$click == 1,])

rm(remaining)
###
Xtrain <- train[,-1]
Ytrain <- as.numeric(levels(train$click))[train$click]
Xval <- validation[,-1]
Yval <- as.numeric(levels(validation$click))[validation$click]

xgbtrain <- sparse.model.matrix(data=train,click~.-1)
xgbval <- sparse.model.matrix(data=validation,click~.-1)

xgb_train <- xgb.DMatrix(data=xgbtrain,label=Ytrain)
xgb_val <- xgb.DMatrix(data=xgbval,label=Yval)

params = list(
  booster="gbtree",
  eta=0.1,
  max_depth=17,
  min_child_weight=5,
  colsample_bytree=0.8,
  gamma=3,
  subsample=0.75,
  objective="binary:logistic",
  eval_metric="logloss"
)


xgb <- xgb.train(
  params=params,
  data=xgb_train,
  nrounds=3000,
  nthreads=0,
  early_stopping_rounds=10,
  watchlist=list(val1=xgb_train,val2=xgb_val),
  verbose=1
)

Xtest <- test[,-1]
Ytest <- as.numeric(levels(test$click))[test$click]

xgbtest <- sparse.model.matrix(data=test,click~.-1)
xgb_test <- xgb.DMatrix(data=xgbtest,label=Ytest)

YHat <- predict(xgb,xgb_test)

LL <- function(Pred,YVal){
  ll <- -mean(YVal*log(Pred)+(1-YVal)*log(1-Pred))
  return(ll)
}  
LL(YHat,Ytest)

trainRF <- sample_n(train,size=500000)
FM <- click~.
RF <- randomForest(FM,data=trainRF,ntree=50)
RF
PHat <- predict(RF,newdata=validation,type="prob")
PHat <- PHat[,2]

MLmetrics::LogLoss(PHat,Yval)
LL(PHat,Yval)
