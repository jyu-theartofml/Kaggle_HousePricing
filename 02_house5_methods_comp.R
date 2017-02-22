# HOuse pricing data from Kaggle
#Jenny Yu

library(caret)
library(Metrics)
library(ggplot2)
library(moments)
library(e1071)
library(dummy)

#setwd("YOUR PATH")
trainData=read.csv("train.csv", stringsAsFactors=TRUE)
testData=read.csv("test.csv", stringsAsFactors=TRUE)
feature_importance=read.csv("GBM_FeatureImportance.csv")
dim(trainData)
trainData$log_price<-log(trainData$SalePrice)

#load the pre-imputated dataset
total_data_imputed<-read.csv("concatenated_total_data_imputed.csv")
total_data_imputed$MSSubClass<-as.factor(total_data_imputed$MSSubClass)

#select top 50 important features based on previous GBM model training
select_feat<-as.character(feature_importance$var[1:50])
total_trans<-subset(total_data_imputed, select=select_feat)
head(total_trans)
dim(total_trans)



#apply mean centered transformation to the appropriate columns based on feature importance from previous GBM model
preProcValues <- preProcess(total_trans, method = c("center", "scale"))
train_trans<-predict(preProcValues, total_trans)



##########prepare dataset without dummy vectors###########
k=dim(trainData)[1]
training=train_trans[1:k,] #should be 1460
g=dim(train_trans)[1]
k=k+1
test_set=train_trans[k:g,] # should be 1459

training_price=cbind(training, trainData$log_price)
colnames(training_price)[colnames(training_price)=="trainData$log_price"] <- "log_price"
#validation set
set.seed(124)
ind <- sample(2, nrow(training_price), replace=T, prob=c(0.8,0.20))
training_set<-training_price[ind==1,]
validate_set <-training_price[ind==2,]
#NOTE: for many methods in R, it takes categorical features without the need to 
#do one-hot-encoding. 
#RandomForest in R has a limit of 32 features/factor levels it can accept as 
#categorical values without encoding.


#############try GBM gridsearch###############################
#use formula in caret train package later, it would automaticall create dummy variables for 
#categorical factors. Use non-formula interface to compare to GBM package later

#gridsearch in caret should be performed simultaneously for all parameters at low shrinkage.

gbmGrid1 <-  expand.grid(interaction.depth= c(10,14,16), 
                         shrinkage = 0.0005,
                         n.trees=1000,
                         n.minobsinnode=c(2,4,6,8)) 
                         
set.seed(21)                 
fitControl <- trainControl(method = "cv",number = 5)
#use non-formula syntax gave slightly better rmse
                        
gbmFit1<-train(training_set[,-51], training_set$log_price,
               method = "gbm", metric='RMSE',
               trControl = fitControl, 
               verbose = FALSE,
               tuneGrid = gbmGrid1)
                         
ggplot(gbmFit1)
gbmFit1$results
prediction_val_C<-predict(gbmFit1, validate_set) 
rmse(validate_set$log_price, prediction_val_C)

#from previous gridsearch, optima depth is 14, minobs is 8
#Now lower the shrinkage (learning rate)
gbmGrid2 <-  expand.grid(interaction.depth= 14, 
                         shrinkage = 0.0005,
                         n.trees=c(26000,28000),
                         n.minobsinnode=8) ## or try 10 from past runs

set.seed(21)                 
fitControl <- trainControl(method = "cv",number = 5)

gbmFit2<-train(training_set[,-51], training_set$log_price, 
               method = "gbm", metric='RMSE',
               trControl = fitControl, 
               verbose = FALSE,
               tuneGrid = gbmGrid2)

ggplot(gbmFit2)
gbmFit2$results

prediction_val_gbm<-predict(gbmFit2, validate_set[,-51]) 
rmse(validate_set$log_price, prediction_val_gbm)

##optimized for depth 14, minobs 8, lr=0.0005, ntrees=30000
#LB=0.13533,, val_loss=0.1419481


################## Feature Selection using Univariate Filters #################
#Feature Selection using Univariate statistical methods
library(gam)
library(randomForest)
filterCtrl <- sbfControl(functions = rfSBF, method = "cv", number = 5)
set.seed(10)
rfWithFilter <- sbf(training_set[,-51], training_set$log_price, sbfControl = filterCtrl)
#fit to training data
model_sbf<-rfSBF$fit(training_set[,-51], training_set$log_price)
#obtain RMSE on validation set
prediction_val_sbf<-predict(model_sbf,validate_set[,-51]) 
rmse(validate_set$log_price, prediction_val_sbf)

#val_loss=0.1621139

################# Random Forest by Randomization (extratrees) ############################
library(randomForest)
library(extraTrees)
rfrGrid1 <-  expand.grid(mtry= c(17,20), 
                         numRandomCuts=c(4,6))

set.seed(21)                 
fitControl <- trainControl(method = "cv",number = 5)
#use non-formula syntax gave slightly better rmse
#for RF, it requires formula to convert to dummy variables, number of categorical variables in a feature exceeded the limit (32)
rfrFit1<-train(log_price~., data=training_set,
               method = "extraTrees", metric='RMSE',
               trControl = fitControl,
               tuneGrid = rfrGrid1)
##val_loss=0.1565611

#second gridsearch, this method is computationally expensive

rfrGrid2 <-  expand.grid(mtry= c(17,18), 
                         numRandomCuts=c(9,12,14))

set.seed(21)                 
fitControl <- trainControl(method = "cv",number = 5)
#use non-formula syntax gave slightly better rmse
#for RF, it requires formula to convert to dummy variables, number of categorical variables in a feature exceeded the limit (32)
rfrFit2<-train(log_price~., data=training_set,
               method = "extraTrees", metric='RMSE',
               trControl = fitControl,
               tuneGrid = rfrGrid2)

ggplot(rfrFit2)

prediction_val_rfr<-predict(rfrFit2, validate_set) 
rmse(validate_set$log_price, prediction_val_rfr)
##val_loss=0.154461

###############make final prediction on test set
prediction_test<-predict(gbmFit2, test_set)
prediction_price<-exp(prediction_test)

submission<-read.csv("sample_submission.csv", header=TRUE)
submission$SalePrice<-prediction_price
write.csv(submission, file="submission(gbm_mc11).csv", row.names=FALSE)
