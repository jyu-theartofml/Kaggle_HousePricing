# HOuse pricing data from Kaggle
#Jenny Yu

library(caret)
library(Metrics)
setwd("C:/Users/Jenny/Machine Learning (Kaggle)/HousePricing")
trainData=read.csv("train.csv", stringsAsFactors=TRUE)
testData=read.csv("test.csv", header=TRUE)
dim(trainData)
## to see column names
colnames(trainData)
trainData$MSSubClass<-as.factor(trainData$MSSubClass)
testData$MSSubClass<-as.factor(testData$MSSubClass)
# to show data types of the columns

sapply(trainData, class)

#find number of n/a, null values in each columns
num_na<-apply(trainData, 2, function(x) length(which(is.na(x))))
trainData$Alley<-NULL
trainData$FireplaceQu<-NULL
trainData$PoolQC<-NULL
trainData$Fence<-NULL
trainData$MiscFeature<-NULL

trainData$GarageCond<-NULL
trainData$LotFrontage<-NULL

#test data
testData$Alley<-NULL
testData$FireplaceQu<-NULL
testData$PoolQC<-NULL
testData$Fence<-NULL
testData$MiscFeature<-NULL

testData$GarageCond<-NULL
testData$LotFrontage<-NULL

  
  

dim_train<-dim(trainData)
dim_test<-dim(testData)

#select only numeric column to look at correlation
train_numeric<-sapply(trainData, is.numeric)
num_data<-trainData[, train_numeric]
dim(num_data)

#using VIF, value inflation factors
library(usdm)
v1<-vifcor(num_data, th=0.6)


#drop the following  column variables

trainData2=subset(trainData, select=-c(GarageCars,YearBuilt, X1stFlrSF, GrLivArea, TotRmsAbvGrd, BsmtFinSF1, GarageYrBlt, X2ndFlrSF))
trainData2$Id<-NULL

#test set
testData2=subset(testData, select=-c(GarageCars,YearBuilt, X1stFlrSF, GrLivArea, TotRmsAbvGrd, BsmtFinSF1, GarageYrBlt, X2ndFlrSF))
test_id=testData$Id
testData2$Id<-NULL
boxplot(trainData2$SalePrice, las=2)
boxplot(SalePrice~SaleCondition, data=trainData2)


#King and Bonoit, this snippet can be useful to harmonize levels:
  
#harmonize the factor levels for test and training set
library(gdata)
for(attr in colnames(trainData2)) {
  if(is.factor(trainData2[[attr]])) { 
    map <- mapLevels(x=list(trainData2[,attr], testData2[,attr]), codes=FALSE, combine=TRUE)
    mapLevels(trainData2[,attr]) <- map
    
    mapLevels(testData2[,attr]) <- map
    
  }

  }

   






#try RF, unlike GBM, RF cannot handle missing, NA values
library(randomForest)
#need to impute missing values first for RF, then sample for test and validation set
rf_data<-rfImpute(SalePrice~., data=trainData2)
ind <- sample(2, nrow(rf_data), replace=T, prob=c(0.8,0.20))
training_set<-rf_data[ind==1,]
validate_set <-rf_data[ind==2,]
write.csv(training_set, file="training_set.csv", row.names=FALSE)
write.csv(validate_set, file="validate_set.csv", row.names=FALSE)

#also impute test data set! missForest works on both numeric and categorical
library(missForest)
testData2_imp<-missForest(testData2, maxiter=10, ntree=100)
testData2_imputed<-testData2_imp$ximp
write.csv(testData2_imputed, file="test_imputed.csv", row.names=FALSE)

#build rf model



#tune RF

control <- trainControl(method="repeatedcv", number=5, repeats=3, search="grid")
set.seed(123)
tunegrid <- expand.grid(.mtry=c(20:25))
rf_gridsearch <- train(SalePrice~., data=training_set, method="rf", metric='mae', tuneGrid=tunegrid, trControl=control)

plot(rf_gridsearch)
#24 mtry was best
prediction_rf<-rf_gridsearch$predict(rf, rf_validate)





set.seed(123)
model_rf<-randomForest(SalePrice~., data=training_set, importance =TRUE, replace=TRUE,mtry=24, ntrees=3000) 
prediction_rf2<-predict(model_rf, validate_set)
rmse(log(validate_set$SalePrice), log(prediction_rf2))
importance(model_rf, type=1)

#making prediction on actual test data
predict_test<-predict(model_rf, testData2_imputed)

submission<-read.csv("sample_submission.csv", header=TRUE)
submission$SalePrice<-predict_test
write.csv(submission, file="submission(RF2).csv", row.names=FALSE)

#after looking the feature importance from RF model above, drop these columns
training_set$Utilities<-NULL
training_set$SaleType<-NULL
training_set$ScreenPorch<-NULL
training_set$Condition1<-NULL
validate_set$Utilities<-NULL
validate_set$SaleType<-NULL
validate_set$ScreenPorch<-NULL
validate_set$Condition1<-NULL
testData2_imputed$Utilities<-NULL
testData2_imputed$SaleType<-NULL
testData2_imputed$ScreenPorch<-NULL
testData2_imputed$Condition1<-NULL

library(gbm)

#NOTE: for many methods in R, it takes categorical features without the need to 
#do one-hot-encoding. 
#RandomForest in R has a limit of 32 features/factor levels it can accept as 
#categorical values without encoding.
#You can define a categorical column (i.e,ordinal data) using as.factor()

#shrinkage=learning rate(best value=0.09), bag.fraction is random subsampling
#5-fold CV,
gbmGrid <-  expand.grid(interaction.depth = c(4,6,8), 
                        shrinkage = c(0.03,0.05),
                        n.trees=500,
                        n.minobsinnode = c(2,3))

set.seed(21)
fitControl <- trainControl(method = "repeatedcv",
  number = 5,
  repeats = 3)

gbmFit<- train(SalePrice~ ., data = training_set, 
                 method = "gbm", metric='RMSE',
                 trControl = fitControl, 
                 verbose = FALSE,
                 tuneGrid = gbmGrid)


plot(gbmFit)


#use interatin depth 4, n.tree 500, 5 minobsinnoide
set.seed(124)
model_gbm<-gbm(SalePrice~., data=training_set,distribution="gaussian", n.trees=500, shrinkage=0.03, bag.fraction=0.8,interaction.depth=21, n.minobsinnode=10,cv.folds=5, keep.data=TRUE, verbose=FALSE, n.cores=1)


prediction<-predict(model_gbm, validate_set)
#calculate RMSE of the log values of prediction and actual values
rmse(log(validate_set$SalePrice), log(prediction))

predict_test<-predict(model_gbm, testData2_imputed)

submission<-read.csv("sample_submission.csv", header=TRUE)
submission$SalePrice<-predict_test
write.csv(submission, file="submission(gbm3).csv", row.names=FALSE)

write.csv(summary(model_gbm), file="GBM_FeatureImportance.csv")
read.csv("GBM_FeatureImportance.csv")
