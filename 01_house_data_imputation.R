# HOuse pricing data from Kaggle
#Jenny Yu

library(caret)
library(Metrics)
library(ggplot2)
setwd("YOUR PATH")
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
#discard these columns that had majority of Null
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

#using VIF, value inflation factors, to check for multicollinearity
library(usdm)
v1<-vifcor(num_data, th=0.6)

#drop the following  column variables
trainData2=subset(trainData, select=-c(GarageCars,YearBuilt, X1stFlrSF, GrLivArea, TotRmsAbvGrd, BsmtFinSF1, GarageYrBlt, X2ndFlrSF))
trainData2$Id<-NULL

#test set
testData2=subset(testData, select=-c(GarageCars,YearBuilt, X1stFlrSF, GrLivArea, TotRmsAbvGrd, BsmtFinSF1, GarageYrBlt, X2ndFlrSF))
testData2$Id<-NULL

#King and Bonoit, this snippet is used to harmonize levels:

library(gdata)
for(attr in colnames(trainData2)) {
  if(is.factor(trainData2[[attr]])) { 
    map <- mapLevels(x=list(trainData2[,attr], testData2[,attr]), codes=FALSE, combine=TRUE)
    mapLevels(trainData2[,attr]) <- map
    mapLevels(testData2[,attr]) <- map
  }
}


#take a look at skewness
library(moments)
skewness(trainData2$SalePrice)
skewness(log(trainData2$SalePrice))

hist(trainData2$SalePrice)
hist(log(trainData2$SalePrice))

#log transform the target later
#combine training and test data for imputation, append test dataset to training
dim(trainData2)

train_variables<-subset(trainData2, select=-c(SalePrice))
total_data<-rbind(train_variables, testData2)


#now impute with missForest
library(missForest)
total_data_imp<-missForest(total_data, maxiter=10, ntree=100)
total_data_imputed<-total_data_imp$ximp
write.csv(total_data_imputed, file="concatenated_total_data_imputed.csv", row.names=FALSE)

#save
total_data_imputed<-read.csv("concatenated_total_data_imputed.csv")
