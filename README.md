# HousePricing dataset - methods comparison
<p> The package <i> Caret </i> is one of the most popular for tackling machine learning problems in R. Using this <a href ="https://www.kaggle.com/c/house-prices-advanced-regression-techniques"> Kaggle competition </a> data set, several of the methods in <i> Caret </i> are compared in terms of RMSE to see how they perform.</p>
<p>First, the data was cleaned up, categorical variable values were harmoninized between training and test sets. Muliticollinearity evaluation was done to identify co-dependent predictors (ones that are redundant and doesn't provide useful info towards model training).Then data imputation was performed with MissForest package in R, which's a powerful(and time consuming) algorithm that creates a bagged decision tree to impute a missing value based on other predictor variables. In <i> Caret </i>, the built in imputation method uses the K-nearest neighbor algorithm, which is less computationally expensive and more practical for larger dataset. The imputed data (training and test data) was then saved for later usage.</p>
<p> In the second .R file, three regression algorithms are compared. Before training the models, the training and test data were mean-centered and scaled (for numerical columns).The target, Sales Price, was transformed by taking the log to remedy the skewness of the distribution,and the training data was split into 80% for training, and 20% for validation. A 5 fold cv was performed for each method.</p>
**Gradient Boosted Method (GBM) gridsearch**
<p>This method allowed grid search on depth of the decision tree, number of estimators (n.trees) and minimum number of observation in a terminal node (not weight). A validation loss (RMSE) of 0.1419481 was obtained (only 275 samples), LB =0.13510. Only prediction using GBM was submitted to Kaggle. </p>

|![lr_overfit](https://github.com/yinniyu/Kaggle_HousePricing/blob/master/lr_overfit.jpeg)|![lr_overfit](https://github.com/yinniyu/Kaggle_HousePricing/blob/master/Lr001.jpeg)|
|:---:|:---:|
|Overfit| Reduced overfit at smaller shrinkage|

**Univariate Feature Selection**
<p>Predictors that have statistically significant differences between the classes are then used for modeling. validation loss was val_loss=0.1621139, the worst among the three methods.</p>

**Regularized Random Forest gridsearch**
<p>This algorithm is the most time consuming of the three. The parameters availabe for grid search is number of random predictor variables selected at each split (mtry), and number of random threshold for each predictor variable (numRandomCuts). Validation loss was 0.1553799.

<p>Genetic Algorithm in the <i> Caret </i> package was tried, but it kept on freezing R studio using up too memory. Same with the Regularized Random Forest algorithm. Parallelization of the computation should be explored in R to resolve these issues. In addition, algorithms like GBM and Xgboost don't yield the same results when evaluated against the corresponding models in <i> Caret </i>.
