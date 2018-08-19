#model with all the indipendent variables
install.packages("C:\\Users\\Ankesh N. Bhoi\\Downloads\\glmnet_2.0-13.zip",repos=NULL)
library(glmnet)
library(caTools)
library(ggplot2)
#library(plotmo)
#import data
dfr <- read.csv("C:/Users/Ankesh N. Bhoi/Desktop/ML/slump_test.csv")
#exploratory analysis
for (i in 2:8){
  feature=colnames(dfr[i])
  print(ggplot(dfr,aes(dfr[i],dfr$FLOW.cm.))+geom_point()+labs(x=feature,y="flow"))
}
#10 iterations
mse_unregularised<-c()
mse_ridge<-c()
mse_lasso<-c()

rsq_unregularised<-c()
rsq_ridge<-c()
rsq_lasso<-c()

cvfit_unregularised_models<-list()
cvfit_ridge_models<-list()
cvfit_lasso_models<-list()

for(i in 1:10){
  #for reprudecible results in splitting
  set.seed(23+i)
  #Split data to train and test set
  split = sample.split(dfr$FLOW.cm., SplitRatio = 85)
  training_set = subset(dfr, split == TRUE)
  test_set = subset(dfr, split == FALSE)
  #split dependent and indipendent vars
  x_test=as.matrix(test_set[2:8])
  y_test=as.matrix(test_set[10])
  x_train=as.matrix(training_set[2:8])
  y_train=as.matrix(training_set[10])
  #glmnet
  #========================================================================================================================    
  #unregularised
  cvfit_unregularised = cv.glmnet(x=as.matrix(x_train), y=as.matrix(y_train), type.measure = "mse", nfolds = 5)
  cvfit_unregularised_models<-list(cvfit_unregularised_models,cvfit_unregularised)
  #mse on test data
  y_pred=predict.cv.glmnet(cvfit_unregularised, newx = as.matrix(x_test), s = 0)#s=0 use lambda =0 to predict unregularised
  mse_unregularised <- c(mse_unregularised,mean((y_test - y_pred)^2))
  #Model Performance on test data
  performance <- data.frame(y_test ,y_pred)
  print(ggplot(performance,aes(y_test,y_pred))+geom_point()+ geom_abline(intercept = 0,slope=1)+ggtitle(paste("performance of unregularised linear model",toString(i),"on test set","mse = ",mse_unregularised[i])) )
  # Sum of Squares Total and Error
  y_pred=predict.cv.glmnet(cvfit_unregularised, newx = as.matrix(x_train), s = 0)#s=0 use lambda =0 to predict unregularised
  sst <- sum((y_train - mean(y_train))^2)
  sse <- sum((y_pred - y_train)^2)
  # R squared
  rsq_unregularised<- c(rsq_unregularised,1 - sse / sst)
  plot(cvfit_unregularised,main=paste("Unregularised model ",toString(i),"Rsq=",toString(rsq_unregularised[i])))
  #========================================================================================================================    
  #ridge lambda.min is the value of ?? that gives minimum mean cross-validated error
  cvfit_ridge = cv.glmnet(x=as.matrix(x_train), y=as.matrix(y_train), type.measure = "mse", nfolds = 5,alpha=0)
  cvfit_ridge_models<-list(cvfit_ridge_models,cvfit_ridge)
  #Regularisation paths
  plot.glmnet(cvfit_ridge$glmnet.fit)
  #mse on test data
  y_pred=predict.cv.glmnet(cvfit_ridge, newx = as.matrix(x_test), s = cvfit_ridge$lambda.min)
  mse_ridge <- c(mse_ridge,mean((y_test - y_pred)^2))
  #Model Performance on test data
  performance <- data.frame(y_test ,y_pred)
  print(ggplot(performance,aes(y_test,y_pred))+geom_point()+ geom_abline(intercept = 0,slope=1)+ggtitle(paste("performance of Ridge Regression model",toString(i),"on test set","mse = ",mse_ridge[i])) )
  # Sum of Squares Total and Error
  y_pred=predict.cv.glmnet(cvfit_unregularised, newx = as.matrix(x_train), s = cvfit_ridge$lambda.min)
  sst <- sum((y_train - mean(y_train))^2)
  sse <- sum((y_pred - y_train)^2)
  # R squared
  rsq_ridge <- c(rsq_ridge,1 - sse / sst)
  #========================================================================================================================    
  #lasso lambda.min is the value of ?? that gives minimum mean cross-validated error
  cvfit_lasso = cv.glmnet(x=as.matrix(x_train), y=as.matrix(y_train), type.measure = "mse", nfolds = 5,alpha=1)
  cvfit_ridge_models<-list(cvfit_ridge_models,cvfit_ridge)
  #Regularisation paths
  plot.glmnet(cvfit_lasso$glmnet.fit)
  #mse on test data
  y_pred=predict.cv.glmnet(cvfit_lasso, newx = as.matrix(x_test), s = cvfit_lasso$lambda.min)
  mse_lasso <- c(mse_lasso,mean((y_test - y_pred)^2))
  #Model Performance on test data
  performance <- data.frame(y_test ,y_pred)
  print(ggplot(performance,aes(y_test,y_pred))+geom_point()+ geom_abline(intercept = 0,slope=1)+ggtitle(paste("performance of Lasso Regression model",toString(i),"on test set","mse = ",mse_lasso[i])) )
  # Sum of Squares Total and Error
  y_pred=predict.cv.glmnet(cvfit_unregularised, newx = as.matrix(x_train), s = cvfit_lasso$lambda.min)#s=0 use lambda =0 to predict
  sst <- sum((y_train - mean(y_train))^2)
  sse <- sum((y_pred - y_train)^2)
  # R squared
  rsq_lasso <- c(rsq_lasso,1 - sse / sst)
}
#mean model performances
mean.mse_unregularised<-mean(mse_unregularised)
mean.mse_ridge<-mean(mse_ridge)
mean.mse_lasso<-mean(mse_lasso)

mean.rsq_unregularised<-mean(rsq_unregularised)
mean.rsq_ridge<-mean(rsq_ridge)
mean.rsq_lasso<-mean(rsq_lasso)
#standard deviation of model performances
sd.mse_unregularised<-sd(mse_unregularised)
sd.mse_ridge<-sd(mse_ridge)
sd.mse_lasso<-sd(mse_lasso)

sd.rsq_unregularised<-sd(rsq_unregularised)
sd.rsq_ridge<-sd(rsq_ridge)
sd.rsq_lasso<-sd(rsq_lasso)

print("Summary X")
print(paste("mean.mse_unregularised=",mean.mse_unregularised,"mean.mse_ridge=",mean.mse_ridge,"mean.mse_lasso=",mean.mse_lasso))
print(paste("sd.mse_unregularised=",sd.mse_unregularised,"sd.mse_ridge=",sd.mse_ridge,"sd.mse_lasso=",sd.mse_lasso))
print(paste("mean.rsq_unregularised=",mean.rsq_unregularised,"mean.rsq_ridge=",mean.rsq_ridge,"mean.rsq_lasso=",mean.rsq_lasso))
print(paste("sd.rsq_unregularised=",sd.rsq_unregularised,"sd.rsq_ridge=",mean.rsq_ridge,"sd.rsq_lasso=",mean.rsq_lasso))
#========================================================================================================================    
#Extra using X^2 insted of X
#model with all the indipendent variables
install.packages("C:\\Users\\Ankesh N. Bhoi\\Downloads\\glmnet_2.0-13.zip",repos=NULL)
library(glmnet)
library(caTools)
library(ggplot2)
#library(plotmo)
#import data
dfr <- read.csv("C:/Users/Ankesh N. Bhoi/Desktop/ML/slump_test.csv")
#using X^2 inted of X
dfr[2:8]=dfr[2:8]*dfr[2:8]
#exploratory analysis
for (i in 2:8){
  feature=colnames(dfr[i])
  print(ggplot(dfr,aes(dfr[i],dfr$FLOW.cm.))+geom_point()+labs(x=feature,y="flow"))
}
#10 iterations
mse_unregularised<-c()
mse_ridge<-c()
mse_lasso<-c()

rsq_unregularised<-c()
rsq_ridge<-c()
rsq_lasso<-c()

cvfit_unregularised_models<-list()
cvfit_ridge_models<-list()
cvfit_lasso_models<-list()

for(i in 1:10){
  #for reprudecible results in splitting
  set.seed(23+i)
  #Split data to train and test set
  split = sample.split(dfr$FLOW.cm., SplitRatio = 85)
  training_set = subset(dfr, split == TRUE)
  test_set = subset(dfr, split == FALSE)
  #split dependent and indipendent vars
  x_test=as.matrix(test_set[2:8])
  y_test=as.matrix(test_set[10])
  x_train=as.matrix(training_set[2:8])
  y_train=as.matrix(training_set[10])
  #glmnet
  #========================================================================================================================    
  #unregularised
  cvfit_unregularised = cv.glmnet(x=as.matrix(x_train), y=as.matrix(y_train), type.measure = "mse", nfolds = 5)
  cvfit_unregularised_models<-list(cvfit_unregularised_models,cvfit_unregularised)
  #mse on test data
  y_pred=predict.cv.glmnet(cvfit_unregularised, newx = as.matrix(x_test), s = 0)#s=0 use lambda =0 to predict unregularised
  mse_unregularised <- c(mse_unregularised,mean((y_test - y_pred)^2))
  #Model Performance on test data
  performance <- data.frame(y_test ,y_pred)
  print(ggplot(performance,aes(y_test,y_pred))+geom_point()+ geom_abline(intercept = 0,slope=1)+ggtitle(paste("performance of unregularised linear model",toString(i),"on test set","mse = ",mse_unregularised[i])) )
  # Sum of Squares Total and Error
  y_pred=predict.cv.glmnet(cvfit_unregularised, newx = as.matrix(x_train), s = 0)#s=0 use lambda =0 to predict unregularised
  sst <- sum((y_train - mean(y_train))^2)
  sse <- sum((y_pred - y_train)^2)
  # R squared
  rsq_unregularised<- c(rsq_unregularised,1 - sse / sst)
  plot(cvfit_unregularised,main=paste("Unregularised model ",toString(i),"Rsq=",toString(rsq_unregularised[i])))
  #========================================================================================================================    
  #ridge lambda.min is the value of ?? that gives minimum mean cross-validated error
  cvfit_ridge = cv.glmnet(x=as.matrix(x_train), y=as.matrix(y_train), type.measure = "mse", nfolds = 5,alpha=0)
  cvfit_ridge_models<-list(cvfit_ridge_models,cvfit_ridge)
  #Regularisation paths
  plot.glmnet(cvfit_ridge$glmnet.fit)
  #mse on test data
  y_pred=predict.cv.glmnet(cvfit_ridge, newx = as.matrix(x_test), s = cvfit_ridge$lambda.min)
  mse_ridge <- c(mse_ridge,mean((y_test - y_pred)^2))
  #Model Performance on test data
  performance <- data.frame(y_test ,y_pred)
  print(ggplot(performance,aes(y_test,y_pred))+geom_point()+ geom_abline(intercept = 0,slope=1)+ggtitle(paste("performance of Ridge Regression model",toString(i),"on test set","mse = ",mse_ridge[i])) )
  # Sum of Squares Total and Error
  y_pred=predict.cv.glmnet(cvfit_unregularised, newx = as.matrix(x_train), s = cvfit_ridge$lambda.min)
  sst <- sum((y_train - mean(y_train))^2)
  sse <- sum((y_pred - y_train)^2)
  # R squared
  rsq_ridge <- c(rsq_ridge,1 - sse / sst)
  #========================================================================================================================    
  #lasso lambda.min is the value of ?? that gives minimum mean cross-validated error
  cvfit_lasso = cv.glmnet(x=as.matrix(x_train), y=as.matrix(y_train), type.measure = "mse", nfolds = 5,alpha=1)
  cvfit_ridge_models<-list(cvfit_ridge_models,cvfit_ridge)
  #Regularisation paths
  plot.glmnet(cvfit_lasso$glmnet.fit)
  #mse on test data
  y_pred=predict.cv.glmnet(cvfit_lasso, newx = as.matrix(x_test), s = cvfit_lasso$lambda.min)
  mse_lasso <- c(mse_lasso,mean((y_test - y_pred)^2))
  #Model Performance on test data
  performance <- data.frame(y_test ,y_pred)
  print(ggplot(performance,aes(y_test,y_pred))+geom_point()+ geom_abline(intercept = 0,slope=1)+ggtitle(paste("performance of Lasso Regression model",toString(i),"on test set","mse = ",mse_lasso[i])) )
  # Sum of Squares Total and Error
  y_pred=predict.cv.glmnet(cvfit_unregularised, newx = as.matrix(x_train), s = cvfit_lasso$lambda.min)#s=0 use lambda =0 to predict
  sst <- sum((y_train - mean(y_train))^2)
  sse <- sum((y_pred - y_train)^2)
  # R squared
  rsq_lasso <- c(rsq_lasso,1 - sse / sst)
}
#mean model performances
mean.mse_unregularised<-mean(mse_unregularised)
mean.mse_ridge<-mean(mse_ridge)
mean.mse_lasso<-mean(mse_lasso)

mean.rsq_unregularised<-mean(rsq_unregularised)
mean.rsq_ridge<-mean(rsq_ridge)
mean.rsq_lasso<-mean(rsq_lasso)
#standard deviation of model performances
sd.mse_unregularised<-sd(mse_unregularised)
sd.mse_ridge<-sd(mse_ridge)
sd.mse_lasso<-sd(mse_lasso)

sd.rsq_unregularised<-sd(rsq_unregularised)
sd.rsq_ridge<-sd(rsq_ridge)
sd.rsq_lasso<-sd(rsq_lasso)
print("Summary X^2")
print(paste("mean.mse_unregularised=",mean.mse_unregularised,"mean.mse_ridge=",mean.mse_ridge,"mean.mse_lasso=",mean.mse_lasso))
print(paste("sd.mse_unregularised=",sd.mse_unregularised,"sd.mse_ridge=",sd.mse_ridge,"sd.mse_lasso=",sd.mse_lasso))
print(paste("mean.rsq_unregularised=",mean.rsq_unregularised,"mean.rsq_ridge=",mean.rsq_ridge,"mean.rsq_lasso=",mean.rsq_lasso))
print(paste("sd.rsq_unregularised=",sd.rsq_unregularised,"sd.rsq_ridge=",mean.rsq_ridge,"sd.rsq_lasso=",mean.rsq_lasso))
#========================================================================================================================    
#Extra using X^3 insted of X
#model with all the indipendent variables
install.packages("C:\\Users\\Ankesh N. Bhoi\\Downloads\\glmnet_2.0-13.zip",repos=NULL)
library(glmnet)
library(caTools)
library(ggplot2)
#library(plotmo)
#import data
dfr <- read.csv("C:/Users/Ankesh N. Bhoi/Desktop/ML/slump_test.csv")
#using X^2 inted of X
dfr[2:8]=dfr[2:8]*dfr[2:8]*dfr[2:8]
#exploratory analysis
for (i in 2:8){
  feature=colnames(dfr[i])
  print(ggplot(dfr,aes(dfr[i],dfr$FLOW.cm.))+geom_point()+labs(x=feature,y="flow"))
}
#10 iterations
mse_unregularised<-c()
mse_ridge<-c()
mse_lasso<-c()

rsq_unregularised<-c()
rsq_ridge<-c()
rsq_lasso<-c()

cvfit_unregularised_models<-list()
cvfit_ridge_models<-list()
cvfit_lasso_models<-list()

for(i in 1:10){
  #for reprudecible results in splitting
  set.seed(23+i)
  #Split data to train and test set
  split = sample.split(dfr$FLOW.cm., SplitRatio = 85)
  training_set = subset(dfr, split == TRUE)
  test_set = subset(dfr, split == FALSE)
  #split dependent and indipendent vars
  x_test=as.matrix(test_set[2:8])
  y_test=as.matrix(test_set[10])
  x_train=as.matrix(training_set[2:8])
  y_train=as.matrix(training_set[10])
  #glmnet
  #========================================================================================================================    
  #unregularised
  cvfit_unregularised = cv.glmnet(x=as.matrix(x_train), y=as.matrix(y_train), type.measure = "mse", nfolds = 5)
  cvfit_unregularised_models<-list(cvfit_unregularised_models,cvfit_unregularised)
  #mse on test data
  y_pred=predict.cv.glmnet(cvfit_unregularised, newx = as.matrix(x_test), s = 0)#s=0 use lambda =0 to predict unregularised
  mse_unregularised <- c(mse_unregularised,mean((y_test - y_pred)^2))
  #Model Performance on test data
  performance <- data.frame(y_test ,y_pred)
  print(ggplot(performance,aes(y_test,y_pred))+geom_point()+ geom_abline(intercept = 0,slope=1)+ggtitle(paste("performance of unregularised linear model",toString(i),"on test set","mse = ",mse_unregularised[i])) )
  # Sum of Squares Total and Error
  y_pred=predict.cv.glmnet(cvfit_unregularised, newx = as.matrix(x_train), s = 0)#s=0 use lambda =0 to predict unregularised
  sst <- sum((y_train - mean(y_train))^2)
  sse <- sum((y_pred - y_train)^2)
  # R squared
  rsq_unregularised<- c(rsq_unregularised,1 - sse / sst)
  plot(cvfit_unregularised,main=paste("Unregularised model ",toString(i),"Rsq=",toString(rsq_unregularised[i])))
  #========================================================================================================================    
  #ridge lambda.min is the value of ?? that gives minimum mean cross-validated error
  cvfit_ridge = cv.glmnet(x=as.matrix(x_train), y=as.matrix(y_train), type.measure = "mse", nfolds = 5,alpha=0)
  cvfit_ridge_models<-list(cvfit_ridge_models,cvfit_ridge)
  #Regularisation paths
  plot.glmnet(cvfit_ridge$glmnet.fit)
  #mse on test data
  y_pred=predict.cv.glmnet(cvfit_ridge, newx = as.matrix(x_test), s = cvfit_ridge$lambda.min)
  mse_ridge <- c(mse_ridge,mean((y_test - y_pred)^2))
  #Model Performance on test data
  performance <- data.frame(y_test ,y_pred)
  print(ggplot(performance,aes(y_test,y_pred))+geom_point()+ geom_abline(intercept = 0,slope=1)+ggtitle(paste("performance of Ridge Regression model",toString(i),"on test set","mse = ",mse_ridge[i])) )
  # Sum of Squares Total and Error
  y_pred=predict.cv.glmnet(cvfit_unregularised, newx = as.matrix(x_train), s = cvfit_ridge$lambda.min)
  sst <- sum((y_train - mean(y_train))^2)
  sse <- sum((y_pred - y_train)^2)
  # R squared
  rsq_ridge <- c(rsq_ridge,1 - sse / sst)
  #========================================================================================================================    
  #lasso lambda.min is the value of ?? that gives minimum mean cross-validated error
  cvfit_lasso = cv.glmnet(x=as.matrix(x_train), y=as.matrix(y_train), type.measure = "mse", nfolds = 5,alpha=1)
  cvfit_ridge_models<-list(cvfit_ridge_models,cvfit_ridge)
  #Regularisation paths
  plot.glmnet(cvfit_lasso$glmnet.fit)
  #mse on test data
  y_pred=predict.cv.glmnet(cvfit_lasso, newx = as.matrix(x_test), s = cvfit_lasso$lambda.min)
  mse_lasso <- c(mse_lasso,mean((y_test - y_pred)^2))
  #Model Performance on test data
  performance <- data.frame(y_test ,y_pred)
  print(ggplot(performance,aes(y_test,y_pred))+geom_point()+ geom_abline(intercept = 0,slope=1)+ggtitle(paste("performance of Lasso Regression model",toString(i),"on test set","mse = ",mse_lasso[i])) )
  # Sum of Squares Total and Error
  y_pred=predict.cv.glmnet(cvfit_unregularised, newx = as.matrix(x_train), s = cvfit_lasso$lambda.min)#s=0 use lambda =0 to predict
  sst <- sum((y_train - mean(y_train))^2)
  sse <- sum((y_pred - y_train)^2)
  # R squared
  rsq_lasso <- c(rsq_lasso,1 - sse / sst)
}
#mean model performances
mean.mse_unregularised<-mean(mse_unregularised)
mean.mse_ridge<-mean(mse_ridge)
mean.mse_lasso<-mean(mse_lasso)

mean.rsq_unregularised<-mean(rsq_unregularised)
mean.rsq_ridge<-mean(rsq_ridge)
mean.rsq_lasso<-mean(rsq_lasso)
#standard deviation of model performances
sd.mse_unregularised<-sd(mse_unregularised)
sd.mse_ridge<-sd(mse_ridge)
sd.mse_lasso<-sd(mse_lasso)

sd.rsq_unregularised<-sd(rsq_unregularised)
sd.rsq_ridge<-sd(rsq_ridge)
sd.rsq_lasso<-sd(rsq_lasso)
print("Summary X^3")
print(paste("mean.mse_unregularised=",mean.mse_unregularised,"mean.mse_ridge=",mean.mse_ridge,"mean.mse_lasso=",mean.mse_lasso))
print(paste("sd.mse_unregularised=",sd.mse_unregularised,"sd.mse_ridge=",sd.mse_ridge,"sd.mse_lasso=",sd.mse_lasso))
print(paste("mean.rsq_unregularised=",mean.rsq_unregularised,"mean.rsq_ridge=",mean.rsq_ridge,"mean.rsq_lasso=",mean.rsq_lasso))
print(paste("sd.rsq_unregularised=",sd.rsq_unregularised,"sd.rsq_ridge=",mean.rsq_ridge,"sd.rsq_lasso=",mean.rsq_lasso))
#========================================================================================================================    
#Extra using log(X) insted of X
#model with all the indipendent variables
install.packages("C:\\Users\\Ankesh N. Bhoi\\Downloads\\glmnet_2.0-13.zip",repos=NULL)
library(glmnet)
library(caTools)
library(ggplot2)
#library(plotmo)
#import data
dfr <- read.csv("C:/Users/Ankesh N. Bhoi/Desktop/ML/slump_test.csv")
#using X^2 inted of X tackle lof 0 = INF
dfr[2:8]=log(dfr[2:8],base=exp(1))
for (i in 1:103) if(dfr[i,3]==-Inf) dfr[i,3]=0
for (i in 1:103) if(dfr[i,4]==-Inf) dfr[i,4]=0
#exploratory analysis
for (i in 2:8){
  feature=colnames(dfr[i])
  print(ggplot(dfr,aes(dfr[i],dfr$FLOW.cm.))+geom_point()+labs(x=feature,y="flow"))
}
#10 iterations
mse_unregularised<-c()
mse_ridge<-c()
mse_lasso<-c()

rsq_unregularised<-c()
rsq_ridge<-c()
rsq_lasso<-c()

cvfit_unregularised_models<-list()
cvfit_ridge_models<-list()
cvfit_lasso_models<-list()

for(i in 1:10){
  #for reprudecible results in splitting
  set.seed(23+i)
  #Split data to train and test set
  split = sample.split(dfr$FLOW.cm., SplitRatio = 85)
  training_set = subset(dfr, split == TRUE)
  test_set = subset(dfr, split == FALSE)
  #split dependent and indipendent vars
  x_test=as.matrix(test_set[2:8])
  y_test=as.matrix(test_set[10])
  x_train=as.matrix(training_set[2:8])
  y_train=as.matrix(training_set[10])
  #glmnet
  #========================================================================================================================    
  #unregularised
  cvfit_unregularised = cv.glmnet(x=as.matrix(x_train), y=as.matrix(y_train), type.measure = "mse", nfolds = 5)
  cvfit_unregularised_models<-list(cvfit_unregularised_models,cvfit_unregularised)
  #mse on test data
  y_pred=predict.cv.glmnet(cvfit_unregularised, newx = as.matrix(x_test), s = 0)#s=0 use lambda =0 to predict unregularised
  mse_unregularised <- c(mse_unregularised,mean((y_test - y_pred)^2))
  #Model Performance on test data
  performance <- data.frame(y_test ,y_pred)
  print(ggplot(performance,aes(y_test,y_pred))+geom_point()+ geom_abline(intercept = 0,slope=1)+ggtitle(paste("performance of unregularised linear model",toString(i),"on test set","mse = ",mse_unregularised[i])) )
  # Sum of Squares Total and Error
  y_pred=predict.cv.glmnet(cvfit_unregularised, newx = as.matrix(x_train), s = 0)#s=0 use lambda =0 to predict unregularised
  sst <- sum((y_train - mean(y_train))^2)
  sse <- sum((y_pred - y_train)^2)
  # R squared
  rsq_unregularised<- c(rsq_unregularised,1 - sse / sst)
  plot(cvfit_unregularised,main=paste("Unregularised model ",toString(i),"Rsq=",toString(rsq_unregularised[i])))
  #========================================================================================================================    
  #ridge lambda.min is the value of ?? that gives minimum mean cross-validated error
  cvfit_ridge = cv.glmnet(x=as.matrix(x_train), y=as.matrix(y_train), type.measure = "mse", nfolds = 5,alpha=0)
  cvfit_ridge_models<-list(cvfit_ridge_models,cvfit_ridge)
  #Regularisation paths
  plot.glmnet(cvfit_ridge$glmnet.fit)
  #mse on test data
  y_pred=predict.cv.glmnet(cvfit_ridge, newx = as.matrix(x_test), s = cvfit_ridge$lambda.min)
  mse_ridge <- c(mse_ridge,mean((y_test - y_pred)^2))
  #Model Performance on test data
  performance <- data.frame(y_test ,y_pred)
  print(ggplot(performance,aes(y_test,y_pred))+geom_point()+ geom_abline(intercept = 0,slope=1)+ggtitle(paste("performance of Ridge Regression model",toString(i),"on test set","mse = ",mse_ridge[i])) )
  # Sum of Squares Total and Error
  y_pred=predict.cv.glmnet(cvfit_unregularised, newx = as.matrix(x_train), s = cvfit_ridge$lambda.min)
  sst <- sum((y_train - mean(y_train))^2)
  sse <- sum((y_pred - y_train)^2)
  # R squared
  rsq_ridge <- c(rsq_ridge,1 - sse / sst)
  #========================================================================================================================    
  #lasso lambda.min is the value of ?? that gives minimum mean cross-validated error
  cvfit_lasso = cv.glmnet(x=as.matrix(x_train), y=as.matrix(y_train), type.measure = "mse", nfolds = 5,alpha=1)
  cvfit_ridge_models<-list(cvfit_ridge_models,cvfit_ridge)
  #Regularisation paths
  plot.glmnet(cvfit_lasso$glmnet.fit)
  #mse on test data
  y_pred=predict.cv.glmnet(cvfit_lasso, newx = as.matrix(x_test), s = cvfit_lasso$lambda.min)
  mse_lasso <- c(mse_lasso,mean((y_test - y_pred)^2))
  #Model Performance on test data
  performance <- data.frame(y_test ,y_pred)
  print(ggplot(performance,aes(y_test,y_pred))+geom_point()+ geom_abline(intercept = 0,slope=1)+ggtitle(paste("performance of Lasso Regression model",toString(i),"on test set","mse = ",mse_lasso[i])) )
  # Sum of Squares Total and Error
  y_pred=predict.cv.glmnet(cvfit_unregularised, newx = as.matrix(x_train), s = cvfit_lasso$lambda.min)#s=0 use lambda =0 to predict
  sst <- sum((y_train - mean(y_train))^2)
  sse <- sum((y_pred - y_train)^2)
  # R squared
  rsq_lasso <- c(rsq_lasso,1 - sse / sst)
}
#mean model performances
mean.mse_unregularised<-mean(mse_unregularised)
mean.mse_ridge<-mean(mse_ridge)
mean.mse_lasso<-mean(mse_lasso)

mean.rsq_unregularised<-mean(rsq_unregularised)
mean.rsq_ridge<-mean(rsq_ridge)
mean.rsq_lasso<-mean(rsq_lasso)
#standard deviation of model performances
sd.mse_unregularised<-sd(mse_unregularised)
sd.mse_ridge<-sd(mse_ridge)
sd.mse_lasso<-sd(mse_lasso)

sd.rsq_unregularised<-sd(rsq_unregularised)
sd.rsq_ridge<-sd(rsq_ridge)
sd.rsq_lasso<-sd(rsq_lasso)

print("Summary log X")
print(paste("mean.mse_unregularised=",mean.mse_unregularised,"mean.mse_ridge=",mean.mse_ridge,"mean.mse_lasso=",mean.mse_lasso))
print(paste("sd.mse_unregularised=",sd.mse_unregularised,"sd.mse_ridge=",sd.mse_ridge,"sd.mse_lasso=",sd.mse_lasso))
print(paste("mean.rsq_unregularised=",mean.rsq_unregularised,"mean.rsq_ridge=",mean.rsq_ridge,"mean.rsq_lasso=",mean.rsq_lasso))
print(paste("sd.rsq_unregularised=",sd.rsq_unregularised,"sd.rsq_ridge=",mean.rsq_ridge,"sd.rsq_lasso=",mean.rsq_lasso))

