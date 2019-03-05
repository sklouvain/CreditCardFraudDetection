#Thesis (MS in Business Engineering)
#Norwegian School of Economics / UCLouvain
#Project: Credit Card Fraud Detection
#Author: Sergei Kurin
#Supervisor: Professor Stein Ivar Steinshamn (Norwegian School of Economics)
#Supervisor: Professor Marco Saerens (UCLouvain)

#Preprocessing: Oversampling
#Classification Models: Logistic Regression + Adjustment for Priors 

#################Packages###############################


library(ISLR)
library(ggplot2)
library(dplyr)
library(ggridges)
library(leaps)
library(gam)
library(fmsb)
library(class)
library(ROSE)
library(rpart)
library(DMwR)
library(pROC)
library(caret)
library(e1071)
library(rafalib) 
library(ROSE)
library(rgl)
library(MASS)
library(PerformanceAnalytics)
library(nnet)
library(randomForest)

#################READ DATASET###########################


ptm <- proc.time()
#hyper <-read.csv('http://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/hypothyroid.data', header=F)
Auto <- read.csv("creditcard.csv/creditcard.csv", sep=",",dec = ".",header=TRUE)
proc.time() - ptm

names(Auto)
dim(Auto)

Auto %>%
  gather(variable, value, -Class) %>%   ###Prob distributions of variables for both classes 
  ggplot(aes(y = as.factor(variable), 
             fill = as.factor(Class), 
             x = percent_rank(value))) +
  geom_density_ridges()

Auto <- Auto[,2:31]   ###1st variable (Time) is excluded since prob distributions are the same for both classes
boxplot(Auto[,1:29])

#################HIST FOR ALL FEATURES##################


for (i in 1:29){
  par(mfrow=c(2, 2))
  hist(Auto[, i], col="gray", prob=TRUE, xlim=c(min(Auto[, i]), max(Auto[, i])), main="")
  curve(dnorm(x, mean=mean(Auto[, i]), sd=sd(Auto[, i])), add=TRUE, lwd=2, col="red")
  boxplot(Auto[, i])
  points(mean(Auto[, i]), col="orange", pch=18)
  qqnorm(Auto[, i])
  qqline(Auto[, i])
}

#################SHUFFLE DATA###########################


Auto <- Auto[sample(nrow(Auto)),]

##################FOLDS FOR 3-FOLD CROSS VALIDATION####


set.seed(1234)
idx <- createFolds(Auto$Class, k=3)   ###k=3 -> 3-fold Cross Validation
sapply(idx, length)

sapply(idx, function(i) table(Auto$Class[i])) #positive/negative cases

prop.table(table(Auto$Class)) #proportion of positive/negative cases 
table(Auto$Class)

Auto$Class <- as.factor(Auto$Class)

####################LOGISTIC REGRESSON################# 
####################NO SAMPLING METHODS################


ptm <-proc.time()
res.lr <- sapply(seq_along(idx), function(i) {
  
  ###Loop over each of the 3 Cross-Validation folds
  
  set.seed(1+i)
  
  SLR <- multinom(formula = Class ~ ., family=binomial, data = Auto[-idx[[i]], ])  ###Fit Logistic Regression model
  print(SLR)
  
  pred <- predict(SLR, Auto[idx[[i]],-30], type="class")   ###Class prediction
  
  au <- roc.curve(Auto$Class[idx[[i]]], pred)  ###ROC curve <- accuracy measure
  CCoutLR <- capture.output(au$auc)   ###Area Under Curve (AUC)
  cat(CCoutLR, file="CC_outputLR_AUC.txt", sep=",", fill = TRUE, append=TRUE)  ###Write AUC in txt file
  
  CM <- table(Auto$Class[idx[[i]]], pred)   ###Confusion matrix
  accuracy <- sum(diag(CM))/sum(CM)         ###Accuracy based on Confusion Matrix
  CCout1LR <- capture.output(accuracy)   
  cat(CCout1LR, file="CC_outputLR_Acc_CM.txt", sep=",", fill = TRUE, append=TRUE)  ###Write Confusion matrix accuracy in txt file
  
  CCout2LR <- capture.output(CM)             
  cat(CCout2LR, file="CC_outputLR_CM.txt", sep=",", fill = TRUE, append=TRUE)  ###Write Confusion matrix in txt file
  
  print(au$auc)
  
})
mean(res.lr)
proc.time() - ptm

####################LOGISTIC REGRESSON###############
####################OVERSAMPLING#####################


set.seed(1)
ptm <- proc.time()
res.lr <- sapply(seq_along(idx), function(i){
  
  ###Loop over each of the 3 Cross-Validation folds
  
  set.seed(1+i)
  
  data_balanced_over <- ovun.sample(Class ~ ., data = Auto[-idx[[i]], ], method = "over", N = 379086, seed = 1)$data
  CCtable <- capture.output(table(data_balanced_over$Class))
  cat(CCtable, file="CCtable.txt", sep=",", fill = TRUE, append=TRUE)
  
  SLR <- multinom(formula = Class ~ ., family=binomial, data = data_balanced_over)  ###Fit Logistic Regression model
  print(SLR)
  
  pred <- predict(SLR, Auto[idx[[i]],-30], type="class")  ###Class prediction
  
  au <- roc.curve(Auto$Class[idx[[i]]], pred)   ###ROC curve
  CCoutLRS <- capture.output(au$auc)   ###Area Under Curve (AUC)
  cat(CCoutLRS, file="CC_outputLR_OS_AUC.txt", sep=",", fill = TRUE, append=TRUE)   ###Write AUC in txt file
  
  CM <- table(Auto$Class[idx[[i]]], pred)   ###Confusion matrix
  accuracy <- sum(diag(CM))/sum(CM)       ###Accuracy based on Confusion Matrix
  CCout1LRS <- capture.output(accuracy)
  cat(CCout1LRS, file="CC_outputLR_OS_Acc_CM.txt", sep=",", fill = TRUE, append=TRUE)   ###Write Confusion matrix accuracy in txt file
  
  CCout2LRS <- capture.output(CM)
  cat(CCout2LRS, file="CC_outputLR_OS_CM.txt", sep=",", fill = TRUE, append=TRUE)  ###Write Confusion matrix in txt file
  
  print(au$auc)
  
})
mean(res.lr)
proc.time() - ptm


##################LOGISTIC REGRESSION#################
##################PRIORS ARE NOT KNOWN################
##################NO SAMPLING METHODS#################

ptm <- proc.time()
res.lr <- sapply(seq_along(idx), function(i){
  
  ###Loop over each of the 3 Cross-Validation folds
  
  set.seed(1+i)
  
  SLR <- multinom(formula = Class ~ ., family=binomial, data = Auto[-idx[[i]], ])
  print(SLR)
  
  pred <- predict(SLR, Auto[idx[[i]],-30], type="prob")   ###Prob prediction
  
  
  ###Adjustment for priors
  pt <- prop.table(table(Auto$Class[-idx[[i]]]))
  pt_wneg <- pt[1]  
  pt_wpos <- pt[2]
  
  pt_wneg
  pt_wpos
  
  p_wneg <- rep(0, 20)
  p_wpos <- rep(0, 20)
  
  p_wneg[1] <- pt_wneg
  p_wpos[1] <- pt_wpos
  
  s <- 1
  
  while ((abs(p_wneg[s+1] - p_wneg[s]) + abs(p_wpos[s+1] - p_wpos[s])) > 0.0001){   ###Convergence test
    
    preds <- ((p_wpos[s]/pt_wpos)*pred)/((p_wpos[s]/pt_wpos)*pred + (p_wneg[s]/pt_wneg)*(1-pred))
    preds
    
    p_wneg[s+1] <- (1/length(Auto$Class[idx[[i]]]))*sum((1-preds))
    p_wpos[s+1] <- (1/length(Auto$Class[idx[[i]]]))*sum(preds)
    
    s <- s + 1
  }
  
  outLREM <- capture.output(p_wneg)
  out1LREM <- capture.output(p_wpos)
  cat(outLREM, file="outputLREM.txt", sep=",", fill = TRUE, append=TRUE)
  cat(out1LREM, file="output1LREM.txt", sep=",", fill = TRUE, append=TRUE)
  
  au <- roc.curve(Auto$Class[idx[[i]]], preds) ###ROC curve  
  out2LREM <- capture.output(au$auc)  ###AUC
  cat(out2LREM, file="output2LREM.txt", sep=",", fill = TRUE, append=TRUE)
  
  CM <- table(Auto$Class[idx[[i]]], preds > 0.5)  ###Confusion matrix
  accuracy <- sum(diag(CM))/sum(CM)
  out3LREM <- capture.output(accuracy)
  cat(out3LREM, file="output3LREM.txt", sep=",", fill = TRUE, append=TRUE)
  out4LREM <- capture.output(CM)
  cat(out4LREM, file="output4LREM.txt", sep=",", fill = TRUE, append=TRUE)
  
  print(au$auc)
  
})
mean(res.lr)
proc.time() - ptm


##################LOGISTIC REGRESSION#################
##################PRIORS ARE NOT KNOWN################
##################OVERSAMPLING#################

set.seed(1)
ptm <- proc.time()
res.lr <- sapply(seq_along(idx), function(i){
  ##loop over each of the 3 cross-validation folds
  
  set.seed(1+i)
  
  data_balanced_over <- ovun.sample(Class ~ ., data = Auto[-idx[[i]], ], method = "over", N=379086, seed = 1)$data
  table(data_balanced_over$Class)
  outTable <- capture.output(table(data_balanced_over$Class))
  cat(outTable, file="outputTable.txt", sep=",", fill = TRUE, append=TRUE)
  
  
  #data_balanced_over <- data_balanced_over[sample(nrow(data_balanced_over)),]
  #data_balanced_over
  #print(dim(data_balanced_over))
  
  SLR <- multinom(formula = Class ~ ., family=binomial, data = data_balanced_over)
  
  pred <- predict(SLR, Auto[idx[[i]],-30], type="prob")
  
  ###Adjustment for priors  
  pt <- prop.table(table(data_balanced_over$Class))
  pt_wneg <- pt[1]
  pt_wpos <- pt[2]
  
  pt_wneg
  pt_wpos

  p_wneg <- rep(0, 20)
  p_wpos <- rep(0, 20)
  
  p_wneg[1] <- pt_wneg
  p_wpos[1] <- pt_wpos
  
  s <- 1
  
  while ((abs(p_wneg[s+1] - p_wneg[s]) + abs(p_wpos[s+1] - p_wpos[s])) > 0.0001){   ###Convergence test
    
    preds <- ((p_wpos[s]/pt_wpos)*pred)/((p_wpos[s]/pt_wpos)*pred + (p_wneg[s]/pt_wneg)*(1-pred))
    preds
    
    p_wneg[s+1] <- (1/length(Auto$Class[idx[[i]]]))*sum((1-preds))
    p_wpos[s+1] <- (1/length(Auto$Class[idx[[i]]]))*sum(preds)
    
    s <- s + 1
  }
  
  outLREMS <- capture.output(p_wneg)
  out1LREMS <- capture.output(p_wpos)
  cat(outLREMS, file="outputLREMS.txt", sep=",", fill = TRUE, append=TRUE)
  cat(out1LREMS, file="output1LREMS.txt", sep=",", fill = TRUE, append=TRUE)
  
  
  au <-  roc.curve(Auto$Class[idx[[i]]], preds)   ###ROC curve
  out2LREMS <-  capture.output(au$auc)   ###AUC
  cat(out2LREMS, file="output2LREMS.txt", sep=",", fill = TRUE, append=TRUE)
  
  ##the ratio of misclassified samples
  CM <- table(Auto$Class[idx[[i]]], preds > 0.5)   ###Confusion matrix
  accuracy <-  sum(diag(CM))/sum(CM)
  out3LREMS <- capture.output(accuracy)
  cat(out3LREMS, file="output3LREMS.txt", sep=",", fill = TRUE, append=TRUE)
  out4LREMS = capture.output(CM)
  cat(out4LREMS, file="output4LREMS.txt", sep=",", fill = TRUE, append=TRUE)
  
  print(au$auc)
  
})

mean(res.lr)
proc.time() - ptm