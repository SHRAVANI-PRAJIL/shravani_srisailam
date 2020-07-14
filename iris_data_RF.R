# Using Random Forest
install.packages("randomForest")
library(randomForest)
data(iris)
View(iris)
# Splitting data into training and testing. As the species are in order 
# splitting the data based on species 
iris_setosa<-iris[iris$Species=="setosa",] # 50
iris_versicolor <- iris[iris$Species=="versicolor",] # 50
iris_virginica <- iris[iris$Species=="virginica",] # 50
iris_train <- rbind(iris_setosa[1:25,],iris_versicolor[1:25,],iris_virginica[1:25,])
iris_test <- rbind(iris_setosa[26:50,],iris_versicolor[26:50,],iris_virginica[26:50,])

# Building a random forest model on training data 
fit.forest <- randomForest(Species~.,data=iris_train, na.action=na.roughfix,importance=TRUE)
fit.forest$ntree
# Training accuracy 
mean(iris_train$Species==predict(fit.forest,iris_train)) # 100% accuracy 

# Predicting test data 
pred_test <- predict(fit.forest,newdata=iris_test)
mean(pred_test==iris_test$Species) 
library(gmodels)
# Cross table 
rf_perf<-CrossTable(iris_test$Species, pred_test)


### boosting technique
library(methods)
install.packages("xgboost")
library(xgboost)
library(caret)

data(iris)
View(iris)

datagbm<-iris

mean<-c()
for(i in 1:25){
  
  inTraining <- createDataPartition(datagbm$Species, p = .8, list = FALSE)
  training <- datagbm[ inTraining,]
  testing  <- datagbm[-inTraining,]
  
  
  labeltraining = as.numeric(training[[5]])
  datatraining = as.matrix(training[1:4])
  
  labeltesting  = as.numeric(testing [[5]])
  datatesting  = as.matrix(testing [1:4])
  
  xgtraining <- xgb.DMatrix(data=datatraining, label = labeltraining)
  
  xgtesting <- xgb.DMatrix(data=datatesting, label = labeltesting)
  
  param = list("objective" = "multi:softmax", "bst:eta" = 0.005,"bst:max_depth" = 4,"num_class"=4,"nthread" = 4,"gamma" =0.5,"min_child_weight" = 3)
  ?xgboost
  
  model = xgboost(params = param, data = xgtraining, nround = 1000)#,subsample = 0.8,colsample_bytree = 0.8)
  
  
  ypred = predict(model, xgtesting)
  
  #mean<-mean(labeltesting==ypred)
  
  mean <- c(mean,mean(labeltesting==ypred) )
  
}

summary(mean)
