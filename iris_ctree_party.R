install.packages("party")
library(party)
data("iris")
View(iris)
set.seed(1234) #To get reproducible result
ind <- sample(2,nrow(iris), replace=TRUE, prob=c(0.7,0.3))
trainData <- iris[ind==1,]
testData <- iris[ind==2,]
myFormula <- Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width
iris_ctree <- ctree(myFormula, data=trainData)
#train_predict <- predict(iris_ctree)
train_predict <- predict(iris_ctree,trainData,type="response")
table(train_predict,trainData$Species)
acc_train = (100 - (mean(train_predict != trainData$Species)* 100))
acc_train        # accuracy of train = 96.42%
test_predict <- predict(iris_ctree, newdata= testData,type="response")
table(test_predict, testData$Species)
acc_test = (100 - (mean(test_predict != testData$Species)* 100))
acc_test      # accuracy of test = 94.73%
print(iris_ctree)
plot(iris_ctree)
plot(iris_ctree, type="simple")

