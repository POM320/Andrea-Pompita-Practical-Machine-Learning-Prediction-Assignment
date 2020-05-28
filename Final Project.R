library(caret)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(randomForest)
library(corrplot)
library(gbm)
library(dplyr)

# Loading Data
train.data = read.csv("C:/Users/Hades/OneDrive/EDHEC/Semester 2/Topics in FE/Project/pml-training.csv", header=T)
val.data = read.csv("C:/Users/Hades/OneDrive/EDHEC/Semester 2/Topics in FE/Project/pml-testing.csv", header=T)


# Data cleaning

# Removing all the columns containing "NA"
train.no.na.index = colSums(is.na(train.data))==0
val.no.na.index = colSums(is.na(val.data))==0

training = train.data[,train.no.na.index]
validation = val.data[,val.no.na.index]

#Removing the first 7 columns that contain values not useful for predictions
training = training[,c(8:ncol(training))]
validation = validation[,c(8:ncol(validation))]

#Preparing dataset
set.seed(1234)
#Splitting the training dataset in two
inTrain = createDataPartition(training$classe, p=0.7, list=F)
train.set = training[inTrain,]
test.set = training[-inTrain,]
# Removing non useful variables (near zero variance)
NZV = nearZeroVar(train.set)
train.set = train.set[,-NZV]
test.set = test.set[,-NZV]

# Choose predictors
# Selecting only predictors higly correlated with the outcome (cor>=0.75)
cor.mat = cor(train.set[,-53])
predictors = findCorrelation(cor.mat, cutoff=0.75)
names(train.set)[predictors]

# Models training
# 3 models are used: Decision Tree, Random Forest and Gradient Boosting
# The one that perform better will be used on the validation data

# Decision Tree
set.seed(12345)
tree.mod = train(classe ~ ., data=train.set, method="rpart")
fancyRpartPlot(tree.mod$finalModel)

predict.tree = predict(tree.mod, newdata=test.set)
cmtree = confusionMatrix(predict.tree, test.set$classe)
cmtree

# Random Forest
control.rf = trainControl(method="cv", number=5, verboseIter=FALSE)
rf.mod = train(classe ~ ., data=train.set, method='rf', trControl=control.rf)
predict.rf = predict(rf.mod, newdata=test.set)
cmrf = confusionMatrix(predict.rf, test.set$classe)
cmrf

# GBM
gbm.mod = train(classe ~ ., data=train.set, method='gbm', verbose= F)
predict.gbm = predict(gbm.mod, newdata=test.set)
cmgbm = confusionMatrix(predict.gbm, test.set$classe)
cmgbm

# Prediction
# The algorithm performing better is the random forest. It will be used to
# for prediction on the validation data set

results = predict(rf.mod, newdata = validation)
results