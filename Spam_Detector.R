#Please set the currrent working directory to the downloaded directory
setwd("/Users/sahilgaba/Desktop/GitHub/SpamFilter")
dataImport = read.table("SpamData.txt", sep = ",", header = FALSE)

#Importing all the libraries
library(caret)
library(MASS)
library(klaR)
library(ada)
library(mice)
library(glmnet)
library(corrplot)
library(xgboost)

#Checking data for abnormal values and outliers
summary(dataImport)
md.pattern(dataImport)
corrplot(cor(xValues), order="AOE",tl.pos="n", method = "square")

#Generating training and test samples
xValues = dataImport[, 1:57]
yValues = dataImport[, 58]
randomSample = createDataPartition(yValues, p = 0.8, list = FALSE)
yTrain = yValues[randomSample]
yTest = yValues[-randomSample]
xTrain = xValues[randomSample,]
xTest = xValues[-randomSample,]

#Correlation Plot
corrplot(cor(xValues), order="AOE",tl.pos="n", method = "square")


#Performing Logistic Regression
#Rating the models based on customised score = 0.7 * (Ratio of non-spams correctly classified) + 0.3* (Ratio of spams correctly classified)
#Note that classfying non-spams correctly is given more importance

#Ridge
lModel0 = cv.glmnet(x = as.matrix(xTrain), y = as.matrix(yTrain), nfolds = 10, family = "binomial", alpha = 0)
plot(lModel0)
predictLModel0 = ifelse(predict(lModel0, newx = as.matrix(xTest), type = "response" ,s = "lambda.min") > 0.5, 1, 0)
confMatrix0 = confusionMatrix(predictLModel0, yTest)
scoreLModel0 = confMatrix0$byClass[1] * 0.7 + confMatrix0$byClass[2] * 0.3 #Score calculated using sensitivity and specificity 

#Elasticnet with alpha = 0.25
lModel0.25 = cv.glmnet(x = as.matrix(xTrain), y = as.matrix(yTrain), nfolds = 10, family = "binomial", alpha = 0.25)
plot(lModel0.25)
predictLModel0.25 = ifelse(predict(lModel0.25, newx = as.matrix(xTest), type = "response" ,s = "lambda.min") > 0.5, 1, 0)
confMatrix0.25 = confusionMatrix(predictLModel0.25, yTest)
scoreLModel0.25 = confMatrix0.25$byClass[1] * 0.7 + confMatrix0.25$byClass[2] * 0.3

#Elasticnet with alpha = 0.5
lModel0.5 = cv.glmnet(x = as.matrix(xTrain), y = as.matrix(yTrain), nfolds = 10, family = "binomial", alpha = 0.5)
plot(lModel0.5)
predictLModel0.5 = ifelse(predict(lModel0.5, newx = as.matrix(xTest), type = "response" ,s = "lambda.min") > 0.5, 1, 0)
confMatrix0.5 = confusionMatrix(predictLModel0.5, yTest)
scoreLModel0.5 = confMatrix0.25$byClass[1] * 0.7 + confMatrix0.5$byClass[2] * 0.3

#Elasticnet with alpha = 0.75
lModel0.75 = cv.glmnet(x = as.matrix(xTrain), y = as.matrix(yTrain), nfolds = 10, family = "binomial", alpha = 0.75)
plot(lModel0.75)
predictLModel0.75 = ifelse(predict(lModel0.75, newx = as.matrix(xTest), type = "response" ,s = "lambda.min") > 0.5, 1, 0)
confMatrix0.75 = confusionMatrix(predictLModel0.75, yTest)
scoreLModel0.75 = confMatrix0.75$byClass[1] * 0.7 + confMatrix0.75$byClass[2] * 0.3

#Lasso
lModel1 = cv.glmnet(x = as.matrix(xTrain), y = as.matrix(yTrain), nfolds = 10, family = "binomial", alpha = 1)
plot(lModel1)
predictLModel1 = ifelse(predict(lModel1, newx = as.matrix(xTest), type = "response" ,s = "lambda.min") > 0.5, 1, 0)
confMatrix1 = confusionMatrix(predictLModel1, yTest)
scoreLModel1 = confMatrix1$byClass[1] * 0.7 + confMatrix1$byClass[2] * 0.3

#alpha = 0.5 gives the best (best Logistic model)

#Running diagnostics on lasso model
predictLModel0.5Train = ifelse(predict(lModel0.5, newx = as.matrix(xTrain), type = "response" ,s = "lambda.min") > 0.5, 1, 0)
confMatrix0.5Train = confusionMatrix(predictLModel0.5Train, yTrain)
scoreLModel0.5Train = confMatrix0.5Train$byClass[1] * 0.7 + confMatrix0.5Train$byClass[2] * 0.3
# Training error < Test error --> Some overfitting (High variance) might be there

#Extracting attributes as suggested by Lasso
indexLambdaMin = which(lModel1$lambda == lModel1$lambda.min)
lModel1$glmnet.fit[2]$beta[,indexLambdaMin]
xTrain = xTrain[, -c(32,34)]
xTest = xTest[, -c(32,34)]

#PCA to see if attributes can be reduced further to decrease variance
#Default uses 95% information available
pca = preProcess(xTrain, method = "pca")
pcaXTrain = as.data.frame(as.matrix(xTrain) %*% as.matrix(pca$rotation))
pcaXTest = as.data.frame(as.matrix(xTest) %*% as.matrix(pca$rotation))
pcaLModel0.5 = cv.glmnet(x = as.matrix(pcaXTrain), y = as.matrix(yTrain), nfolds = 10, family = "binomial", alpha = 0.5)
plot(pcaLModel0.5)
predictPcaLModel0.5 = ifelse(predict(pcaLModel0.5, newx = as.matrix(pcaXTest), type = "response" ,s = "lambda.min") > 0.5, 1, 0)
pcaConfMatrix0.5 = confusionMatrix(predictPcaLModel0.5, yTest)
pcaScoreLModel0.5 = pcaConfMatrix0.5$byClass[1] * 0.7 + pcaConfMatrix0.5$byClass[2] * 0.3
#Score reduced drastically by choosing 95% information to work with

#We will have to use all 55 features

#SVM
#Tuning for the Cost

#Normalizing for SVM (helps when algo has distance calculation)
xTrainSVM = scale(xTrain)
xTestSVM = scale(xTest)
#C = 100
trainControl = trainControl(method = "cv", number = 10)
svmGrid100 = expand.grid(C = 100)
svmModel100 = train(y = as.factor(yTrain), x = xTrainSVM, trControl = trainControl, method = "svmLinear", tuneGrid = svmGrid100)
predictSVM100 = predict(svmModel100, data.frame(xTestSVM))
svmConfMatrix100 = confusionMatrix(predictSVM100, yTest)
scoreSVM100 = svmConfMatrix100$byClass[1] * 0.7 + svmConfMatrix100$byClass[2] * 0.3

#C = 10
svmGrid10 = expand.grid(C = 10)
svmModel10 = train(y = as.factor(yTrain), x = xTrainSVM, trControl = trainControl, method = "svmLinear", tuneGrid = svmGrid10)
predictSVM10 = predict(svmModel10, data.frame(xTestSVM))
svmConfMatrix10 = confusionMatrix(predictSVM10, yTest)
scoreSVM10 = svmConfMatrix10$byClass[1] * 0.7 + svmConfMatrix10$byClass[2] * 0.3

#C = 1
svmGrid1 = expand.grid(C = 1)
svmModel1 = train(y = as.factor(yTrain), x = xTrainSVM, trControl = trainControl, method = "svmLinear", tuneGrid = svmGrid1)
predictSVM1 = predict(svmModel1, data.frame(xTestSVM))
svmConfMatrix1 = confusionMatrix(predictSVM1, yTest)
scoreSVM1 = svmConfMatrix1$byClass[1] * 0.7 + svmConfMatrix1$byClass[2] * 0.3

#C = 10 has best score among other linear SVM models

#SVM Radial
svmRadialData = data.frame(y = yTrain, x = xTrainSVM)
#Tune radial svm
svmRadialTune = sigest(y ~ ., svmRadialData, frac = 1)
svmGridRadial = expand.grid(.C = 10, .sigma = svmRadialTune[1])
svmModelRadial = train(y = as.factor(yTrain), x = xTrainSVM, trControl = trainControl, method = "svmRadial", tuneGrid = svmGridRadial)
predictSVMRadial = predict(svmModelRadial, data.frame(xTestSVM))
svmConfMatrixRadial = confusionMatrix(predictSVMRadial, yTest)
scoreSVMRadial = svmConfMatrixRadial$byClass[1] * 0.7 + svmConfMatrixRadial$byClass[2] * 0.3

#SVM radial works better than linear (best for SVM)

#RF
rfModel500 = train(y = as.factor(yTrain), x = xTrain, trControl = trainControl, method = "rf", ntree = 500, allowParallel = TRUE)
predictRF = predict(rfModel500, data.frame(xTest))
rfConfMatrix500 = confusionMatrix(predictRF, yTest)
scoreRF500 = rfConfMatrix500$byClass[1] * 0.7 + rfConfMatrix500$byClass[2] * 0.3

#Score has improved further. Let's tune our RF
k = tuneRF(x = xTrain, y = yTrain, plot = TRUE, mtryStart = 2, stepFactor = 2)
#mtry 8 gives best results

#RF mtry = 8
rfGrid = expand.grid(.mtry = 8)
rfModel500 = train(y = as.factor(yTrain), x = xTrain, trControl = trainControl, method = "rf", ntree = 500, allowParallel = TRUE, tuneGrid = rfGrid)
predictRF = predict(rfModel500, data.frame(xTest))
rfConfMatrix500 = confusionMatrix(predictRF, yTest)
scoreRF500 = rfConfMatrix500$byClass[1] * 0.7 + rfConfMatrix500$byClass[2] * 0.3

#there is not much difference between two RF models, we can choose either

#Let's crank it up with boosting

#Gradient Boosting
#Tuning GBM for shrinkage (learning rate)

#shrinkage = 0.1
gbGrid0.1 = expand.grid(n.trees = 500, interaction.depth = 10, shrinkage = 0.1, n.minobsinnode = 0)
gbModel0.1 = train(y = as.factor(yTrain), x = xTrain, trControl = trainControl, method = "gbm", tuneGrid = gbGrid0.1)
predictGB0.1 = predict(gbModel0.1, xTest)
gbConfMatrix0.1 = confusionMatrix(predictGB0.1, yTest)
scoreGB0.1 = gbConfMatrix0.1$byClass[1] * 0.7 + gbConfMatrix0.1$byClass[2] * 0.3

#shrinkage = 0.05
gbGrid0.05 = expand.grid(n.trees = 500, interaction.depth = 10, shrinkage = 0.05, n.minobsinnode = 0)
gbModel0.05 = train(y = as.factor(yTrain), x = xTrain, trControl = trainControl, method = "gbm", tuneGrid = gbGrid0.05)
predictGB0.05 = predict(gbModel0.05, xTest)
gbConfMatrix0.05 = confusionMatrix(predictGB0.05, yTest)
scoreGB0.05 = gbConfMatrix0.05$byClass[1] * 0.7 + gbConfMatrix0.05$byClass[2] * 0.3

#shrinkage = 0.03
gbGrid0.03 = expand.grid(n.trees = 500, interaction.depth = 10, shrinkage = 0.03, n.minobsinnode = 0)
gbModel0.03 = train(y = as.factor(yTrain), x = xTrain, trControl = trainControl, method = "gbm", tuneGrid = gbGrid0.03)
predictGB0.03 = predict(gbModel0.03, xTest)
gbConfMatrix0.03 = confusionMatrix(predictGB0.03, yTest)
scoreGB0.03 = gbConfMatrix0.03$byClass[1] * 0.7 + gbConfMatrix0.03$byClass[2] * 0.3

#GB for shrinkage = 0.05 works the best (best for GB)

#Now, the darling of Kagglers, XGBoost
#optimizing for eta (learning rate)

#eta = 0.7
xgbModel0.7 <- xgboost(data = data.matrix(xTrain), 
                       label = yTrain, 
                       eta = 0.7,
                       max_depth = 10, 
                       nround = 25, 
                       subsample = 1,
                       eval_metric = "logloss",
                       objective = "reg:logistic",
                       nthread = 3, lambda = 1, alpha = 0)
predictXGB0.7 <- ifelse(predict(xgbModel0.7, data.matrix(xTest)) > 0.5, 1,0)
xgbConfMatrix0.7 = confusionMatrix(predictXGB0.7, yTest)
scoreXGB0.7 = xgbConfMatrix0.1$byClass[1] * 0.7 + xgbConfMatrix0.1$byClass[2] * 0.3

#eta = 0.5
xgbModel0.5 <- xgboost(data = data.matrix(xTrain), 
               label = yTrain, 
               eta = 0.5,
               max_depth = 10, 
               nround = 25, 
               subsample = 1,
               eval_metric = "logloss",
               objective = "reg:logistic",
               nthread = 3, lambda = 1, alpha = 0)
predictXGB0.5 <- ifelse(predict(xgbModel0.5, data.matrix(xTest)) > 0.5, 1,0)
xgbConfMatrix0.5 = confusionMatrix(predictXGB0.5, yTest)
scoreXGB0.5 = xgbConfMatrix0.5$byClass[1] * 0.7 + xgbConfMatrix0.5$byClass[2] * 0.3

#eta = 0.3
xgbModel0.3 <- xgboost(data = data.matrix(xTrain), 
                       label = yTrain, 
                       eta = 0.3,
                       max_depth = 10, 
                       nround = 25, 
                       subsample = 1,
                       eval_metric = "logloss",
                       objective = "reg:logistic",
                       nthread = 3, lambda = 1, alpha = 0)
predictXGB0.3 <- ifelse(predict(xgbModel0.3, data.matrix(xTest)) > 0.5, 1,0)
xgbConfMatrix0.3 = confusionMatrix(predictXGB0.3, yTest)
scoreXGB0.3 = xgbConfMatrix0.3$byClass[1] * 0.7 + xgbConfMatrix0.3$byClass[2] * 0.3

#XGB for eta = 0.5 works the best (best for XGB)

#Creating ensemble
#Correlation Plot
resultAll = data.frame(log = as.numeric(predictLModel0.5), SVMRadial = as.numeric(predictSVMRadial) - 1, RF = as.numeric(predictRF) - 1, GB = as.numeric(predictGB0.05) - 1, XGBoost = as.numeric(predictXGB0.5))
corrplot(cor(resultAll), method="number")

#Voting Ensembles
#XGB, SVM, Logistic ensemble
resultXGBLSVM = data.frame(XGB = as.numeric(predictXGB0.5), SVMRadial = as.numeric(predictSVMRadial) - 1, RF = as.numeric(predictRF) - 1)
predictXGBLSVM = ifelse(rowSums(resultXGBLSVM) < 2, 0,1)
xgbLSVMConfMatrix = confusionMatrix(predictXGBLSVM, yTest)
scoreXGBLSVM = xgbLSVMConfMatrix$byClass[1] * 0.7 + xgbLSVMConfMatrix$byClass[2] * 0.3

#XGB, RF emsemble
predictXGBRF = ifelse(predictXGB0.5 == 1, 1, as.numeric(predictRF) - 1)
xgbRFConfMatrix = confusionMatrix(predictXGBRF, yTest)
scoreXGBRF = xgbRFConfMatrix$byClass[1] * 0.7 + xgbRFConfMatrix$byClass[2] * 0.3


#XGB, GB, RF ensemble
resultXGBRFGB = data.frame(XGB = as.numeric(predictXGB0.5), GB0.05 = as.numeric(predictGB0.05) - 1, RF = as.numeric(predictRF) - 1)
predictXGBRFGB = ifelse(rowSums(resultXGBRFGB) < 2, 0,1)
xgbRFGBConfMatrix = confusionMatrix(predictXGBRFGB, yTest)
scoreXGBRFGB = xgbRFGBConfMatrix$byClass[1] * 0.7 + xgbRFGBConfMatrix$byClass[2] * 0.3

#None of the ensembles seem to be useful but XGB, GB and RF ensemble is better among others

#Stacking ensemble

#XGB Stacking on XGB, GB, RF 
#Creating training DF
predictXGB0.5Train <- ifelse(predict(xgbModel0.5, data.matrix(xTrain)) > 0.5, 1,0)
predictGB0.05Train = predict(gbModel0.05, xTrain)
predictRFTrain = predict(rfModel500, data.frame(xTrain))
resultXGBRFGBTrain = data.frame(XGB = as.numeric(predictXGB0.5Train), GB0.05 = as.numeric(predictGB0.05Train) - 1, RF = as.numeric(predictRFTrain) - 1)
#Creating testing DF
predictXGB0.5Test <- ifelse(predict(xgbModel0.5, data.matrix(xTest)) > 0.5, 1,0)
predictGB0.05Test = predict(gbModel0.05, xTest)
predictRFTest = predict(rfModel500, data.frame(xTest))
resultXGBRFGBTest = data.frame(XGB = as.numeric(predictXGB0.5Test), GB0.05 = as.numeric(predictGB0.05Test) - 1, RF = as.numeric(predictRFTest) - 1)
#Running stacking XGB model
xgbModelStackXGBRFGB0.3 <- xgboost(data = data.matrix(resultXGBRFGBTrain), 
                       label = yTrain, 
                       eta = 0.05,
                       max_depth = 8, 
                       nround = 15, 
                       subsample = 1,
                       eval_metric = "logloss",
                       objective = "reg:logistic",
                       nthread = 3, lambda = 1, alpha = 0)

predictXGBRFGBStack = ifelse(predict(xgbModelStackXGBRFGB0.3, data.matrix(resultXGBRFGBTest)) > 0.5, 1,0)
xgbConfMatrixXGBRFGBStack = confusionMatrix(predictXGBRFGBStack, yTest)
scoreXGBRFGBStack = xgbConfMatrixXGBRFGBStack$byClass[1] * 0.7 + xgbConfMatrixXGBRFGBStack$byClass[2] * 0.3


#Running stacking RF
rfGrid = expand.grid(.mtry = 2)
rfModel500Stack = train(y = as.factor(yTrain), x = resultXGBRFGBTrain, trControl = trainControl, method = "rf", ntree = 500, allowParallel = TRUE, tuneGrid = rfGrid)
predictRFStack = predict(rfModel500Stack, data.frame(resultXGBRFGBTest))
rfConfMatrix500Stack = confusionMatrix(predictRFStack, yTest)
scoreRF500Stack = rfConfMatrix500Stack$byClass[1] * 0.7 + rfConfMatrix500Stack$byClass[2] * 0.3

#Stacking does not seem to give us any gains. so, we'll stick to RF model

