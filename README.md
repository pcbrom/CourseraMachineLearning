---
title: "Classification model with Random Forest using fine-tuning."
author: "PEDRO CARVALHO BROM"
---

```{r }
## Instructions

# One thing that people regularly do is quantify how much of a particular activity 
# they do, but they rarely quantify how well they do it. In this project, your goal 
# will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 
# 6 participants.

## Review criterialess 

## What you should submit

# The goal of your project is to predict the manner in which they did the exercise. 
# This is the "classe" variable in the training set. You may use any of the other 
# variables to predict with. You should create a report describing how you built 
# your model, how you used cross validation, what you think the expected out of 
# sample error is, and why you made the choices you did. You will also use your 
# prediction model to predict 20 different test cases.

## Peer Review Portion

# Your submission for the Peer Review portion should consist of a link to a Github 
# repo with your R markdown and compiled HTML file describing your analysis. Please 
# constrain the text of the writeup to < 2000 words and the number of figures to be 
# less than 5. It will make it easier for the graders if you submit a repo with a 
# gh-pages branch so the HTML page can be viewed online (and you always want to make 
# it easy on graders :-).

## Course Project Prediction Quiz Portion

# Apply your machine learning algorithm to the 20 test cases available in the test 
# data above and submit your predictions in appropriate format to the Course Project 
# Prediction Quiz for automated grading.

## Reproducibility

# Due to security concerns with the exchange of R code, your code will not be run 
# during the evaluation by your classmates. Please be sure that if they download 
# the repo, they will be able to view the compiled HTML version of your analysis.

## Prediction Assignment Writeupless 

## Background

# Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to 
# collect a large amount of data about personal activity relatively inexpensively. 
# These type of devices are part of the quantified self movement – a group of 
# enthusiasts who take measurements about themselves regularly to improve their 
# health, to find patterns in their behavior, or because they are tech geeks. One 
# thing that people regularly do is quantify how much of a particular activity they 
# do, but they rarely quantify how well they do it. In this project, your goal will 
# be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 
# participants. They were asked to perform barbell lifts correctly and incorrectly 
# in 5 different ways. More information is available from the website here: 
# http://groupware.les.inf.puc-rio.br/har 
# (see the section on the Weight Lifting Exercise Dataset).

## Data

# The training data for this project are available here:
# https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
# https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

# The data for this project come from this source: 
# http://groupware.les.inf.puc-rio.br/har
# If you use the document you create for this class for any purpose please cite them 
# as they have been very generous in allowing their data to be used for this kind of 
# assignment.

# Choosing the prediction algorithm

# Steps Taken
# 1. Tidy data. Remove columns with little/no data.
# 2. Create Training and test data from traing data for cross validation checking.
# 3. Trial one method only: Random Forrest using Train Control Method "cv" 
# and number of resampling 25.
# 4. Fine tune model through combinations of above methods, reduction of input 
# variables or similar. The fine tuning will take into account accuracy first and 
# speed of analysis second.

setwd("/home/pcbrom/Dropbox/Trabalho e Estudo/Cursos Livres/Machine Learning/Curse Project")

# Do Multiple Cores
suppressMessages(require(doMC)); registerDoMC(cores = 4)

# GET DATA

# IMPORT TRAINING AND TESTING

training = read.csv("pml-training.csv")

# Eliminating useless variables

# Note that:
# "X"
# "user_name"               
# "raw_timestamp_part_1"
# "raw_timestamp_part_2"    
# "cvtd_timestamp"
# "new_window"              
# "num_window"

# They are variables that can be added to the model, but every model by adding variables,
# even if it is a numeric sequence with spurious correlation with the response variable,
# increase the hit rate. On the other hand are variables that do not qualitatively 
# contribute to the system, ie, it is not coherent to keep them only to improve the 
# accuracy of the model and say that everything is fine.

training = training[, -c(1:7)]

# Remove bad columns

bad.col = !apply(training, 2, function(x) sum(is.na(x)) > 0.95*nrow(training) || 
                     sum(x == "") > 0.95*nrow(training))
bad.col[is.na(bad.col) == T] = F
training = training[, bad.col]

# Remove near zero values

suppressMessages(require(caret))

training.zeroVar = nearZeroVar(training, saveMetrics = T)

# Remove incomplete lines

training = training[complete.cases(training), ]

# Assessing the Data

str(training)
summary(training)

# DATA ANALYSIS

# Assessing correlated col

suppressMessages(require(corrr))

rdf = correlate(subset(training, select = -c(classe)))
rplot(rdf, print_cor = T, legend = T, colours = heat.colors(20, alpha = .5))

# Using Random Forest

set.seed(2964)

# Partition rows into training and crossvalidation

# In this pretest I made a model valuation adjustment using only 5% of the training database. 
# The aim is to quickly calibrate a model to separate the most important variables. After this 
# step we will use only the most significant with the full dataset training.

inTrain = createDataPartition(training$classe, p = 0.05, list = F)
crossv = training[-inTrain, ]
training2 = training[inTrain, ]

dim(crossv); dim(training2)

mod = suppressMessages(
    train(classe ~ ., method = "rf", data = training2, 
          trControl = trainControl(method = "cv"), number = 25)
)

mod$finalModel
pred.test = predict(mod, crossv); confusionMatrix(pred.test, crossv$classe)

# As might be expected, the accuracy is not high at this time Accuracy: Accuracy: 0.9025 and 95% 
# CI: (0.8981, 0.9067), for use only a trickle minimum database.

# Create fine tunning

mod.varImp = varImp(mod)
plot(mod.varImp, main = "Importance of all Variables for 'rf' model")

# According to the image "Importance of all Variables for 'rf' model" have a potential variable 
# filter with more than 35% of importance to be candidates of the final model.

mod.col = mod.varImp$importance > 35
training = training[, mod.col]

# Create FINE TUNNING, on original training set

mod.ft = suppressMessages(
    train(classe ~ ., method = "rf", data = training, 
          trControl = trainControl(method = "cv"), number = 25)
)

pred.ft.test = predict(mod.ft, crossv)
confusionMatrix(pred.ft.test, crossv$classe)
mod.ft$finalModel

# Thus the final model could not be better. The results are notable: Accuracy > 0.99,
# CI extremely precise and estimate of error rate: < 2%.

# Let's get a beautiful decision tree

suppressMessages(require(tree))

tr = tree(classe ~ . , data = training)
plot(tr); text(tr, cex = .75)

# Prepare the submission. (using COURSERA provided code)

testing = read.csv("pml-testing.csv")
testing = testing[, -c(1:7)]
bad.col = !apply(testing, 2, function(x) sum(is.na(x)) > 0.95*nrow(testing) || 
                     sum(x == "") > 0.95*nrow(testing))
bad.col[is.na(bad.col) == T] = F
testing = testing[, bad.col]
testing = testing[complete.cases(testing), ]

pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}
x = testing

answers = predict(mod.ft, newdata = x); answers
pml_write_files(answers)

# Reference

# [1] Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity
# Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference
# in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

# [2] Revolution Analytics and Steve Weston (2015). doMC: Foreach Parallel Adaptor for 
# 'parallel'. R package version 1.3.4. https://CRAN.R-project.org/package=doMC

# [3] Max Kuhn. Contributions from Jed Wing, Steve Weston, Andre Williams, Chris Keefer, 
# Allan Engelhardt, Tony Cooper, Zachary Mayer, Brenton Kenkel, the R Core Team, Michael 
# Benesty, Reynald Lescarbeau, Andrew Ziem, Luca Scrucca, Yuan Tang and Can Candan. (2016). 
# caret: Classification and Regression Training. R package version 6.0-71. 
# https://CRAN.R-project.org/package=caret

# [4] Simon Jackson (2016). corrr: Correlations in R. R package version 0.2.1. 
# https://CRAN.R-project.org/package=corrr

# [5] Brian Ripley (2016). tree: Classification and Regression Trees. R package version 
# 1.0-37. https://CRAN.R-project.org/package=tree
```
