
library(mlbench)
library(caret)

setwd("~/Desktop/DS340W/term_project/data/wisconsin")

train <- read.csv("wisconsin_train_balanced.csv", header = TRUE)
#View(train)

train$diagnosis <- as.factor(train$diagnosis)

# define the control using random forest as selection function
control <- rfeControl(functions=rfFuncs, method="cv", number=10)

# run the RFE algorithm
results <- rfe(train[,1:30], train[,31], sizes=c(1:30), rfeControl=control)

# summarize the results
print(results)

# identify chosen features
predictors(results)

# plot results
plot(results, type=c("g", "o"))

# convert results to df and export as csv
resultsTable <- as.data.frame(results$results)
write.csv(resultsTable, "rfeResults.csv")

# convert predictors to df and export as csv
rfePreds <- as.data.frame(predictors(results))
write.csv(rfePreds, "rfe_predictors.csv")
