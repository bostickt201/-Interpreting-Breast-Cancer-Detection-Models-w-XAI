
getwd()
setwd("~/Desktop/DS340W/term_project/data/wisconsin")

# read in data
train <- read.csv("wisconsin_train.csv", header = TRUE)

# subset rfe-picked cols and target
rfe_subset <- train[, c(2,8,11,14,21,22,23,24,25,27,28,31)]
#View(rfe_subset)

# save rfe subset as csv
write.csv(rfe_subset, "rfe_train.csv", row.names = FALSE)