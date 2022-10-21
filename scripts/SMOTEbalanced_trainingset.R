
library(DMwR)
library(ggplot2)

getwd()

setwd("~/Desktop/DS340W/term_project/data/wisconsin")

train <- read.csv("wisconsin_train.csv", header = TRUE)
test <- read.csv("wisconsin_test.csv", header = TRUE)

train$diagnosis <- as.factor(train$diagnosis)
test$diagnosis <- as.factor(test$diagnosis)
#View(train)
#View(test)

# Apply SMOTE to training data
train_bal <- SMOTE(diagnosis ~ ., train, perc.over = 400, perc.under = 120)

nrow(train_bal) # now have 1460 total training cases
table(train_bal$diagnosis) # approximately equal class distribution, with slightly more malignant cases than benign

ggplot(train_bal, aes(x=reorder(diagnosis, diagnosis, function(x)-length(x)))) +
  geom_bar(fill='light green') +  labs(x='Diagnosis')

# save balanced/augmented training data
write.csv(train_bal,"wisconsin_train_balanced.csv", row.names = FALSE)