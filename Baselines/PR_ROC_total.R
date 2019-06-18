#==============================================================================#
# Description: Calculate total PR and ROC AUCs over 10 folds (not averaged)
# Author: Justin Clarke
# Date: 5/28/19
#==============================================================================#

# read in desired model output
#data = read.csv("trait_count_fold_prediction_error_randomForest_10000trees.csv", na.strings = c("","NA"))
data = read.csv("trait_count_fold_prediction_error_gbm_10000trees.csv", na.strings = c("","NA"))
#data = read.csv("trait_count_fold_prediction_error_glm.csv", na.strings = c("","NA"))


pos = data[which(data$count > 0), ]
neg = data[which(data$count == 0), ]

# uncomment rows with correct column name
#pos = pos$rfPrediction
#neg = neg$rfPrediction
pos = pos$gbmPrediction
neg = neg$gbmPrediction
#pos = pos$glmPrediction
#neg = neg$glmPrediction
pr = pr.curve(pos, neg, curve=TRUE)
roc = roc.curve(pos, neg, curve=TRUE)
pr_auc = pr$auc.integral
roc_auc = roc$auc

print("PR AUC")
print(pr_auc)
print("ROC AUC")
print(roc_auc)





