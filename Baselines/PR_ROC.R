#==============================================================================#
# Description: Calculate average PR and ROC AUCs over 10 folds
# Author: Justin Clarke
# Date: 5/28/19
#==============================================================================#

data = read.csv("trait_count_fold_prediction_error_gbm_10000trees.csv", na.strings = c("","NA"))

pr_sum = 0
roc_sum = 0
pr_vals = numeric()
roc_vals = numeric()

for (i in 1:10) {
	rows = data[which(data$fold == i), ]
	pos = rows[which(rows$count > 0), ]
	neg = rows[which(rows$count == 0), ]
	if (nrow(rows) != nrow(pos) + nrow(neg)) {
		print("Rows don't add up")
	}
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
	pr_vals = c(pr_vals, pr_auc)
	roc_vals = c(roc_vals, roc_auc)
	pr_sum = pr_sum + pr_auc
	roc_sum = roc_sum + roc_auc
	
}

print("average PR AUC")
print(pr_sum/10)
print("average ROC AUC")
print(roc_sum/10)





