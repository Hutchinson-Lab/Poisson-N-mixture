#==============================================================================#
# Description: tune for best number of trees for each cross validation fold 
# using RMSE, ROC_AUC, and PR_AUC
# (gbm, randomForest)
# Author: Justin Clarke
# Date: 5/25/19
#==============================================================================#

data = read.csv("trait_count_fold.csv", na.strings = c("","NA"))

# include count in traitCols
traitCols = c(3:14)

ntree_lst_rmse = integer()
ntree_lst_roc = integer()
ntree_lst_pr = integer()
rmse_lst = numeric()

for (i in 1:10) {
	rmse_min = 1000000
	roc_auc_max = 0
	pr_auc_max = 0
	best_tree_num_rmse = 0
	best_tree_num_roc = 0
	best_tree_num_pr = 0
	test = data[which(data$fold == i), ]
	print(" ")
	print("test fold")
	print(unique(test$fold))
	train = data[which(data$fold != i), ]
	print("train folds")
	print(unique(train$fold))
	test = test[ , traitCols]
	train = train[ , traitCols]
	tree_num = c(1000, 2500, 5000, 10000)
	for (j in 1:4) {
		
		# uncomment desired model. Make sure predict function matches desired model.
		#model = gbm(count~., distribution="poisson", data=train, n.trees=tree_num[j])
		model = randomForest(count~., data=train, ntree=tree_num[j])
		
		# uncomment desired predict function
		# predict function for randomForest
		results = predict(model, newdata=test, type="response")
		# predict function for gbm
		#results = predict(model, newdata=test, type="response", n.trees=tree_num[j])
		
		d = test
		if (length(which(is.na(results))) > 0) {
			print("********** NA values in results **********")
		}
		d$Prediction = results
		d$SqError = (d$count - d$Prediction)^2
		rmse = sqrt(sum(d$SqError)/nrow(d))
		if (rmse < rmse_min) {
			rmse_min = rmse
			best_tree_num_rmse = tree_num[j]
		}

		pos = d[which(d$count > 0), ]
		neg = d[which(d$count == 0), ]
		pos = pos$Prediction
		neg = neg$Prediction
		pr = pr.curve(pos, neg, curve=TRUE)
		roc = roc.curve(pos, neg, curve=TRUE)
		pr_auc = pr$auc.integral
		roc_auc = roc$auc

		if (roc_auc > roc_auc_max) {
			roc_auc_max = roc_auc
			best_tree_num_roc = tree_num[j]
		}

		if (pr_auc > pr_auc_max) {
			pr_auc_max = pr_auc
			best_tree_num_pr = tree_num[j]
		}
		
		# print data for each fold
		print("fold rmse")
		print(rmse)
		print("fold ROC AUC")
		print(roc_auc)
		print("fold PR AUC")
		print(pr_auc)
	}
	ntree_lst_rmse = c(ntree_lst_rmse, best_tree_num_rmse)
	ntree_lst_roc = c(ntree_lst_roc, best_tree_num_roc)
	ntree_lst_pr = c(ntree_lst_pr, best_tree_num_pr)
	rmse_lst = c(rmse_lst, rmse_min)
	print("")
	print("best tree num RMSE")
	print(best_tree_num_rmse)
	print("best tree num ROC")
	print(best_tree_num_roc)
	print("best tree num PR")
	print(best_tree_num_pr)

}

print("RMSE votes")
print(ntree_lst_rmse)
print("ROC votes")
print(ntree_lst_roc)
print("PR votes")
print(ntree_lst_pr)







