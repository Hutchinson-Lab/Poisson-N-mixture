#==============================================================================#
# Description: Find RMSE across all cross validation folds. 
# (gbm)
# Author: Justin Clarke
# Date: 5/26/19
#==============================================================================#

	
data = read.csv("trait_count_fold.csv", na.strings = c("","NA"))

# include count in traitCols
traitCols = c(3:14)
score_lst = list()
gbmSqErrorSum = 0

for (i in 1:10) {
	test = data[which(data$fold == i), ]
	print("test fold")
	print(unique(test$fold))
	train = data[which(data$fold != i), ]
	print("train folds")
	print(unique(train$fold))
	test = test[ , traitCols]
	train = train[ , traitCols]
	model = gbm(count~., distribution="poisson", data=train, n.trees=2500)
	#print("=============================================================================")
	#print(paste("Run: ", as.character(i)))
	#print("Summary: ")
	#print(summary(model))
	#print("+++++++++++++++++++++++++++++++++++")
	#print(anova(model))
	#print("+++++++++++++++++++++++++++++++++++")
	#print("R^2")
	#print(pR2(model))
	#print("+++++++++++++++++++++++++++++++++++")
	#print("Testing model")
	# predict function for gbm
	results = predict(model, newdata=test, type="response", n.trees=2500)
	
	if (length(which(is.na(results))) > 0) {
		print("********** NA values in results **********")
	}
	d = test
	d$gbmPrediction = results
	d$gbmSqError = (d$count - d$gbmPrediction)^2
	rmse = sqrt(sum(d$gbmSqError)/nrow(d))
	print("fold RMSE")
	print(rmse)

	gbmSqErrorSum = gbmSqErrorSum + sum(d$gbmSqError)

	score_lst[[i]] = results
}

# sort by fold, add scores column and calculate RMSE
scores = score_lst[[1]]
for (i in 2:10) {scores = c(scores, score_lst[[i]])}
d = data[order(data$fold), ]
d$gbmPrediction = scores
d$gbmSqError = (d$count - d$gbmPrediction)^2
rmse = sqrt(sum(d$gbmSqError)/nrow(d))
print("Final RMSE")
print(rmse)
rRMSE = rmse / mean(d$count)
print("Final rRMSE")
print(rRMSE)

if (abs(sum(d$gbmSqError) - gbmSqErrorSum) > .00001) {
	print("********** Error in sqError sum **********")
}

write.csv(d, "trait_count_fold_prediction_error_gbm_2500trees.csv", row.names=FALSE)








