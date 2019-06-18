#==============================================================================#
# Description: Find RMSE across all cross validation folds. 
# (glm)
# Author: Justin Clarke
# Date: 5/26/19
#==============================================================================#

	
data = read.csv("trait_count_fold.csv", na.strings = c("","NA"))

# include count in traitCols
traitCols = c(3:14)
score_lst = list()
glmSqErrorSum = 0

for (i in 1:10) {
	test = data[which(data$fold == i), ]
	print("test fold")
	print(unique(test$fold))
	train = data[which(data$fold != i), ]
	print("train folds")
	print(unique(train$fold))
	test = test[ , traitCols]
	train = train[ , traitCols]
	model = glm(count~., family=poisson, data=train)
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
	# predict function for glm and randomForest
	results = predict(model, newdata=test, type="response")
	
	if (length(which(is.na(results))) > 0) {
		print("********** NA values in results **********")
	}
	d = test
	d$glmPrediction = results
	d$glmSqError = (d$count - d$glmPrediction)^2
	rmse = sqrt(sum(d$glmSqError)/nrow(d))
	print("fold RMSE")
	print(rmse)

	glmSqErrorSum = glmSqErrorSum + sum(d$glmSqError)

	score_lst[[i]] = results
}

# sort by fold, add scores column and calculate RMSE
scores = score_lst[[1]]
for (i in 2:10) {scores = c(scores, score_lst[[i]])}
d = data[order(data$fold), ]
d$glmPrediction = scores
d$glmSqError = (d$count - d$glmPrediction)^2
rmse = sqrt(sum(d$glmSqError)/nrow(d))
print("Final RMSE")
print(rmse)
rRMSE = rmse / mean(d$count)
print("Final rRMSE")
print(rRMSE)

if (abs(sum(d$glmSqError) - glmSqErrorSum) > .00001) {
	print("********** Error in sqError sum **********")
}

write.csv(d, "trait_count_fold_prediction_error_glm.csv", row.names=FALSE)








