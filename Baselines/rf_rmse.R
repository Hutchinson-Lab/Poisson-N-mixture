#==============================================================================#
# Description: Find RMSE across all cross validation folds. 
# (randomForest)
# Author: Justin Clarke
# Date: 5/25/19
#==============================================================================#

data = read.csv("trait_count_fold.csv", na.strings = c("","NA"))

# include count in traitCols
traitCols = c(3:14)
score_lst = list()
rfSqErrorSum = 0

for (i in 1:10) {
	test = data[which(data$fold == i), ]
	print("test fold")
	print(unique(test$fold))
	train = data[which(data$fold != i), ]
	print("train folds")
	print(unique(train$fold))
	test = test[ , traitCols]
	train = train[ , traitCols]
	model = randomForest(count~., data=train, ntree=10000)
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
	d$rfPrediction = results
	d$rfSqError = (d$count - d$rfPrediction)^2
	rmse = sqrt(sum(d$rfSqError)/nrow(d))
	print("fold RMSE")
	print(rmse)

	# error checking
	print("10th prediction:")
	print(d$rfPrediction[10])
	rfSqErrorSum = rfSqErrorSum + sum(d$rfSqError)

	score_lst[[i]] = results
}

scores = score_lst[[1]]
for (i in 2:10) {scores = c(scores, score_lst[[i]])}
d = data[order(data$fold), ]
d$rfPrediction = scores
d$rfSqError = (d$count - d$rfPrediction)^2
rmse = sqrt(sum(d$rfSqError)/nrow(d))
print("Final RMSE")
print(rmse)
rRMSE = rmse / mean(d$count)
print("Final rRMSE")
print(rRMSE)

if (abs(sum(d$rfSqError) - rfSqErrorSum) > .00001) {
	print("********** Error in sqError sum **********")
}

write.csv(d, "trait_count_fold_prediction_error_randomForest_10000trees.csv", row.names=FALSE)








