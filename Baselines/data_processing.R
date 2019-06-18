#==============================================================================#
# Description: populate trait table with interaction counts and fold numbers.
# Author: Justin Clarke
# Date: 5/25/2019
#==============================================================================#

data = read.csv("Z.csv")
ints = read.csv("Y_Name.csv", row.names=1)
folds = read.csv("folds_new.csv", row.names=1)

# Add interaction column
data$"count" = 9999
data$"fold" = 9999

# iterate over data file
# for each pair get interaction count from ints file
# and fold number from folds file
for (i in 1:nrow(data)) {
	host = data[i, 1]
	parasite = data[i, 2]

	num = ints[parasite, host]
	fld = folds[parasite, host]

	data$"count"[i] = num
	data$"fold"[i] = fld

}

write.csv(data, "trait_count_fold.csv", row.names=FALSE)










