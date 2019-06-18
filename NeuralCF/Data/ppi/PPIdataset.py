import pandas as pd
import scipy.sparse as sp
import numpy as np
import pdb

Y = pd.read_csv('Y.csv', header=None)
F = pd.read_csv('folds.csv', header=None)
print(Y.shape)
mat = sp.dok_matrix(Y, dtype=np.int32)
F = sp.dok_matrix(F, dtype=np.int32)


bimat = (mat > 0) * 1
bimat.sum(1)

for test in range(1,11):
	train = list(range(1,11))
	if test == 1:
		valid = 10
	else:
		valid = test - 1
	train.remove(test)
	train.remove(valid)
	print(train, valid, test)

	f= open("fold-"+str(test)+".train.rating","w+")
	for i in range(mat.shape[0]):
		for j in range(mat.shape[1]):
			if mat[i,j] > 0 and F[i,j] in train:
				f.write("%d\t%d\t%d\t%d\n" % (i,j,mat[i,j],0))
	f.close()

	f= open("fold-"+str(test)+".valid.rating","w+")
	for i in range(mat.shape[0]):
		for j in range(mat.shape[1]):
			if mat[i,j] > 0 and F[i,j] == valid:
				f.write("%d\t%d\t%d\t%d\n" % (i,j,mat[i,j],0))
	f.close()

	f= open("fold-"+str(test)+".test.rating","w+")
	for i in range(mat.shape[0]):
		for j in range(mat.shape[1]):
			if mat[i,j] > 0 and F[i,j] == test:
				f.write("%d\t%d\t%d\t%d\n" % (i,j,mat[i,j],0))
	f.close()

	f= open("fold-"+str(test)+".valid.all","w+")
	for i in range(mat.shape[0]):
		f.write("(%d,%d)" % (i,i))
		for j in range(mat.shape[1]):
			if F[i,j] == valid:
				f.write("\t%d" %j)
		f.write('\n')
	f.close()

	f= open("fold-"+str(test)+".test.all","w+")
	for i in range(mat.shape[0]):
		f.write("(%d,%d)" % (i,i))
		for j in range(mat.shape[1]):
			if F[i,j] == test:
				f.write("\t%d" %j)
		f.write('\n')
	f.close()