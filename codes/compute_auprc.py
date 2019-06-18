from numpy import genfromtxt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn import metrics

y_true = genfromtxt('results/y_true.csv', delimiter=',')
y_score = genfromtxt('results/y_score.csv', delimiter=',')
precision, recall, thresholds = precision_recall_curve(y_true, y_score)

#PR = genfromtxt('results/PR.csv', delimiter=',')
#precision = PR[:,0]
#recall = PR[:,1]
auprc = metrics.auc(recall, precision)

print(round(auprc,4))