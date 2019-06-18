'''
Created on Apr 15, 2016
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)

@author: hexiangnan
'''
import math
import heapq # for retrieval topK
import multiprocessing
import numpy as np
from time import time
import pdb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
#from numba import jit, autojit

# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None

def xrange(x):
    return iter(range(x))

def evaluate_model(model, testRatings, testNegatives, K, num_thread):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _K
    global _test
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives    
    _K = K
        
    hits, ndcgs = [],[]
    if(num_thread > 1): # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return (hits, ndcgs)
    # Single thread
    for idx in xrange(len(_testRatings)):
        (hr,ndcg) = eval_one_rating(idx)   
        hits.extend(hr)   # y_true per user   
        ndcgs.extend(ndcg) # y_score per user
    
    #pdb.set_trace()
    y_true = hits # y_true for all
    y_score = ndcgs # y_score for all
    auroc = roc_auc_score(y_true, y_score)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    auprc = metrics.auc(recall, precision)
    return (auroc, auprc, y_true, y_score)
    
    #return (hits, ndcgs)


def eval_one_rating(idx):    
    rating = _testRatings[idx]
    items = _testNegatives[idx]    
    u = rating[0]
    gtItem = rating[1]
    #items.extend(gtItem)
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype = 'int32')
    y_score = _model.predict([users, np.array(items)], batch_size=100, verbose=0)
    y_true = np.zeros(len(y_score))
    true_idx = [i for (i,elem) in enumerate(items) if elem in gtItem]
    y_true[true_idx] = 1
    """
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    #pdb.set_trace()
    items.pop()
    
    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    #pdb.set_trace()
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    #pdb.set_trace()
    """

    return (y_true, y_score)

def getHitRatio(ranklist, gtItem):
    #pdb.set_trace()
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0