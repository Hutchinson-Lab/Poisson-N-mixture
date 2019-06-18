'''
Created on Aug 9, 2016

Keras Implementation of Generalized Matrix Factorization (GMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.  

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import numpy as np
import keras
from keras import backend as K
from keras import initializers
from keras.models import Sequential, Model, load_model, save_model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.regularizers import l2
from Dataset import Dataset
from evaluate import evaluate_model
from time import time
import multiprocessing as mp
import sys
import math
import argparse
import pdb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn import metrics

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run GMF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size.')
    parser.add_argument('--regs', nargs='?', default='[0,0]',
                        help="Regularization for user and item embeddings.")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()

def init_normal(shape, name=None):
    return initializers.normal(shape, scale=0.01, name=name)

def get_model(num_users, num_items, latent_dim, regs=[0,0]):
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')

    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = latent_dim, name = 'user_embedding',
                                  embeddings_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None), embeddings_regularizer = l2(regs[0]), input_length=1)
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = latent_dim, name = 'item_embedding',
                                  embeddings_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None), embeddings_regularizer = l2(regs[1]), input_length=1)   
    
    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MF_Embedding_User(user_input))
    item_latent = Flatten()(MF_Embedding_Item(item_input))
    
    # Element-wise product of user and item embeddings 
    #predict_vector = merge([user_latent, item_latent], mode = 'mul')
    predict_vector = keras.layers.Multiply()([user_latent, item_latent])
    
    # Final prediction layer
    #prediction = Lambda(lambda x: K.sigmoid(K.sum(x)), output_shape=(1,))(predict_vector)
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = 'prediction')(predict_vector)
    
    model = Model(inputs=[user_input, item_input], 
                outputs=[prediction])

    return model

def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [],[],[]
    num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train: #train.has_key((u, j)):
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

if __name__ == '__main__':
    args = parse_args()
    num_factors = args.num_factors
    regs = eval(args.regs)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    verbose = args.verbose
    
    topK = 10
    evaluation_threads = 1 #mp.cpu_count()
    print("GMF arguments: %s" %(args))
    y_true_list = []
    y_score_list = []   
    for fold in range(1,11):
        args.dataset = "PPI/fold-"+str(fold)
        model_out_file = 'Pretrain/%s_GMF_%d.h5' %(args.dataset, num_factors)
        
        # Loading data
        t1 = time()
        dataset = Dataset(args.path + args.dataset)
        #train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
        train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings[0], dataset.testNegatives
        validRatings, validNegatives = dataset.validRatings[0], dataset.validNegatives
        num_users, num_items = train.shape
        print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
              %(time()-t1, num_users, num_items, train.nnz, len(testRatings)))
        
        # Build model
        model = get_model(num_users, num_items, num_factors, regs)
        if learner.lower() == "adagrad": 
            model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
        elif learner.lower() == "rmsprop":
            model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
        elif learner.lower() == "adam":
            model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
        else:
            model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')
        #print(model.summary())
        
        # Init performance
        t1 = time()
        #(hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
        (auroc, auprc, y_true, y_score) = evaluate_model(model, validRatings, validNegatives, topK, evaluation_threads)
        #hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        #mf_embedding_norm = np.linalg.norm(model.get_layer('user_embedding').get_weights())+np.linalg.norm(model.get_layer('item_embedding').get_weights())
        #p_norm = np.linalg.norm(model.get_layer('prediction').get_weights()[0])
        #print('Init: HR = %.4f, NDCG = %.4f\t [%.1f s]' % (hr, ndcg, time()-t1))
        print('Init: AUROC = %.4f, AUPRC = %.4f' % (auroc, auprc))

        # Train model
        best_hr, best_ndcg, best_iter = auroc, auprc, -1
        for epoch in range(epochs):
            t1 = time()
            # Generate training instances
            user_input, item_input, labels = get_train_instances(train, num_negatives)
            
            # Training
            hist = model.fit([np.array(user_input), np.array(item_input)], #input
                             np.array(labels), # labels 
                             batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
            t2 = time()
            
            # Evaluation            
            #(hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)                
            #hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            """
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                  % (epoch,  t2-t1, hr, ndcg, loss, time()-t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch                
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)
            """
            (auroc, auprc, y_true, y_score) = evaluate_model(model, validRatings, validNegatives, topK, evaluation_threads)
            loss = hist.history['loss'][0]
            if epoch % verbose == 0:
                print('Fold %d Iteration %d [%.1f s]: AUROC = %.4f, AUPRC = %.4f, loss = %.4f [%.1f s]' 
                          % (fold, epoch,  t2-t1, auroc, auprc, loss, time()-t2))
            if auroc > best_hr:
                best_hr, best_ndcg, best_iter = auroc, auprc, epoch
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)

        print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
        if args.out > 0:
            print("The best GMF model is saved to %s" %(model_out_file))

        model.load_weights(model_out_file)
        (auroc_test, auprc_test, y_true, y_score) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
        print("Test:  AUROC = %.4f, AUPRC = %.4f. " %(auroc_test, auprc_test))
        y_true_list.extend(y_true)
        y_score_list.extend(y_score)

    auroc_avg = roc_auc_score(y_true_list, y_score_list)
    precision, recall, thresholds = precision_recall_curve(y_true_list, y_score_list)
    auprc_avg = metrics.auc(recall, precision)
    print("Avergage:  AUROC = %.4f, AUPRC = %.4f. " %(auroc_avg, auprc_avg))