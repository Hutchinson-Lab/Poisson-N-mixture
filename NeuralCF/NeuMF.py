'''
Created on Aug 9, 2016
Keras Implementation of Neural Matrix Factorization (NeuMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.  

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import numpy as np


import keras
from keras import backend as K
from keras import initializers
from keras.regularizers import l1, l2
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten, Dropout
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from evaluate import evaluate_model
from Dataset import Dataset
from time import time
import sys
import GMF, MLP
import argparse
import pdb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
import csv

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size of MF model.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_mf', type=float, default=0,
                        help='Regularization for MF embeddings.')                    
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each MLP layer. reg_layers[0] is the regularization for embeddings.")
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
    parser.add_argument('--mf_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MF part. If empty, no pretrain will be used')
    parser.add_argument('--mlp_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MLP part. If empty, no pretrain will be used')
    parser.add_argument('--metric', nargs='?', default='auroc',
                        help='Specify the pretrain model file for MLP part. If empty, no pretrain will be used')
    return parser.parse_args()

def init_normal(shape, name=None):
    return initializers.normal(shape, scale=0.01, name=name)

def get_model(num_users, num_items, mf_dim=10, layers=[10], reg_layers=[0], reg_mf=0):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers) #Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')
    
    # Embedding layer
    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = mf_dim, name = 'mf_embedding_user',
                                  embeddings_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None), embeddings_regularizer = l2(reg_mf), input_length=1)
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = mf_dim, name = 'mf_embedding_item',
                                  embeddings_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None), embeddings_regularizer = l2(reg_mf), input_length=1)   

    MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = int(layers[0]/2), name = "mlp_embedding_user",
                                  embeddings_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None), embeddings_regularizer = l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = int(layers[0]/2), name = 'mlp_embedding_item',
                                  embeddings_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None), embeddings_regularizer = l2(reg_layers[0]), input_length=1)   
    
    # MF part
    mf_user_latent = Flatten()(MF_Embedding_User(user_input))
    mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
    #mf_vector = merge([mf_user_latent, mf_item_latent], mode = 'mul') # element-wise multiply
    mf_vector = keras.layers.Multiply()([mf_user_latent, mf_item_latent])

    # MLP part 
    mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
    mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
    #mlp_vector = merge([mlp_user_latent, mlp_item_latent], mode = 'concat')
    mlp_vector = keras.layers.Concatenate(axis=-1)([mlp_user_latent, mlp_item_latent])
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], kernel_regularizer = l2(reg_layers[idx]),  bias_regularizer = l2(reg_layers[idx]),  activation='relu', name="layer%d" %idx)
        mlp_vector = layer(mlp_vector)

    # Concatenate MF and MLP parts
    #mf_vector = Lambda(lambda x: x * alpha)(mf_vector)
    #mlp_vector = Lambda(lambda x : x * (1-alpha))(mlp_vector)
    #predict_vector = merge([mf_vector, mlp_vector], mode = 'concat')
    predict_vector = keras.layers.Concatenate(axis=-1)([mf_vector, mlp_vector])
    
    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', bias_initializer ='lecun_uniform', name = "prediction")(predict_vector)
    
    model = Model(inputs=[user_input, item_input], 
                  outputs=[prediction])
    
    return model

def load_pretrain_model(model, gmf_model, mlp_model, num_layers):
    # MF embeddings
    gmf_user_embeddings = gmf_model.get_layer('user_embedding').get_weights()
    gmf_item_embeddings = gmf_model.get_layer('item_embedding').get_weights()
    model.get_layer('mf_embedding_user').set_weights(gmf_user_embeddings)
    model.get_layer('mf_embedding_item').set_weights(gmf_item_embeddings)
    
    # MLP embeddings
    mlp_user_embeddings = mlp_model.get_layer('user_embedding').get_weights()
    mlp_item_embeddings = mlp_model.get_layer('item_embedding').get_weights()
    model.get_layer('mlp_embedding_user').set_weights(mlp_user_embeddings)
    model.get_layer('mlp_embedding_item').set_weights(mlp_item_embeddings)
    
    # MLP layers
    for i in range(1, num_layers):
        mlp_layer_weights = mlp_model.get_layer('layer%d' %i).get_weights()
        model.get_layer('layer%d' %i).set_weights(mlp_layer_weights)
        
    # Prediction weights
    gmf_prediction = gmf_model.get_layer('prediction').get_weights()
    mlp_prediction = mlp_model.get_layer('prediction').get_weights()
    new_weights = np.concatenate((gmf_prediction[0], mlp_prediction[0]), axis=0)
    new_b = gmf_prediction[1] + mlp_prediction[1]
    model.get_layer('prediction').set_weights([0.5*new_weights, 0.5*new_b])    
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
    start = time()
    args = parse_args()
    num_epochs = args.epochs
    batch_size = args.batch_size
    mf_dim = args.num_factors
    layers = eval(args.layers)
    reg_mf = args.reg_mf
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learning_rate = args.lr
    learner = args.learner
    verbose = args.verbose
    mf_pretrain = args.mf_pretrain
    mlp_pretrain = args.mlp_pretrain
            
    topK = 10
    evaluation_threads = 1#mp.cpu_count()
    #print("NeuMF arguments: %s " %(args))
    model_out_file = 'Pretrain/%s_NeuMF_%d_%s_%d.h5' %(args.dataset, mf_dim, args.layers, time())

    if args.dataset == "ppi":
        max_fold = 10
    else:
        max_fold = 10

    rep_auroc = []
    rep_auprc = []
    for repeat in range(1):
        # Loading data
        auroc_list = []
        auprc_list = []
        y_true_list = []
        y_score_list = []
        for fold in range(1,max_fold+1):
            t1 = time()
            foldPath = args.dataset + "/fold-"+str(fold)
            dataset = Dataset(args.path + foldPath)
            train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings[0], dataset.testNegatives
            validRatings, validNegatives = dataset.validRatings[0], dataset.validNegatives
            num_users, num_items = train.shape
            print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
                  %(time()-t1, num_users, num_items, train.nnz, len(testRatings)))
            
            # Build model
            model = get_model(num_users, num_items, mf_dim, layers, reg_layers, reg_mf)
            if learner.lower() == "adagrad": 
                model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
            elif learner.lower() == "rmsprop":
                model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
            elif learner.lower() == "adam":
                model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
            else:
                model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')
            
            # Load pretrain model
            if mf_pretrain != '' and mlp_pretrain != '':
                mf_pretrain_path = "Pretrain/" + args.dataset + mf_pretrain
                mlp_pretrain_path = "Pretrain/" + args.dataset + mlp_pretrain

                gmf_model = GMF.get_model(num_users,num_items,mf_dim)
                gmf_model.load_weights(mf_pretrain_path)
                mlp_model = MLP.get_model(num_users,num_items, layers, reg_layers)
                mlp_model.load_weights(mlp_pretrain_path)
                model = load_pretrain_model(model, gmf_model, mlp_model, len(layers))
                #print("Load pretrained GMF (%s) and MLP (%s) models done. " %(mf_pretrain_path, mlp_pretrain_path))
                
            # Init performance
            (auroc, auprc, y_true, y_score) = evaluate_model(model, validRatings, validNegatives, topK, evaluation_threads)
            #hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
            #print('Init: AUROC = %.4f, AUPRC = %.4f' % (auroc, auprc))
            best_auc, best_prc, best_iter = auroc, auprc, -1            
            if args.out > 0:
                model.save_weights(model_out_file, overwrite=True) 
                
            # Training model
            for epoch in range(num_epochs):
                t1 = time()
                # Generate training instances
                user_input, item_input, labels = get_train_instances(train, num_negatives)
                
                # Training
                hist = model.fit([np.array(user_input), np.array(item_input)], #input
                                 np.array(labels), # labels 
                                 batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
                t2 = time()
                
                # Evaluation
                (auroc, auprc, y_true, y_score) = evaluate_model(model, validRatings, validNegatives, topK, evaluation_threads)
                #hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
                loss = hist.history['loss'][0]
                if epoch % verbose == 0:
                    print('Repeat %d Fold %d Iteration %d [%.1f s]: AUROC = %.4f, AUPRC = %.4f, loss = %.4f [%.1f s]' 
                         % (repeat, fold, epoch,  t2-t1, auroc, auprc, loss, time()-t2))
                if args.metric == "auprc":
                    if auprc > best_prc:
                        best_auc, best_prc, best_iter = auroc, auprc, epoch
                        if args.out > 0:
                            model.save_weights(model_out_file, overwrite=True)
                else:
                    if auroc > best_auc:
                        best_auc, best_prc, best_iter = auroc, auprc, epoch
                        if args.out > 0:
                            model.save_weights(model_out_file, overwrite=True) 

            #print("End. Best Iteration %d:  AUROC = %.4f, AUPRC = %.4f. " %(best_iter, best_auc, best_prc))
            #if args.out > 0:
            #    print("The best NeuMF model is saved to %s" %(model_out_file))

            model.load_weights(model_out_file)
            (auroc_test, auprc_test, y_true, y_score) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
            #print("Test:  AUROC = %.4f, AUPRC = %.4f. " %(auroc_test, auprc_test))
            auroc_list.append(auroc_test)
            auprc_list.append(auprc_test)
            y_true_list.extend(y_true)
            y_score_list.extend(y_score)

        #pdb.set_trace()
        if args.dataset == "ppi" or args.dataset == "hpi":
            auroc_avg = roc_auc_score(y_true_list, y_score_list)
            precision, recall, thresholds = precision_recall_curve(y_true_list, y_score_list)
            auprc_avg = metrics.auc(recall, precision)
            print("Avergage:  AUROC = %.4f, AUPRC = %.4f. " %(auroc_avg, auprc_avg))
            rep_auroc.append(auroc_avg)
            rep_auprc.append(auprc_avg)
        else:
            rep_auroc.append(np.array(auroc_list).mean())
            rep_auprc.append(np.array(auprc_list).mean())

    auroc_final, auprc_final = np.array(rep_auroc).mean(), np.array(rep_auprc).mean()
    print("Final: AUROC = %.4f, AUPRC = %.4f. " %(auroc_final, auprc_final))
    #csvfile = 'Result/nF_%d_layers_%s_batch_%d_lr_%d.csv' %(mf_dim, args.layers, batch_size, learning_rate)
    csvfile = 'Result/' + args.dataset + '_' + args.metric + '.csv'
    results = [mf_dim, args.layers, batch_size, learning_rate, auroc_final, auprc_final]
    with open(csvfile, "a", newline='') as fp:
        wr = csv.writer(fp, dialect='excel')
        wr.writerow(results)
    end = time()
    print(end - start)