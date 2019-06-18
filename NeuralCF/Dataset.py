'''
Created on Aug 8, 2016
Processing datasets. 

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import scipy.sparse as sp
import numpy as np
import pdb
import re

class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.trainMatrix = self.load_rating_file_as_matrix(path + ".train.rating")
        self.validNegatives, validUserList = self.load_negative_file(path + ".valid.all")
        self.validRatings = self.load_rating_file_as_list(path + ".valid.rating", validUserList)        
        self.testNegatives, testUserList = self.load_negative_file(path + ".test.all")
        self.testRatings = self.load_rating_file_as_list(path + ".test.rating", testUserList)        

        assert len(self.testRatings[0]) == len(self.testNegatives)
        assert len(self.validRatings[0]) == len(self.validNegatives)        
                        
        self.num_users, self.num_items = self.trainMatrix.shape        
        #self.allUnobserve = self.load_negative_file(path + ".all.unobserve", list(range(self.num_users)))
        #pdb.set_trace()
        #self.testNegativesPPI = self.load_negative_file_ppi()
    
    def load_rating_file_as_matrix_ppi(self):
        Y = pd.read_csv('Y.csv', header=None)
        print(Y.shape)
        mat = sp.dok_matrix(Y, dtype=np.float32)    
        return mat

    """
    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList
    """
    def load_rating_file_as_list(self, filename, testUserList):
        ratingList = []
        current_user = testUserList[0]
        items = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                while testUserList[0] < user: # the user with only negative test samples
                    #pdb.set_trace()
                    tmp_user = testUserList.pop(0) # remove the first user                    
                    ratingList.append([tmp_user, items]) 
                    items = []                    
                    current_user = tmp_user
                
                items.append(item)
                line = f.readline()

            ratingList.append([current_user, items])
            tmp_user = testUserList.pop(0)
            items = []  

            while len(testUserList) > 0: # the user with only negative test samples
                tmp_user = testUserList.pop(0) # remove the first user                    
                ratingList.append([tmp_user, items]) 

        return ratingList, testUserList
    
    def load_negative_file(self, filename):
        negativeList = []
        testUserList = []
        count = 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user = int(re.split(',|\(| |\)|', arr[0])[1])                
                if len(arr) > 1:
                    testUserList.append(user)
                    count = count + 1
                    negatives = []
                    for x in arr[1: ]:
                        negatives.append(int(x))
                    negativeList.append(negatives)
                line = f.readline()
        return negativeList, testUserList
    
    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
                line = f.readline()    
        return mat


