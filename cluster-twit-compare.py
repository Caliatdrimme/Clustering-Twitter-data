# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:57:56 2020

@author: sveta
"""

#load data file

#process twit data into BOW and dictionary

#fit with EM

#fit with k-means

#fit with SOM

#compare the models' fits

#graph results

import itertools

from sklearn import mixture

from csv import reader
from sklearn.feature_extraction.text import CountVectorizer

def preprocess(file, column):
   #takes in a file name (including path) of a csv file
   #and the column index of the column that contains the text of the tweets (0-based)
   #processes the tweets into bag of words representation for further processing
   #returns a vector of the processed tweets and a dictionary

   #read in the file as a list of lists
   with open(file, 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Pass reader object to list() to get a list of lists
    list_of_rows = list(csv_reader)
    
   #extract the tweet texts only
   #original is a list of all the tweets in original full-text form
   original = []
   #my computer runs out of memory if doing full training dataset at once
   #so only with 10k for now
   for i in range(1, 100):
      original.append(list_of_rows[i][column])
      
   #turn into bag of words representation
   vectorizer = CountVectorizer()
   
   X = vectorizer.fit_transform(original)

   nary = vectorizer.get_feature_names()
   
   res = X.toarray()
      
   return res, nary


def fit_EM(data):
   # fit a Gaussian Mixture Model with five components

   gmm = mixture.GaussianMixture(n_components=5, covariance_type='full').fit(data)
   
   print("the model was fit\n")
   return gmm

   
#constants definitions
file_test = "test.csv"
file_train = "train.csv"

#preprocess the training data
train, dict_train = preprocess(file_train, 2)

print(train[0])

model = fit_EM(train)

print(model.predict(train))


#print(dict_train)



#preprocess the test data
#share the dictionary between the train and test datasets??
#test, dict_test = preprocess(file_test, 2)




