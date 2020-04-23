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

import math

from sklearn import mixture
from sklearn.cluster import KMeans

from csv import reader
from sklearn.feature_extraction.text import CountVectorizer


def vect(train, test):
   #takes in two arrays of strings
   #vectorizes both from the dictionary formed from the first array
   #returns both vectorized with the dictionary created
   
   #turn into bag of words representation
   vectorizer = CountVectorizer()
   
   X = vectorizer.fit_transform(train)

   nary = vectorizer.get_feature_names()
   
   res = X.toarray()
   
   test_res = vectorizer.transform(test)
   
   return res, nary, test_res.toarray()

def preprocess(file_train, column, file_test, column_test):
   #takes in a file name (including path) of a csv file
   #and the column index of the column that contains the text of the tweets (0-based)
   #processes the tweets into bag of words representation for further processing
   #returns a vector of the processed tweets and a dictionary

   #TRAINING DATA
   #read in the file as a list of lists
   with open(file_train, 'r') as read_obj:
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
      
   #TEST DATA
   #read in the file as a list of lists
   with open(file_test, 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader_test = reader(read_obj)
    # Pass reader object to list() to get a list of lists
    list_of_rows_test = list(csv_reader_test)
    
   #extract the tweet texts only
   #original is a list of all the tweets in original full-text form
   original_test = []
   #my computer runs out of memory if doing full training dataset at once
   #so only with 10k for now
   for i in range(1, 50):
      original_test.append(list_of_rows_test[i][column_test])
      
   res, nary, test = vect(original,original_test)
         
   return res, nary, test


def fit_EM(data, n):
   # fit a Gaussian Mixture Model with five components

   gmm = mixture.GaussianMixture(n_components=n, covariance_type='full').fit(data)
   
   print("the model was fit\n")
   return gmm


def EM(train, test, m):
   
   #finds the best EM model and returns the resulting cluster weights
   
   clusters = []
   score = 0
   n = 0
   
   #find best EM model 2 to m clusters
   for i in range(2,m):
      model = fit_EM(train,i)
      #print(str(model.score(train)) + " the score of " + str(i) +" clusters \n")
      #print("current best score is " + str(score) + "\n")
      if model.score(train) > score:
         clusters = model.means_
         score = model.score(train)
         n = i
         best_model = model
      
   #print(model.predict(train))
   print(clusters)
   print(n)
   print(score)
   
   print("testing score for EM is " + str(best_model.score(test)))
   
   return clusters

def fit_kmeans(data, n):
   # fit a Gaussian Mixture Model with five components

   kmeans = KMeans(n_clusters=n, random_state=0).fit(data)
   
   #print("the model was fit\n")
   return kmeans

def Kmeans(train, test, m):
   
   #finds the best kmeans model and returns the resulting cluster weights
   
   clusters = []
   score = -10000000000
   n = 0
   
   #find best kmeans model 2 to m clusters
   for i in range(2,m):
      model = fit_kmeans(train,i)
      #print(str(model.score(train)) + " the score of " + str(i) +" clusters \n")
      #print("current best score is " + str(score) + "\n")
      if model.score(train) > score:
         clusters = model.cluster_centers_
         score = model.score(train)
         n = i
         best_model = model
      
   #print(model.predict(train))
   print(clusters)
   print(n)
   print(score)
   
   print("testing score for KMeans is " + str(best_model.score(test)))
   
   #best_model.predict(test)
   
   return clusters

   
#constants definitions
file_test = "test.csv"
file_train = "train.csv"

#max number of clusters we want to find
m = 5

#preprocess the data
train, dict_train, test = preprocess(file_train, 2, file_test, 1)

#print(train[0])

em_res = EM(train, test, m+1)

print(len(em_res[1]))
print(len(dict_train))

kmeans_res = Kmeans(train, test, m+1)

#report results in nice graphics




