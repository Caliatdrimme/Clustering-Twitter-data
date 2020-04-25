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
import numpy


def vect(train):
   #takes in two arrays of strings
   #vectorizes both from the dictionary formed from the first array
   #returns both vectorized with the dictionary created
   
   #turn into bag of words representation
   vectorizer = CountVectorizer()
   
   X = vectorizer.fit_transform(train)

   nary = vectorizer.get_feature_names()
   
   res = X.toarray()
   
   return res, nary

def preprocess(file_train, column_text, column_tag):
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
   #so only with 10k max for now
   for i in range(1, n_total+1):
      original.append(list_of_rows[i][column_text])
      
  #first half as train second half as test
   res, nary = vect(original)
   
   answers = []
   for i in range(1, n_total+1):
      answers.append(list_of_rows[i][column_tag])
         
   return res, nary, answers


def fit_EM(data, n):
   # fit a Gaussian Mixture Model with five components

   gmm = mixture.GaussianMixture(n_components=n, covariance_type='full').fit(data)
   
   #print("the model was fit\n")
   return gmm


def EM(train, m):
   
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
   #print(clusters)
   #print(n)
   #print(score)

   
   labels_kmeans = best_model.predict(train)
   
   return clusters, labels_kmeans

def fit_kmeans(data, n):
   # fit a Gaussian Mixture Model with five components

   kmeans = KMeans(n_clusters=n, random_state=0).fit(data)
   
   #print("the model was fit\n")
   return kmeans

def Kmeans(train, m):
   
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
   #print(clusters)
   #print(n)
   #print(score)

   
   labels_em = best_model.predict(train)
   
   return clusters, labels_em

def error(real, models):
    #use first 100 to max out the labels
    same = 0
    for i in range(0, 100):
        if models[i] == int(real[i]):
            same = same + 1
    
    #calculate correctness
    correct = 0       
    
    if same >= 50:    

        for i in range(100, n_total):
            if models[i] == 1:
                if int(real[i]) == 1:
                    correct = correct + 1
            if models[i] == 0:
                if int(real[i]) == 0:
                    correct = correct + 1
                    
    if same <50:
        
        for i in range(100, n_total):
            if models[i] == 1:
                if int(real[i]) == 0:
                    correct = correct + 1
            if models[i] == 0:
                if int(real[i]) == 1:
                    correct = correct + 1

    return correct/n_total
        
    
    #then count number of tweets classified correctly in the second half 
    
    
    
    return 0
   
#constants definitions
#only on labelled data for now
#file_test = "test.csv"
file_train = "train.csv"

#max number of clusters we want to find
m = 2

results = []

for i in range(100, 1000, 100):
    #making the test set out of the original training set as it is labelled
    n_total = i + 100
    print("Using " + str(i) + " data samples\n")
    
    #preprocess the data
    train, dict_train, labels = preprocess(file_train, 2, 1)
    
    #print(train[0])
    
    em_res, em_labels = EM(train,  m+1)
    
    #print(len(em_res[1]))
    #print(len(dict_train))
    
    kmeans_res, kmeans_labels = Kmeans(train, m+1)
    
    
    print("The training correctness of gmm is " + str(error(labels, em_labels)) + "\n")
    print("The training correctness of kmeans is " + str(error(labels, kmeans_labels)) + "\n")

    
    results.append([error(labels, em_labels),error(labels, kmeans_labels)])
    

print(results)
#report results in nice graphics




