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

from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV



def vect(train, ngram=None):
   #takes in two arrays of strings
   #vectorizes both from the dictionary formed from the first array
   #returns both vectorized with the dictionary created
   
   #turn into bag of words representation
    if(ngram==None):
        vectorizer = CountVectorizer()
    else:
        vectorizer = CountVectorizer(ngram_range=ngram)
        
    
    X = vectorizer.fit_transform(train)

    nary = vectorizer.get_feature_names()
    res = X.toarray()

    return res, nary


def preprocess(file_train, column_text, column_tag, ngram=None):
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
    res, nary = vect(original,ngram)
    
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

   
   labels_em = best_model.predict(train)
   
   samples_em, labels_samples = best_model.sample(10)
   print(samples_em)
   print(labels_samples)
   
   for i in range(0, 10):
      samples_em[i] = numpy.absolute(samples_em[i])
      samples_em[i] = numpy.around(samples_em[i])
      
   print(samples_em)
   #samples_em = numpy.absolute(samples_em).astype('int32')
   
   return clusters, labels_em, samples_em

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

   
   labels_kmeans = best_model.predict(train)
   
 
   return clusters, labels_kmeans

def correct(real, models):
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
 
#Based on this tutorial from sklearn:
#https://scikit-learn.org/stable/auto_examples/neighbors/plot_digits_kde_sampling.html#sphx-glr-auto-examples-neighbors-plot-digits-kde-sampling-py
def sentence_generator(dataset, pca_divider = 4, total_samples = 10):
    
    data_size,feature_size = dataset.shape
    max_pca = min(data_size, feature_size)
    pca_components = min(feature_size//pca_divider, max_pca)
    pca = PCA(n_components=pca_components)
    #Find first 50 principle componenets
    dataset_pca = pca.fit_transform(dataset)

    #tune hyper-pramaters and perform density estimation
    hyper_parameters = {'bandwidth': numpy.logspace(-1, 1, 50)}
    grid = GridSearchCV(KernelDensity(), hyper_parameters, iid=True, cv=3)
    grid.fit(dataset_pca)
    density = grid.best_estimator_


    samples = density.sample(10)
    samples = pca.inverse_transform(samples)
    samples = numpy.absolute(numpy.around(samples)).astype('int32')
    return samples
   
def print_samples(samples, dictionary):
    total_samples,dimensions = samples.shape
    
    for sample in range(total_samples):
        for i in range(dimensions):
            if(samples[sample,i] != 0):
                print(dictionary[i], end = ' ')
        print()
   
#constants definitions
#only on labelled data for now
#file_test = "test.csv"
file_train = "train.csv"

#max number of clusters we want to find
m = 2

results = []

results_n = []

for i in range(100, 300, 100):
    #making the test set out of the original training set as it is labelled
    n_total = i + 100
    print("Using " + str(i) + " data samples\n")
    
    #preprocess the data
    train, dict_train, labels = preprocess(file_train, 2, 1)
    
    #print(train[0])
    
    em_res, em_labels, em_samples = EM(train,  m+1)
    
    #print(len(em_res[1]))
    #print(len(dict_train))
    
    kmeans_res, kmeans_labels = Kmeans(train, m+1)
    
    
    print("The training correctness of gmm is " + str(correct(labels, em_labels)) + "\n")
    print("The training correctness of kmeans is " + str(correct(labels, kmeans_labels)) + "\n")
    
    #REPORT MOST LIKELY WORDS OF EACH CLUSTER FROM THE CENTER REPRESENTATION INSTEAD
    #print("KMeans centres\n")
    #print_samples(kmeans_res, dict_train)
    
    #print("EM clusters\n")
    #print_samples(em_res, dict_train)
    
    results.append([correct(labels, em_labels),correct(labels, kmeans_labels)])
    
    print("EM samples\n")
    print_samples(em_samples, dict_train)
   
    
    print("\n\nsentences for PCA with 1/4 of original features size")
    samples = sentence_generator(train, pca_divider=4, total_samples = 10)
    print_samples(samples, dict_train)
    print("\n\nsentences for PCA with 1/2 of original features size")
    samples = sentence_generator(train, pca_divider=2, total_samples = 10)
    print_samples(samples, dict_train)
    
    print("Here n-gram of 3 is used instead of bag-of-words and we can see dependency between words such as subject etc. are captured\n")
    dataset,dictionary,labels = preprocess(file_train, 2, 1, ngram=(3,3))
    
    em_res, em_labels, em_samples = EM(dataset, m+1)
    
    kmeans_res, kmeans_labels = Kmeans(dataset, m+1)
        
    print("The training correctness of gmm is " + str(correct(labels, em_labels)) + "\n")
    print("The training correctness of kmeans is " + str(correct(labels, kmeans_labels)) + "\n")
    
    #REPORT MOST LIKELY WORDS OF THE CENTRE REPRESENTATIONS
    
    results_n.append([correct(labels, em_labels),correct(labels, kmeans_labels)])
    
    print("EM samples\n")
    print_samples(em_samples, dictionary)
   
    
    print("\n\nsentences for PCA with 1/4 of original features size")
    samples = sentence_generator(dataset, pca_divider=4, total_samples = 10)
    print_samples(samples, dictionary)
    print("\n\nsentences for PCA with 1/2 of original features size")
    samples = sentence_generator(dataset, pca_divider=2, total_samples = 10)
    print_samples(samples, dictionary)
    
    
    

print(results)
print(results_n)
#report results in nice graphics

    




