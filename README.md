# Clustering-Twitter-data

The dataset is from https://www.kaggle.com/c/twitter-sentiment-analysis2/data

The Python script turns the data into a Bag of Words format for processing

Fits different size of data in increments to two clusters by different algorithms:
1) GMM by EM
2) Kmeans

Returns the cluster means (EM)/centres (kmeans) and labels of data 

The two clusters are assumed to be like the sentiment that has real labels in the trainig part of the data
Uses first 100 tweets to figure out if the labels are the same or should be swapped
Calculates correct/total as a measure to compare algorithms

