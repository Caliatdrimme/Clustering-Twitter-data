# Clustering-Twitter-data

The dataset is from https://www.kaggle.com/c/twitter-sentiment-analysis2/data

The Python script turns the data into a Bag of Words format for processing

Then it fits the data by finding the best number of clusters between 2 and 5:
1) with a Gaussian Mixture model with the EM algorithm from scikit learn
2) with kmeans

Reports on the scores, and returns the cluster means (EM) and centres (kmeans)
Can use the .predict to compare clusterings 
