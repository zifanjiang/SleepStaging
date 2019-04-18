# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 15:45:29 2019

@author: Shafaat
"""

import scipy.io as spio
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.model_selection import cross_val_score


matfile = "/Users/Cyrus/Documents/Classes/ECE 6254/Project/slpdb_additional_features.mat"
data =spio.loadmat(matfile)
raw_y = pd.DataFrame(data['y'])
raw_X = pd.DataFrame(data['x'])
raw_Xy = [raw_y,raw_X]
result_Xy = pd.concat(raw_Xy,axis=1).dropna()
y = result_Xy.iloc[:,0]
X = result_Xy.iloc[:,1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.20, random_state=42)
scaler = StandardScaler()  
scaler.fit(X_train)
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)


######################################################################

# creating odd list of K for KNN
neighbors = list(range(1,50, 2))
nfeat_list = [14,7,4]

# empty list that will hold cv sclenores
cv_scores = np.zeros(( len(nfeat_list),len(neighbors) ))




# perform 10 fold cross validations
### WOrking on using PCA for cross validation:
kfcv = KFold(n_splits=10, shuffle = True, random_state=6254)
for i in range(0,len(nfeat_list)):
    for train_idx, test_idx in kfcv.split(X_train, y_train):
        Xcv_train, Xcv_test = X_train[train_idx], X_train[test_idx]
        ycv_train, ycv_test = y_train[train_idx], y_train[test_idx]
        pca = PCA(n_components = nfeat_list[i])
        post_extract = pca.fit_transform(Xcv_train, ycv_train)
        for j in range(0, len(neighbors)):
            knn = KNeighborsClassifier(n_neighbors=neighbors[j])
            knn.fit(pca.transform(Xcv_train), ycv_train)
            accuracy = knn.predict(ycv_train)
    
########## KNN #####################

# Questions:
### How do we determine the number of features to selction
### we need to perform feature selection for each fold, so I don't think we can use
### the cross_val_score() method as we are below.

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print ("The optimal number of neighbors is %d" % optimal_k)

# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()


classifier = KNeighborsClassifier(n_neighbors=5)  
classifier.fit(X_train, y_train)  
y_pred = classifier.predict(X_test)  


print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  