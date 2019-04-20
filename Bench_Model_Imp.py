#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 16:48:06 2019

@author: Cyrus
"""
from __future__ import division

import scipy.io as spio
import scipy.stats as stats
import math
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import cohen_kappa_score

matfile = "data/slpdb_additional_features.mat"
data = spio.loadmat(matfile)
nfolds = 10
seed = 42
feat_ns = [27, 14, 7, 4]
raw_y = data["y"]
raw_X = data["x"]
mask = ~np.any(np.isnan(raw_X), axis=1)
y = raw_y[mask]
X = raw_X[mask]
clfname = 'svm'  # knn,svm,MLR,xgboost

if clfname == 'knn':
    ########## KNN ###############
    neighbors = list(range(1, 50, 2))
    opt_k_list = -1 * np.ones((len(feat_ns), nfolds))
    acc_list_knn = -1 * np.ones((len(feat_ns), nfolds))
    auc_list_knn = -1 * np.ones((len(feat_ns), nfolds))
    kappa_list_knn = -1 * np.ones((len(feat_ns), nfolds))

    kfcv = KFold(nfolds, True, seed)
    fold_no = 0
    ##CV loop
    for maj_idx, min_idx in kfcv.split(X):
        X_maj, X_min = X[maj_idx], X[min_idx]
        y_maj, y_min = y[maj_idx], y[min_idx]
        # Split the major fold into validation and train sets

        ### Loop through values for # of features
        for n_idx in range(0, len(feat_ns)):
            X_train, X_valid, y_train, y_valid = train_test_split(X_maj, y_maj, test_size=1 / nfolds, random_state=seed)

            feat_selector = PCA(n_components=feat_ns[n_idx])
            feat_selector.fit(X_train, y_train)
            X_train = feat_selector.transform(X_train)
            X_valid = feat_selector.transform(X_valid)
            # X_min = feat_selector.transform(X_min)

            record_acc = -1
            record_k = -1
            record_model = None
            cand = -1
            for k in neighbors:
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train, y_train)
                cand = accuracy_score(y_valid, knn.predict(X_valid))
                if (cand > record_acc):
                    record_acc = cand
                    record_k = k
                    record_model = knn
            print("select k of %d, with accuracy %f" % (record_k, record_acc))
            opt_k_list[n_idx][fold_no] = record_k
            predicted = record_model.predict(feat_selector.transform(X_min))
            acc_list_knn[n_idx][fold_no] = accuracy_score(y_min, predicted)
            kappa_list_knn[n_idx][fold_no] = cohen_kappa_score(y_min, predicted)

        fold_no += 1
    print("mean performance for each dimensionality of feature space is ", np.mean(acc_list_knn, axis=1), ' std:'
          , np.std(acc_list_knn, axis=1))
    print("mode of opt_k for each dimensionality of feature space is  ", stats.mode(opt_k_list, axis=1).mode)
    print("mean cohen's Kappa for each dimensionality of feature space is ", np.mean(kappa_list_knn, axis=1))

if clfname == 'MLR':
    ############ Multinomial Logistic Regression ###################
    acc_list_mlr = -1 * np.ones((len(feat_ns), nfolds))
    kappa_list_mlr = -1 * np.ones((len(feat_ns), nfolds))

    kfcv = KFold(nfolds, True, seed)
    fold_no = 0
    ##CV loop
    for maj_idx, min_idx in kfcv.split(X):
        X_maj, X_min = X[maj_idx], X[min_idx]
        y_maj, y_min = y[maj_idx], y[min_idx]
        # Split the major fold into validation and train sets

        ### Loop through values for # of features
        for n_idx in range(0, len(feat_ns)):
            X_train, X_valid, y_train, y_valid = train_test_split(X_maj, y_maj, test_size=1 / nfolds, random_state=seed)

            feat_selector = PCA(n_components=feat_ns[n_idx])
            feat_selector.fit(X_train, y_train)
            X_train = feat_selector.transform(X_train)
            X_valid = feat_selector.transform(X_valid)

            mlr = linear_model.LogisticRegression(multi_class="multinomial", solver="newton-cg")
            mlr.fit(X_train, y_train)
            predicted = mlr.predict(feat_selector.transform(X_min))
            acc_list_mlr[n_idx][fold_no] = accuracy_score(y_min, predicted)
            kappa_list_mlr[n_idx][fold_no] = cohen_kappa_score(y_min, predicted)
            print(acc_list_mlr[n_idx][fold_no])
        #        record_acc = -1
        #        record_k = -1
        #        record_model = None
        #        cand = -1
        #        for k in neighbors:
        #            knn = KNeighborsClassifier(n_neighbors = k)
        #            knn.fit(X_train, y_train)
        #            cand = accuracy_score(y_valid, knn.predict(X_valid))
        #            if (cand > record_acc):
        #                record_acc = cand
        #                record_k = k
        #                record_model = knn
        #        print("select k of %d, with accuracy %f" % (record_k, record_acc))
        #        opt_k_list[n_idx][fold_no] = record_k
        #        acc_list_knn[n_idx][fold_no] = accuracy_score(y_min,record_model.predict(feat_selector.transform(X_min)))
        fold_no += 1
        print("new fold")
    print("mean performance for each dimensionality of feature space is ", np.mean(acc_list_mlr, axis=1))
    print("mean cohen's Kappa for each dimensionality of feature space is ", np.mean(kappa_list_mlr, axis=1))

if clfname == 'svm':
    ############ SVM ###################
    acc_list_svc = -1 * np.ones((len(feat_ns), nfolds))
    opt_c_list = -1 * np.ones((len(feat_ns), nfolds))
    kappa_list_svc = -1 * np.ones((len(feat_ns), nfolds))

    kfcv = KFold(nfolds, True, seed)
    fold_no = 0
    Cs = np.logspace(0, 2, 10)
    Cs = np.logspace(0, 0, 1)
    ##CV loop
    for maj_idx, min_idx in kfcv.split(X):
        X_maj, X_min = X[maj_idx], X[min_idx]
        y_maj, y_min = y[maj_idx], y[min_idx]
        # Split the major fold into validation and train sets

        ### Loop through values for # of features
        for n_idx in range(0, len(feat_ns)):
            X_train, X_valid, y_train, y_valid = train_test_split(X_maj, y_maj, test_size=1 / nfolds, random_state=seed)

            feat_selector = PCA(n_components=feat_ns[n_idx])
            feat_selector.fit(X_train, y_train)
            X_train = feat_selector.transform(X_train)
            X_valid = feat_selector.transform(X_valid)

            record_acc = -1
            record_c = -1
            record_model = None
            cand = -1
            for c in Cs:
                clf = SVC(C=c, kernel='rbf')
                clf.fit(X_train, y_train)
                cand = accuracy_score(y_valid, clf.predict(X_valid))
                if (cand > record_acc):
                    record_acc = cand
                    record_c = c
                    record_model = clf
            print("select c of %f, with accuracy %f" % (record_c, record_acc))
            opt_c_list[n_idx][fold_no] = record_c
            predicted = record_model.predict(feat_selector.transform(X_min))
            acc_list_svc[n_idx][fold_no] = accuracy_score(y_min, predicted)
            kappa_list_svc[n_idx][fold_no] = cohen_kappa_score(y_min, predicted)
        fold_no += 1
        print("new fold")
    print("mean performance for each dimensionality of feature space is ", np.mean(acc_list_svc, axis=1))
    print("mode of opt_c for each dimensionality of feature space is  ", stats.mode(opt_c_list, axis=1).mode)
    print("mean cohen's Kappa for each dimensionality of feature space is ", np.mean(kappa_list_svc, axis=1))

if clfname == 'xgboost':
    ################### XgBoost ############################
    acc_list_xgb = [-1] * nfolds
    kappa_list_xgb = [-1] * nfolds

    params = {
        'objective': 'multi:softprob',  # error evaluation for multiclass training
        'max_depth': 3,
        'learning_rate': 1.0,  # the maximum depth of each tree
        'silent': 1.0,  # logging mode - quiet
        'n_estimators': 100  # no. of trees
    }

    kfcv = KFold(nfolds, True, seed)
    fold_no = 0
    ##CV loop
    for maj_idx, min_idx in kfcv.split(X):
        X_maj, X_min = X[maj_idx], X[min_idx]
        y_maj, y_min = y[maj_idx], y[min_idx]

        X_train, X_valid, y_train, y_valid = train_test_split(X_maj, y_maj, test_size=1 / nfolds, random_state=seed)
        # feat_selector = PCA(n_components=feat_ns[n_idx])
        # feat_selector.fit(X_train, y_train)
        # X_train = feat_selector.transform(X_train)
        # X_valid = feat_selector.transform(X_valid)
        params = {
            'objective': 'multi:softprob',  # error evaluation for multiclass training
            'max_depth': 3,
            'learning_rate': 1.0,  # the maximum depth of each tree
            'silent': 1,  # logging mode - quiet
            'n_estimators': 100  # no. of trees
        }

        xgb = XGBClassifier(**params).fit(X_train, y_train)

        predicted = xgb.predict(X_min)
        acc_list_xgb[fold_no] = accuracy_score(y_min, xgb.predict(X_min))
        kappa_list_xgb[fold_no] = cohen_kappa_score(y_min, predicted)
        print(acc_list_xgb[fold_no])
        fold_no += 1
        print("new fold")
    print("mean performance for each dimensionality of feature space is ", np.mean(acc_list_xgb, axis=0))
    print("mean cohen's Kappa for each dimensionality of feature space is ", np.mean(kappa_list_xgb, axis=0))
