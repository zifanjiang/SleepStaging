import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn import  metrics, model_selection
from xgboost.sklearn import XGBClassifier


matfile = "C:/Users/Shafaat/Dropbox/6254_proj/latest_features/slpdb_additional_features.mat"
data =spio.loadmat(matfile)
raw_y = pd.DataFrame(data['y'])
raw_X = pd.DataFrame(data['x'])
raw_Xy = [raw_y,raw_X]
result_Xy = pd.concat(raw_Xy,axis=1).dropna()
y = result_Xy.iloc[:,0]
X = result_Xy.iloc[:,1:]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.20, random_state=42)


params = {
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'max_depth': 3,
    'learning_rate': 1.0,  # the maximum depth of each tree
    'silent': 1.0,         # logging mode - quiet
    'n_estimators': 100    # no. of trees
}


model = XGBClassifier(**params).fit(X_train, y_train)

# use the model to make predictions with the test data
y_pred = model.predict(X_test)
# how did our model perform?
count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))
