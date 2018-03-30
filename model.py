# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 11:34:17 2018

@author: varun
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import ensemble
from sklearn import preprocessing
from feature_values import category


def display_correlation(train_data_set,target):
    #correlation value of salePrice and each value 
    corr = train_data_set.corr()
    print (corr['SalePrice'].sort_values(ascending=False), '\n')
    
    #plot of each column with salesprice and check correllation
    for colm in train_data_set:
        plt.scatter(x=train_data_set[colm], y=target)
        plt.ylabel('Sale Price')
        plt.xlabel(colm)
        plt.show()        
    return

def remove_outliers(train_data_set):
    train_data_set = train_data_set[train_data_set['GrLivArea'] < 5000]
    return train_data_set

def pickle_write(filename, model):
    #Saving the classifier
    fileObject = open(filename,'wb') 
    pickle.dump(model, fileObject)
    fileObject.close()
    return

def plot_prediction(predictions,actual_values,title):
    plt.scatter(predictions, actual_values, alpha=.75,
                color='b') #alpha helps to show overlapping data
    plt.xlabel('Predicted Price')
    plt.ylabel('Actual Price')
    plt.title(title)
    plt.show()
    return

data = pd.read_csv('train1.csv')
print(data.shape)

data_set = data.select_dtypes(include={np.number})

categoricals = data.select_dtypes(exclude=[np.number])
label_enc = {}

for colm in categoricals:
    if pd.isnull(categoricals[colm]).values.any():
        categoricals[colm] = categoricals[colm].astype('category').cat.add_categories(['NAN'])
        #categoricals[colm][pd.isnull(categoricals[colm])] = 'NAN'
        categoricals.loc[pd.isnull(categoricals[colm]),colm] = 'NAN'
    if colm in category:
        categoricals[colm] = categoricals[colm].astype('category').apply(lambda x : category[colm][x] if x in (category[colm]) else 0)
        categoricals[colm] = categoricals[colm].astype('float').fillna(0)
        print(colm)
    else:
        #categoricals[colm] = categoricals[colm].astype('category')
        le = preprocessing.LabelEncoder()
        le.fit(categoricals[colm])
        categoricals[colm] = le.transform(categoricals[colm])
        label_enc[colm] = le


train_data_set = data_set.join(categoricals)
train_data_set = train_data_set.astype('float').fillna(0)
#Saving the label encoder
pickle_write("label_encoder.p", label_enc)

print(train_data_set.shape)

#normalizing the output data
target = np.log(data_set.SalePrice)

#display_correlation(train_data_set,target)
train_data_set = remove_outliers(train_data_set)

y_sample = train_data_set[train_data_set['Id'] == 1]

y = np.log(train_data_set.SalePrice)
X = train_data_set.drop(['SalePrice', 'Id'], axis=1)

pickle_write("column_names.p", list(X.columns))

X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, random_state=42, test_size=.10)

#Linear Regression Model
lr = linear_model.LinearRegression(copy_X=True, fit_intercept=True, normalize=False)

model = lr.fit(X_train, y_train)

print('Model score : ', model.score(X_test,y_test))
predictions = model.predict(X_test)
print ('LR RMSE is: \n', mean_squared_error(y_test, predictions))
pickle_write("LRclassifier.p", model) #Saving the classifier
plot_prediction(predictions,y_test,'Linear Regression Model')

#Gradient Boosting Model
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)
mse = mean_squared_error(y_test, clf.predict(X_test))
print("GB MSE: %.4f" % mse)
pickle_write("gradientBoosting.p", clf) 

actual_values = y_test
plot_prediction(clf.predict(X_test),y_test,'gradient boosting Model')