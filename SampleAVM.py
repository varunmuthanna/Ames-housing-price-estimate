# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 11:31:01 2018

@author: varun
"""

import numpy as np
import pandas as pd
import pickle
import os

from feature_values import category

file = open('gradientBoosting.p','rb')
file1 = open('LRclassifier.p','rb')
file2 = open('label_encoder.p','rb')
file4 = open('column_names.p','rb')
classifier = pickle.load(file)
classifier1 = pickle.load(file1)
label_enc = pickle.load(file2)
column_name = pickle.load(file4)

data = pd.read_csv('test.csv')
print(data.shape)
X_input = pd.DataFrame(columns=column_name)

X_input = X_input.astype('float')
for colm in data:
    if pd.isnull(data[colm]).values.any():
        if colm in label_enc:
            X_input.loc[0,colm] = float(label_enc[colm].transform(['NAN'])[0])
        else:
            X_input.loc[0,colm] = 0.0 
    elif colm in category:
        X_input.loc[0,colm] = float(category[colm][data[colm].values[0]])
    elif colm in label_enc:
        X_input.loc[0,colm] = float(label_enc[colm].transform(data[colm])[0])
    else:
        X_input.loc[0,colm] = float(data[colm].values[0])

       
y_test = classifier.predict(X_input)

print(y_test)
prediction = y_test[0]
print(np.exp(prediction))

y_test = classifier1.predict(X_input)

print(y_test)
prediction = y_test[0]
print(np.exp(prediction))



#print("Enter the type of dwelling involved :")
#print("Possible values: \n\
#        20	1-STORY 1946 & NEWER ALL STYLES \n\
#        30	1-STORY 1945 & OLDER \n\
#        40	1-STORY W/FINISHED ATTIC ALL AGES \n\
#        45	1-1/2 STORY - UNFINISHED ALL AGES \n\
#        50	1-1/2 STORY FINISHED ALL AGES \n\
#        60	2-STORY 1946 & NEWER \n\
#        70	2-STORY 1945 & OLDER \n\
#        75	2-1/2 STORY ALL AGES \n\
#        80	SPLIT OR MULTI-LEVEL \n\
#        85	SPLIT FOYER \n\
#        90	DUPLEX - ALL STYLES AND AGES \n\
#       120	1-STORY PUD (Planned Unit Development) - 1946 & NEWER \n\
#       150	1-1/2 STORY PUD - ALL AGES \n\
#       160	2-STORY PUD - 1946 & NEWER \n\
#       180	PUD - MULTILEVEL - INCL SPLIT LEV/FOYER \n\
#       190	2 FAMILY CONVERSION - ALL STYLES AND AGES")
#
#MSSubClass = int(input())
#print(MSSubClass)
#
#print("Enter the general zoning classification of the sale")
#print("Possible Values are : \n\
#       A	Agriculture \n\
#       C	Commercial \n\
#       FV	Floating Village Residential \n\
#       I	Industrial \n\
#       RH	Residential High Density \n\
#       RL	Residential Low Density \n\
#       RP	Residential Low Density Park\n\
#       RM	Residential Medium Density")
#
#MSZoning = str(input())
#
#print("Enter the Linear feet of street connected to property")
#LotFrontage = int(input)
#
#print("Enter the Lot size in square feet")
#
#print("Enter the Type of road access to property")
#
#print("Enter the Type of alley access to property")
#
#print("Enter the General shape of property")
#
#print("Enter the Flatness of the property")
#
#print("Enter the Type of utilities available")
#
#print("Enter the Lot configuration")
#
#print("Enter the Slope of property")
#
#print("Enter the Physical locations within Ames city limits")
#
#print("Enter the Proximity to various conditions")
#
#print("Enter the Proximity to various conditions (if more than one is present)")
#
#print("Enter the Type of dwelling")
#
#print("Enter the Style of dwelling")
#
#print("Enter the Rates the overall material and finish of the house")
#
#print("Enter the  Rates the overall condition of the house")
#
#print("Enter the Original construction date")
#
#print("Enter the Remodel date (same as construction date if no remodeling or additions)")
#
#print("Enter the Type of roof")


