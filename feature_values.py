# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 09:46:58 2018

@author: varun
"""

condition = {'Ex' : 5, 'Gd': 4, 'TA':3,'Fa':2,'Po':1,'NA':0}

category= {
               'FireplaceQu' : condition,
               'KitchenQual' : condition,
               'GarageQual' : condition,
               'GarageCond' : condition,
               'PoolQC' : condition,
               'Street' : {'Pave': 2,'Grvl': 1, 'NA': 0},
               'Alley' : {'Pave': 2,'Grvl': 1, 'NA': 0},
               'LotShape': {'Reg':4, 'IR1':3, 'IR2':2,'IR3':1, 'NAN': 0},
               'LandContour':{'Lvl':4, 'Bnk': 3,'HLS':2, 'Low':1,'NA': 0},
               'Utilities' :{'AllPub':4 , 'NoSewr': 3, 'NoSeWa':2,'ELO': 1},
               'LandSlope': {'Gtl':3, 'Mod':2, 'Sev':1},
               'BsmtExposure': {'Gd':5, 'Av':4,'Mn':3,'No':2,'NA':1},
               'BsmtFinType1': { 'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'NA':0},
               'BsmtFinType2': { 'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'NA':0},
          }