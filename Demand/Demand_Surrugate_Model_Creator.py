#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 15:33:58 2019

@author: sergio
"""
import pandas as pd
import matplotlib.pylab as pylab
from sklearn import linear_model
from math import sqrt as sq
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
from joblib import dump
#%%
Village_Population = range(50,570,50)


Villages = ['Espino']
Day_hour = pd.read_csv('Demand_join_minute.csv', index_col=0)   

name = {'Sena':'Amazonia','Espino':'Chaco', 'Raqaypampa':'HighLands'}

for i in Villages:
    instance = pd.DataFrame(index=range(8760))

    for j in Village_Population:
        path_1 = i + '/' + str(j) +'/' + str(j)+ '_'+ i + '.csv' 
        Power_Data = pd.read_csv(path_1,index_col=0)
        Power_Data['Day'] = Day_hour['Day'] 
        Power_Data['hour'] = Day_hour['hour'] 
        foo = Power_Data.groupby(['Day','hour']).mean()
        instance[j] = foo.values

#%%        

y = pd.DataFrame()
y['Demand'] = instance.sum()/1000
y2 = instance.sum()/0.95

X = pd.DataFrame()
X['HouseHolds'] = instance.columns

#%%
scoring = 'r2'#'r2' 'neg_mean_absolute_error' # 'neg_mean_squared_error'

lm = linear_model.LinearRegression(fit_intercept=True)
scores = cross_val_score(lm, X, y, cv=5, scoring=scoring)
score = round(scores.mean(),2)

if scoring == 'neg_mean_squared_error':
    score = sq(-score)    
    print(scoring + ' for the linear regression with the test data set is ' + str(score))
else:    
    print(scoring + ' for the linear regression with the test data set is ' + str(score))



#%%

lm = linear_model.LinearRegression(fit_intercept=True)

lm = lm.fit(X,y)

Name =  name[Villages[0]]

filename = Villages[0] + '/demand_regression_'+ Name  + '.joblib'
dump(lm, filename) 





            