#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 16:14:20 2019

@author: balderrama
"""

import pandas as pd
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

#%%
# Data manipulation
data = pd.read_excel('Data_Base.xls', index_col=0, Header=None)   
#data = data.loc[data['Gap']< 5]


y = data['Battery Capacity']
y=y.astype('int')

X = pd.DataFrame()
X['Renewable Invesment Cost'] = data['Renewable Invesment Cost']   
X['Battery Invesment Cost'] = data['Battery Invesment Cost']
X['Deep of Discharge'] = data['Deep of Discharge']
X['Battery Cycles'] = data['Battery Cycles']
X['Battery Invesment Cost'] = data['Battery Invesment Cost'] 
X['Generator Efficiency'] = data['Generator Efficiency']
X['Low Heating Value'] = data['Low Heating Value']
X['Fuel Cost'] = data['Fuel Cost']
#X['Generator Nominal capacity'] = data['Generator Nominal capacity'] 
X['Max Demand'] = data['Max Demand']
X['Renewable Energy Mean'] = data['Renewable Energy Mean']



y, X = shuffle(y, X)
#min_max_scaler = preprocessing.MinMaxScaler() 
#X = min_max_scaler.fit_transform(X)

 

#%%
## Linear regression
lm = linear_model.LinearRegression(fit_intercept=True)
model = lm.fit(X,y)

y_predictions = lm.predict(X)
y_predictions_max = y_predictions.max()
y_predictions_nor = y_predictions/y_predictions_max

y_max = y.max()
y_nor = y/y_max

# y_true, y_pred
Score_Linear = r2_score(y,y_predictions)
MAE_linear =  mean_absolute_error(y_nor,y_predictions_nor)
RMSE_linear = mean_squared_error(y_nor,y_predictions_nor)

print('R^2 for linear regression is ' + str(Score_Linear*100) + ' %')
print('MAE for linear regression is ' + str(MAE_linear*100) + ' %')
print('RMSE for linear regression is ' + str(RMSE_linear*100) + ' %')

#%%
## Random forest



rf =  RandomForestRegressor(n_estimators=1000)
rf.fit(X,y)
y_rf = rf.predict(X)

Score_Linear = r2_score(y,y_rf)
print('R^2 for random forest is ' + str(Score_Linear*100) + ' %')



from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestfeatures = SelectKBest(score_func=chi2, k=6)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns,dfscores], axis=1)
featureScores.columns = ['Specs','Score']
print(featureScores.nlargest(10, 'Score')) 

#%%














#%%
# Cross validation


Scores = pd.DataFrame()
for i in [500,1000,1500]:
    print(i)
    rf =  RandomForestRegressor(n_estimators=i)
    scores = cross_val_score(rf, X, y, cv=5)
    Scores.loc[i,'Ramdon Forest'] = scores.mean()




lm = linear_model.LinearRegression(fit_intercept=True)
scores = cross_val_score(lm, X, y, cv=5)
print(scores.mean())



#%%
# Sylvain model

X_1 = pd.DataFrame()
X_1['NPC'] = data['NPC']
X_1['LCOE'] = data['LCOE']
X_1['Renewable Invesment Cost'] = data['Renewable Invesment Cost']   
X_1['Battery Invesment Cost'] = data['Battery Invesment Cost']
X_1['Deep of Discharge'] = data['Deep of Discharge']
X_1['Battery Cycles'] = data['Battery Cycles']
X_1['Battery Invesment Cost'] = data['Battery Invesment Cost'] 
X_1['Generator Efficiency'] = data['Generator Efficiency']
X_1['Low Heating Value'] = data['Low Heating Value']
X_1['Fuel Cost'] = data['Fuel Cost']
#X_1['Generator Nominal capacity'] = data['Generator Nominal capacity'] 
X_1['Max Demand'] = data['Max Demand']
X_1['Renewable Energy Mean'] = data['Renewable Energy Mean']

X_1 = round(X_1,2)
X_1.to_csv('Data_Gaussian.csv', index=False)