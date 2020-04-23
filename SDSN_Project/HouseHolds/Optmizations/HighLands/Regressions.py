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
from sklearn import tree
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import GridSearchCV
import numpy as np
#%%
# Data manipulation
# Data manipulation
# Data manipulation
data = pd.read_excel('Data_Base_1.xls', index_col=0, Header=None)   
#data = data.loc[data['Gap']< 1]

y = pd.DataFrame()
target='NPC'
y[target] = data[target]

y=y.astype('float')

X = pd.DataFrame()
X['Renewable Invesment Cost'] = data['Renewable Unitary Invesment Cost']   
X['Battery Unitary Invesment Cost'] = data['Battery Unitary Invesment Cost']
X['Deep of Discharge'] = data['Deep of Discharge']
X['Battery Cycles'] = data['Battery Cycles']
X['GenSet Unitary Invesment Cost'] = data['GenSet Unitary Invesment Cost']
X['Generator Efficiency'] = data['Generator Efficiency']
X['Low Heating Value'] = data['Low Heating Value']
X['Fuel Cost'] = data['Fuel Cost']
X['Generator Nominal capacity'] = data['Generator Nominal capacity'] 
#X['HouseHolds'] = data['HouseHolds']
X['Renewable Energy Unit Total'] = data['Renewable Energy Unit Total']
X['Max Demand'] = data['Max Demand']
X['Y'] = data['Y']


feature_list = list(X.columns)
y, X = shuffle(y, X, random_state=10)

y=np.array(y)
y = y.ravel() 
 

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

Score_Random = r2_score(y,y_rf)
MAE_Random =  mean_absolute_error(y,y_rf)
print('R^2 for random forest is ' + str(Score_Random*100) + ' %')
print('MAE for linear regression is ' + str(MAE_Random*100) + ' %')



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
# Desicion trees
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X, y)
y_clf = clf.predict(X)
Score_tree = r2_score(y,y_clf)
print('R^2 for Desicion trees is ' + str(Score_tree*100) + ' %')
#.get_params()
#%%
# Gaussian process

kernel = RBF(100)
gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=10)
gp.fit(X, y)
y_gp = gp.predict(X)

Score_gp = r2_score(y,y_clf)
print('R^2 for Gaussian regression is ' + str(Score_gp*100) + ' %')


#%%
#### Cross validation


# Linear Cross validation
lm = linear_model.LinearRegression(fit_intercept=True)
scores = cross_val_score(lm, X, y, cv=5,scoring='neg_mean_absolute_error')
scores.mean()

# Ramdon Forest
Scores = pd.DataFrame()
for i in range(2,15):
    print(i)
    rf =  RandomForestRegressor(min_samples_leaf=i,n_estimators=10,random_state=1 )
    scores = cross_val_score(rf, X, y, cv=10)
    Scores.loc[i,'Ramdon Forest'] = scores.mean()

# Desicion trees
clf = tree.DecisionTreeRegressor()
scores = cross_val_score(clf, X, y, cv=10)
scores.mean()

# Gaussian process

gp = GaussianProcessRegressor(kernel=kernel)
scores = cross_val_score(gp, X, y, cv=2)
scores.mean()

#%%
# Sylvain model

X_1 = pd.DataFrame()
X_1['NPC'] = data['NPC']   
X_1['Renewable Invesment Cost'] = data['Renewable Unitary Invesment Cost']   
X_1['Battery Unitary Invesment Cost'] = data['Battery Unitary Invesment Cost']
X_1['Deep of Discharge'] = data['Deep of Discharge']
X_1['Battery Cycles'] = data['Battery Cycles']
X_1['GenSet Unitary Invesment Cost'] = data['GenSet Unitary Invesment Cost']
X_1['Generator Efficiency'] = data['Generator Efficiency']
X_1['Low Heating Value'] = data['Low Heating Value']
X_1['Fuel Cost'] = data['Fuel Cost']
X_1['Generator Nominal capacity'] = data['Generator Nominal capacity'] 
X_1['HouseHolds'] = data['HouseHolds']
X_1['Renewable Energy Unit Total'] = data['Renewable Energy Unit Total']
X_1['Max Demand'] = data['Max Demand']
X_1 = round(X_1,2)
X_1.to_csv('Data_Gaussian.csv', index=False)



#%%
# Hyperparameter tunning
# Random forest
n_estimators = [10,100, 300, 500]
max_depth = [1,5, 15, 25, 30]
#min_samples_split = [2, 5, 10, 15, 100]
#min_samples_leaf = [1, 2, 5, 10] 

hyperF = dict(n_estimators = n_estimators, 
              max_depth = max_depth,  
#              min_samples_split = min_samples_split, 
#             min_samples_leaf = min_samples_leaf
                  )
forest = RandomForestRegressor()
gridF = GridSearchCV(forest, hyperF, cv = 5, verbose = 1, 
                      n_jobs = -1)

bestF = gridF.fit(X, y)
Best_Par = bestF.best_params_
Best_index = bestF.best_index_
Best_Score = bestF.best_score_


print(Best_Par)
print(Best_index)
print(Best_Score)

#0.9216800545856902


rf =  RandomForestRegressor(max_depth= 25, 
#                            min_samples_leaf = 1, 
#                            min_samples_split = 2, 
                            n_estimators= 300,
                            random_state = 1)

scores = cross_val_score(rf, X, y, cv=5)
scores.mean()


# Desiscion Trees

max_depth = [5, 8, 15, 25, 30]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10] 
hyperF = dict(max_depth = max_depth,  
              min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf)
dt = tree.DecisionTreeRegressor(random_state = 1)

gridF = GridSearchCV(dt, hyperF, cv = 5, verbose = 1, 
                      n_jobs = -1)

bestF = gridF.fit(X, y)


clf = tree.DecisionTreeRegressor(max_depth=15, min_samples_leaf= 10, min_samples_split=2)
scores = cross_val_score(clf, X, y, cv=5)
scores.mean()


clf = tree.DecisionTreeRegressor()
scores = cross_val_score(clf, X, y, cv=5)
scores.mean()










