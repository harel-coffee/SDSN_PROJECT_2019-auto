#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 19:03:33 2019

@author: balderrama
"""

import pandas as pd
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz
import pydot
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import math
#%%
# Data manipulation
data = pd.read_excel('Data_Base_1.xls', index_col=0, Header=None)   
#data = data.loc[data['Gap']< 1]

y = pd.DataFrame()
target='Battery Capacity'
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
#X['Generator Nominal capacity'] = data['Generator Nominal capacity'] 
X['HouseHolds'] = data['HouseHolds']
X['Renewable Energy Unit Total'] = data['Renewable Energy Unit Total']
#X['Max Demand'] = data['Max Demand']
#X['Y'] = data['Y']


feature_list = list(X.columns)
y, X = shuffle(y, X, random_state=10)

y=np.array(y)
y = y.ravel() 
#%%

from sklearn.preprocessing import MinMaxScaler



#y = np.array(y)
#y = y.reshape(1, -1) 


y = MinMaxScaler().fit_transform(y)


X = MinMaxScaler().fit_transform(X)

#y = y.transpose()
X = pd.DataFrame(X, columns=feature_list)
y = pd.DataFrame(y, columns=[target])


#%%

#min_max_scaler = preprocessing.MinMaxScaler() 
#X = min_max_scaler.fit_transform(X)


n_estimators = [100, 300, 500]
max_depth = [ 15, 25, 30]
#min_samples_split = [2, 5, 10, 15, 100]
#min_samples_leaf = [ 5, 10,15] 
#max_features = [0.2,0.3,0.4]

hyperF = dict(n_estimators = n_estimators, 
              max_depth = max_depth,  
#              min_samples_split = min_samples_split, 
#             min_samples_leaf = min_samples_leaf,
#            max_features =max_feature
                  )

forest = RandomForestRegressor(random_state=10)
gridF = GridSearchCV(forest, hyperF, cv = 10, verbose = 1, 
                      n_jobs = -1,scoring='r2')

bestF = gridF.fit(X, y)
Best_Par = bestF.best_params_
Best_index = bestF.best_index_
Best_Score = bestF.best_score_
Results = bestF.cv_results_

print(Best_Par)
print(Best_index)
print(Best_Score)

#0.9216800545856902
#{'max_depth': 30, 'n_estimators': 500}
#%%
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint

forest = RandomForestRegressor(random_state=1)

param_dist = {'n_estimators':sp_randint(100, 500),
#              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(2, 11),
              "bootstrap": [True, False],
              "criterion": ["mse", "mae"],
              
              'min_samples_leaf': sp_randint(1,100),
              "max_depth":sp_randint(15, 30)}

n_iter_search = 300
random_search = RandomizedSearchCV(forest, param_distributions=param_dist,
                                   n_iter=n_iter_search, cv=5, iid=False,  n_jobs = -1,scoring='r2')


random_search.fit(X, y)

Best_Par = random_search.best_params_
Best_index = random_search.best_index_
Best_Score = random_search.best_score_


Results = random_search.cv_results_

print(round(Best_Score,3))
#%%
# Cross validation
rf =  RandomForestRegressor(max_depth= 15, 
#                            min_samples_leaf = 1, 
#                            min_samples_split = 2, 
                            n_estimators= 500,
                            random_state = 1)

method = 'r2'
Results = cross_validate(rf, X, y, cv=5, scoring=method)

score = Results['test_score']
score_mean = score.mean()


print(method + ' for the random forest is ' + str(score_mean))
#%%


# Pull out one tree from the forest
tree = rf.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')# Write graph to a png file
graph.write_png('tree.png')
#%%

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
rf =  RandomForestRegressor(max_depth= 15, 
#                            min_samples_leaf = 1, 
#                            min_samples_split = 2, 
                            n_estimators= 500,
                            random_state = 1)


rf = rf.fit(X_train, y_train)


y_test_1 = rf.predict(X_test)
MAE_Random =  round(mean_absolute_error(y_test,y_test_1),2)
Score_Random = round(r2_score(y_test,y_test_1),2)

print(MAE_Random)
print(Score_Random)
#%%



rf = rf.fit(X,y)
# Get numerical feature importances
importances = list(rf.feature_importances_)# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];












