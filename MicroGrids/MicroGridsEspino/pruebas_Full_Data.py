#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 00:36:24 2019
132
@author: balderrama
"""
import pandas as pd
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz
import matplotlib as mpl
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, ExpSineSquared, RationalQuadratic 
import numpy as np
from sklearn import linear_model
import time
from sklearn.model_selection import train_test_split
from joblib import dump
data = pd.read_excel('Data_Base.xls', index_col=0, Header=None)  
#data = data.loc[data['Renewable Capacity']>0]



y = pd.DataFrame()
target= 'Renewable Capacity' #  'Renewable Capacity' 'Renewable Penetration'
y[target] = data[target]

y=y.astype('float')

X = pd.DataFrame()
X['Renewable Invesment Cost'] = data['Renewable Unitary Invesment Cost']   
X['Battery Unitary Invesment Cost'] = data['Battery Unitary Invesment Cost']
#X['Deep of Discharge'] = data['Deep of Discharge']
#X['Battery Cycles'] = data['Battery Cycles']
X['GenSet Unitary Invesment Cost'] = data['GenSet Unitary Invesment Cost']
#X['Generator Efficiency'] = data['Generator Efficiency']
#X['Low Heating Value'] = data['Low Heating Value']
X['Fuel Cost'] = data['Fuel Cost']
#X['Generator Nominal capacity'] = data['Generator Nominal capacity'] 
X['HouseHolds'] = data['HouseHolds']
X['Renewable Energy Unit Total'] = data['Renewable Energy Unit Total']
#X['Max Demand'] = data['Max Demand']
#X['Y'] = data['Y']

feature_list = list(X.columns)
y, X = shuffle(y, X, random_state=10)

start = time.time()
l1 = [1,1,1,1,1,1]
l2 = [1,1,1,1,1,1]
#l3 = [1,1,1,1,1,1,1,1,1,1]

############################        NPC         ###############################    

#l1 = [1.12068469e+03, 2.90190449e+02, 3.22710731e-01, 3.64059780e+03,
#        1.67773906e+03, 1.10912276e-01, 2.56510249e+00, 5.50142341e-01,
#        1.38098175e+02, 1.38544239e+02]

#R^2 for the gaussian process with the train data set is 1.0
#R^2 for the gaussian process with the test data set is 1.0
#MAE for the gaussian process is 0.0
#The Regression took 12602.0 segundos
############################        LCOE         ##############################

#l1 = [4.20954674e+03, 4.56552818e+02, 8.75517313e-01, 7.91795832e+03,
#        6.00844163e+03, 1.31850735e-01, 6.94429376e+00, 5.01221988e-01,
#        1.68373020e+03, 1.34809061e+03]

#R^2 for the gaussian process with the train data set is 1.0
#R^2 for the gaussian process with the test data set is 0.9999999999999705
#MAE for the gaussian process is 0.0
#The Regression took 10268.0 segundos
############################ Renewable Capacity ###############################
#l1 = [1.38e+03, 324, 0.648, 7.69e+03, 1e+05, 0.134, 5.13, 0.635, 244, 194]
#l2 = [734, 6.99, 4.2e-05, 326, 0.691, 548, 5.22e+03, 0.00181, 55.4, 0.212]

#R^2 for the gaussian process with the train data set is 1.0
#R^2 for the gaussian process with the test data set is 1.0
#MAE for the gaussian process is 0.0
#The Regression took 27943.0 segundos

############################ Battery Capacity #################################
#l1 = [0.00206, 10.7, 2.78e+03, 221, 2.7e+03, 0.000578, 0.00329, 0.829, 0.000172, 589]
#l2 = [1.22e+03, 190, 0.414, 6.11e+03, 5.8e+03, 0.0939, 3.59, 0.441, 251, 233]

#R^2 for the gaussian process with the train data set is 1.0
#R^2 for the gaussian process with the test data set is 1.0
#MAE for the gaussian process is 0.0
#The Regression took 29099.0 segundos

#kernel =  (C()**2)*RBF(l1)
#kernel = Matern(l1)  +  Matern(l2) # +  Matern(l3)
kernel =  RBF(l1) #+ RBF(l2) #+ RBF(l3)
#kernel =  RBF(length_scale=l1,length_scale_bounds=(1e-5, 1e5)) #+ RBF(l2)


 
gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=3000,
                              optimizer = 'fmin_l_bfgs_b'
#                              , normalize_y=True
                              
                              )

gp = gp.fit(X, y)

R_2_train = round(gp.score(X,y), 4)

print('R^2 for the gaussian process with the train data set is ' + str(R_2_train))

R_2_test = gp.score(X, y) 

print('R^2 for the gaussian process with the test data set is ' + str(R_2_test))

y_gp = gp.predict(X)
MAE_Random =  round(mean_absolute_error(y,y_gp),2)

print('MAE for the gaussian process is ' + str(MAE_Random))

end = time.time()
print('The Regression took ' + str(round(end - start,0)) + ' segundos')    

# gp.kernel_.get_params()
start = time.time()

#l=[5.48045645e+02, 2.10972522e+02, 9.30673382e+02, 3.89146498e-01,
#        7.83716002e+01, 2.21077774e+02]


#%%

filename = target + '_Chaco.joblib'
dump(gp, filename) 






