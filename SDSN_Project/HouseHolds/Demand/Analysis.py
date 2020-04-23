#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 19:37:30 2019

@author: balderrama
"""


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
#%%
# Data load
data_1 = pd.read_csv('LI_1_household_El_Sena.csv', index_col=0,header=None)   
data_2 = pd.read_csv('HI_1_household_El_Sena.csv', index_col=0,header=None)
data_3 = pd.read_csv('LI_1_Raqaypampa.csv', index_col=0,header=0)   
data_4 = pd.read_csv('HI_1_Raqaypampa.csv', index_col=0,header=0)


demand = pd.DataFrame()

demand['Low_Income_Sena'] = data_1[1]
demand['High_Income_Sena'] = data_2[1]
demand['Low_Income_Raqaypampa'] = data_3['0']
demand['High_Income_Raqaypampa'] = data_4['0']

#%%
demand['Day'] = 0

foo = 0
iterations = int(len(demand)/1440)

for i in range(iterations):
    
    for j in range(1440):
        print(i,j)
        
        index = demand.index[foo]
        demand.loc[index,'Day'] = i
        foo += 1
        
        
demand['hour'] = 0

foo = 0

for i in range(iterations):
    print(i)
    for j in range(int(24)):
        for s in range(60):
            
            index = demand.index[foo]
            demand.loc[index,'hour'] = j
            foo += 1
        
Demand_hourly = demand.groupby(['Day','hour']).mean()

Demand_hourly.index = range(1,8761)
Demand_hourly.columns = [1,2,3,4]        

Demand_hourly.to_excel('Demand_Hourly.xls')        
demand.to_csv('Demand_join_minute.csv')
           
#%%       
        
minute = pd.read_csv('Demand_join_minute.csv', index_col=0,header=0)        
        
demand['Day'] = minute['Day']
demand['hour'] = minute['hour']

Demand_hourly = demand.groupby(['Day','hour']).mean()

sena = pd.DataFrame()

sena[1] = (Demand_hourly['Low_Income_Sena'] + Demand_hourly['High_Income_Sena'])/2
sena.index = range(1,8761)
sena.to_excel('Demand_sena.xls')













