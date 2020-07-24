#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 15:33:58 2019

@author: sergio
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as mtick
import matplotlib.pylab as pylab
from scipy.stats import pearsonr
from matplotlib.sankey import Sankey
import plotly.plotly as py
import pylab
import enlopy as el
import matplotlib as mpl
from pandas import ExcelWriter

#%%
Village_Population = list(range(50,570,50))


Villages = ['Raqaypampa']
Day_hour = pd.read_csv('Demand_join_minute.csv', index_col=0)   



for i in Villages:
    instance = pd.DataFrame(index=range(8760))
    path = i + '/Demand.xls'
    writer = ExcelWriter(path, engine='xlsxwriter')    
    for j in Village_Population:
        instance_2 = pd.DataFrame(index=range(8760))
        path_1 = i + '/' + str(j) +'/' + str(j)+ '_'+ i + '.csv' 
        sheet_name = 'village_' + str(j)
        Power_Data = pd.read_csv(path_1,index_col=0)
        Power_Data['Day'] = Day_hour['Day'] 
        Power_Data['hour'] = Day_hour['hour'] 
        foo = Power_Data.groupby(['Day','hour']).mean()
        instance[sheet_name] = foo.values
        instance_2[sheet_name] = foo.values 
        instance_2.index = range(1,8761)
        instance_2.columns = [1]
        instance_2.to_excel(writer, sheet_name=sheet_name)

    writer.save()


#%%        
Year = instance.sum()        
        
Year.plot()        
        
#%%

instance['hour'] = 0

foo = 0
iterations = 365

for i in range(iterations):
    for j in range(int(24)):
            
            index = instance.index[foo]
            instance.loc[index,'hour'] = j
            foo += 1
        
Demand_hourly = instance.groupby(['hour']).mean()

Demand_hourly.plot()






















        
            