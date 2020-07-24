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

# summer 183 and winter 182
Village_Population = list(range(80,208,12))

folder = 'Results_'
Demand = pd.Series()  
data_1 = pd.DataFrame()
for i in Village_Population:
    data = pd.Series()

    for j in range(1,9):
        path_1 = folder + str(i) + '/s0' + str(j) + '_sum.csv' 
        path_2 = folder + str(i) + '/s0' + str(j) + '_win.csv' 
        Power_Data_1 = pd.read_csv(path_1,index_col=0)
        Power_Data_2 = pd.read_csv(path_2,index_col=0)
        
        mean_1 = Power_Data_1.mean().mean()
        mean_2 = Power_Data_2.mean().mean()
        mean = (mean_1 + mean_2)/2
        
        data.loc[j] = mean
    data_1[i] = data     
    Demand.loc[i] = data.mean() 
        
        

data_2 = data_1.transpose()
data_2.plot(linestyle='--', marker='o')        
        
        
        
        
        
            