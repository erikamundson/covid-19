#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import math
from datetime import datetime, timedelta

#labels for plots, values are every <interval> days since Feb 15
def get_xlabels(interval):
    start = datetime(2020, 3, 1)
    every_int = [start + timedelta(interval*i) for i in range(1000)]
    every_int_str = [day.strftime('%b') +  ' ' + day.strftime('%d') for day in every_int]
    return every_int_str

#returns array to set xticks
def get_xticks(range_max, interval):
    return np.linspace(0, range_max, range_max//interval +1)

#returns two arrays, x is natural numbers up to the length of chosen column, col is the values in df of that column 
def choose_data(df, col):
    x = np.array([i for i in range(len(df[col]))])
    y = np.array(df[col])
    return x, y

#plots the actual data as barplot and seven day moving average as lineplot
def bar_line(df, col, interval):
    sns.set_style('darkgrid')
    plt.figure(figsize = (20,10))
    x, y = choose_data(df, col)
    #compute the seven day moving average
    seven_day_average = [0,0,0,0,0,0] + [np.sum(y[i-7:i])/7 for i in range(6, len(y))]
    ax = plt.subplot()
    #make the legend prettier 
    red_patch = mpatches.Patch(color='IndianRed', label=f'{col}')
    black_patch = mpatches.Patch(color='k', alpha=0.8, label='Seven Day Moving Average')
    sns.barplot(x, y, data=df, palette =  'Reds')
    sns.lineplot(x, seven_day_average, data=df, color = 'white', linewidth=8)
    sns.lineplot(x, seven_day_average, data=df, color = 'k', alpha=1, linewidth=5)
    ax.set_xticks(np.linspace(0, interval * math.ceil(len(x)/interval)+1, len(x)//interval +2))
    ax.set_xticklabels(get_xlabels(interval)[:len(x)//interval+2], fontsize=12)
    ax.set_title(f'{col} of COVID-19', fontsize=20, fontweight='bold')
    ax.set_xlabel('Date', fontsize=16)
    ax.set_ylabel(f'{col}', fontsize=16)
    plt.annotate('Data from JHU CSSE COVID-19 Data: https://github.com/CSSEGISandData/COVID-19', (0,0), (-80,-40), fontsize=10, 
             xycoords='axes fraction', textcoords='offset points', va='top')
    plt.legend(handles = [red_patch, black_patch], loc='upper left', prop={'size': 15})
    plt.savefig(f'{col} 7 day average.png')
    plt.show()
    
#sets up plot with background style, axes labels, title
def setup_plot(df, col, range_max):
    sns.set_style('darkgrid')
    plt.xlabel('Date', fontsize=16)
    plt.ylabel(f'{col}', fontsize=16)
    plt.title(f'{col} of COVID-19', fontsize=20, fontweight='bold')
    #Cite the data and Model
    plt.annotate('Data from JHU CSSE COVID-19 Data: https://github.com/CSSEGISandData/COVID-19', (0,0), (-80,-40), fontsize=10, 
             xycoords='axes fraction', textcoords='offset points', va='top')