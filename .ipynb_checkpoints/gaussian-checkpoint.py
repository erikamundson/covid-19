#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from plotstuff import choose_data, setup_plot, get_xlabels, get_xticks

#define a gaussian
def gaussian_func(x, a, x0, sigma):
    return a * np.exp(-(x-x0)**2/(2*sigma**2))


#get the parameters of a gaussian with given dataframe and column
def gaussian_fit(df, col):
    x, y = choose_data(df, col)
    initial_guess = [1, 40, 75]
    popt, pcov = curve_fit(gaussian_func, x, y, p0 = initial_guess, maxfev=5000)
    return popt, pcov

#CDF, Cumulative Distribution Function, is the integral of a Gaussian but for our purposes we just use a sum since the data is integer-based. get_cdf returns a numeric sum of sum_col, which should correspond to col
def get_cdf(df, col, sum_col, range_max):
    x_plot = np.linspace(0, range_max, range_max+1)
    total_predict = [gaussian_func(x_plot[i], *gaussian_fit(df, sum_col)[0]) for i in range(range_max+1)]
    total_predict_sum = [sum(total_predict[:i]) for i in range(range_max+1)]
    return total_predict_sum

#plots the best-fit gaussian function with given dataframe and column, range_max is the maximum x-axis value. Also plots actual data.
def plot_gaussian(df, col, range_max, interval):
    plt.figure(figsize = (20,10))
    x, y = choose_data(df, col)
    x_predict = np.linspace(0, range_max, 10*(range_max)+1)
    x_plot = np.linspace(0, range_max, range_max+1)
    ax = plt.subplot()
    setup_plot(df, col, range_max)
    xticks = get_xticks(range_max, interval)
    sns.scatterplot(x, y, data=df, label = f'Real {col}', color = 'Black', s=100)
    sns.lineplot(x_predict, gaussian_func(x_predict, *gaussian_fit(df, col)[0]), data=df, label = f'Curve Fit Gaussian Predicted {col}', color = 'Red', alpha = 0.5, linewidth=5)
    ax.set_xticks(xticks)
    ax.set_xticklabels(get_xlabels(interval)[:range_max//interval+1], fontsize=12)
    plt.legend(loc = 'upper left', prop={'size': 15})
    plt.savefig(f'{col} Gaussian.png')
    plt.show()
    
#plots the cdf function with given dataframe and column, sum_col being the column to sum to get col, range max is maximum x-axis value. Also plots actual total data.
def plot_cdf(df, col, sum_col, range_max, interval):
    plt.figure(figsize = (20,10))
    x, y = choose_data(df, col)
    x_plot = np.linspace(0, range_max, range_max+1)
    y_plot = get_cdf(df, col, sum_col, range_max)
    xticks = get_xticks(range_max, interval)
    ax = plt.subplot()
    setup_plot(df, col, range_max)
    sns.scatterplot(x, y, data = df, label = f'Real {col}', color = 'Black', s = 100)
    sns.lineplot(x_plot, y_plot, data = df, label = f'Curve Fit Gaussian Predicted {col}', color = 'Red', alpha = 0.5, linewidth = 5)
    ax.set_xticks(xticks)
    ax.set_xticklabels(get_xlabels(interval)[:range_max//interval + 1], fontsize=12)
    plt.legend(loc = 'upper left', prop={'size': 15})
    ax.text(len(x_plot)-3, 0.9*int(y_plot[-1]), f'{int(y_plot[-1])}\n{col}\nby {get_xlabels(interval)[range_max//interval]}', fontweight = 'bold', bbox=dict(edgecolor='red', fill=False))
    plt.savefig(f'{col} Gaussian.png')
    plt.show()