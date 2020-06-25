#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import DotProduct, WhiteKernel, ExpSineSquared, RationalQuadratic, RBF, Matern
from sklearn.model_selection import GridSearchCV
from plotstuff import choose_data, setup_plot, get_xlabels, get_xticks
from joblib import dump, load

#fit GPR model
def fit_gpr(df, col, range_max):
    #The following hyperparemeters were decided using a grid search cross-validation
    kernel = ExpSineSquared() + RBF() + Matern() + RationalQuadratic()
    if col == 'New Cases':
        alpha = 3
        normalize_y = True
    elif col == 'New Deaths':
        alpha = 0.1
        normalize_y = True
    #Fitting the regressor and making the prediction
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=10, normalize_y = normalize_y)
    X = choose_data(df, col)[0].reshape(-1, 1)
    y = choose_data(df, col)[1].reshape(-1, 1)
    gpr.fit(X, y)
    X_predict = np.array([i for i in range(range_max+1)]).reshape(-1, 1)
    plt_X_predict = np.linspace(0, range_max, range_max+1)
    plt_X = X.reshape(1, -1)[0]
    prediction = [n[0] for n in gpr.predict(X_predict)]
    #save the model so we can load it later
    if col=='New Cases':
        dump(gpr, 'cases.joblib') 
    elif col=='New Deaths':
        dump(gpr, 'deaths.joblib')
    #return the values as arrays, don't want negative values for deaths/cases so take absolute value of predictions
    return np.array(plt_X_predict), abs(np.array(prediction))

#sum the gpr to deal with total data instead of daily data
def get_gpr_sum(df, col,range_max):
    #load the saved model
    if col == 'Total Cases': 
        gpr = load('cases.joblib')
    elif col == 'Total Deaths':
        gpr = load('deaths.joblib')
    #sum the predictions from the model
    X_predict = np.array([i for i in range(range_max+1)]).reshape(-1, 1)
    predict = np.array([n[0] for n in gpr.predict(X_predict)])
    predict_total = np.array([sum(abs(predict[:i+1])) for i in range(range_max+1)]).reshape(-1,1)
    return X_predict, predict_total

#plots the gpr prediction as a lineplot and the actual data as a scatterplot
def plot_gpr(df, col, range_max, interval):
    plt.figure(figsize = (20,10))
    x, y = choose_data(df, col)
    xticks = get_xticks(range_max, interval)
    ax = plt.subplot()
    setup_plot(df, col, range_max)
    X_predict, prediction = fit_gpr(df, col, range_max)
    sns.scatterplot(x, y, data=df, label = f'Actual {col}', color = 'Black', s=100)
    sns.lineplot(X_predict, prediction, data=df, label = f'GPR Predicted {col}', color = 'red', alpha = 0.5, linewidth=5)
    ax.set_xticks(xticks)
    ax.set_xticklabels(get_xlabels(interval)[:range_max//interval + 1], fontsize = 12)
    plt.legend(loc = 'upper left', prop={'size': 15})
    plt.savefig(f'{col} GPR.png')
    plt.show()
    
#plots the sum of gpr as a lineplot and the actual total data as a scatterplot
def plot_gpr_sum(df, col,range_max, interval):
    plt.figure(figsize = (20, 10))
    x, y = choose_data(df, col)
    xticks = get_xticks(range_max, interval)
    ax = plt.subplot()
    setup_plot(df, col, range_max)
    X, predict_total = get_gpr_sum(df, col, range_max)
    sns.scatterplot(x, y, data=df, label = f'Actual {col}', color = 'Black', s=100)
    sns.lineplot(X.flatten(), predict_total.flatten(), data=df, label = f'GPR Predicted {col}', color = 'Red', alpha = 0.5, linewidth=5)
    ax.set_xticks(xticks)
    ax.set_xticklabels(get_xlabels(interval)[:range_max//interval + 1], fontsize = 12)
    plt.legend(loc = 'upper left', prop={'size': 15})
    #Label the expected value at the end of the prediction
    ax.text(len(X)-3, 0.9*int(predict_total[-1]), f'{int(predict_total[-1])}\n{col}\nby {get_xlabels(interval)[range_max//interval]}', fontweight = 'bold', bbox=dict(edgecolor='red', fill=False))
    plt.savefig(f'{col} GPR.png')
    plt.show()
    

