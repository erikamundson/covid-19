#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

#Getting Case Data from COVID-19 Data Repository by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University
data_csv = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv")
columns_to_ignore = ['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Province_State', 'Country_Region', 'Lat', 'Long_', 'Combined_Key', 'Population']
cases = data_csv[[col for col in data_csv.columns if col not in columns_to_ignore]]
total_cases = [sum(cases[col]) for col in cases.columns[1:]]
daily_cases = [0]
for i in range(1,len(total_cases)):
    daily_cases.append(total_cases[i] - total_cases[i-1])
    
date = cases.columns[1:]

#Getting Death Data
death_csv = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv')
deaths = death_csv[[col for col in death_csv.columns if col not in columns_to_ignore]]
total_deaths = [sum(deaths[col]) for col in deaths.columns[1:]]
daily_deaths = [0]
for i in range(1,len(total_deaths)):
    daily_deaths.append(total_deaths[i] - total_deaths[i-1])
    
#Make New DataFrame
df = pd.DataFrame()
df['Date'] = date[39:]
df['New Cases'] = daily_cases[39:]
df['Total Cases'] = total_cases[39:]
df['New Deaths'] = daily_deaths[39:]
df['Total Deaths'] = total_deaths[39:]