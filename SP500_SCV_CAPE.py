# SP500 vs SCV, real and relative to CAPE
# S&P, Gold, Treasury, SC pre-1972, Inflation data from 
#           https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histretSP.html
#       SCV data since 1972 fromportfoliovizualizer.com
#       CAPE data from https://shillerdata.com/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statsmodels.regression.linear_model import OLS

user = os.environ.get('USER', os.environ.get('USERNAME'))
filepath = f"C:/Users/{user}/Downloads/returns_since_1928.csv"

columns = ['Year',
           'CPI',
           'SCV',
           'SP500',
           'Gold',
           '10_yr_T',
           'CAPE']

data = pd.read_csv(filepath, header=0, usecols=columns).sort_values('Year', ascending=True)

# Get real returns for each asset (annual return minus annual inflation)
data['SCV_real'] = data['SCV'] - data['CPI']
data['SP500_real'] = data['SP500'] - data['CPI']
data['Gold_real'] = data['Gold'] - data['CPI']
data['10_yr_T_real'] = data['10_yr_T'] - data['CPI']

# Calculate Future 10-year cumulative returns for each year (2015+ will be null)
def get_future_returns(s, years):
    l = []
    for i in s.index:
        if i > years - 1:
            y = 1
            r = 1
            while y < years:
                r *= 1 + s[i - y]
                y += 1
            l.append(r)
        else:
            l.append(np.NaN)
    return l

def get_future_cape(s, years):
    l=[]
    for i in s.index:
        if i > years - 1:
            cape = s[i - years]
            l.append(cape)
        else:
            l.append(np.NaN)
    return l

data['SCV_Future10'] = get_future_returns(data['SCV_real'], 10)
data['SP500_Future10'] = get_future_returns(data['SP500_real'], 10)
data['Gold_Future10'] = get_future_returns(data['Gold_real'], 10)
data['10_yr_T_Future10'] = get_future_returns(data['10_yr_T_real'], 10)
data['10_yr_CAPE'] = get_future_cape(data['CAPE'], 10)

data['SCV_Future7'] = get_future_returns(data['SCV_real'], 7)
data['SP500_Future7'] = get_future_returns(data['SP500_real'], 7)
data['Gold_Future7'] = get_future_returns(data['Gold_real'], 7)
data['10_yr_T_Future7'] = get_future_returns(data['10_yr_T_real'], 7)
data['7_yr_CAPE'] = get_future_cape(data['CAPE'], 7)

data['SCV_Future5'] = get_future_returns(data['SCV_real'], 5)
data['SP500_Future5'] = get_future_returns(data['SP500_real'], 5)
data['Gold_Future5'] = get_future_returns(data['Gold_real'], 5)
data['10_yr_T_Future5'] = get_future_returns(data['10_yr_T_real'], 5)
data['5_yr_CAPE'] = get_future_cape(data['CAPE'], 5)

data['SCV_Future3'] = get_future_returns(data['SCV_real'], 3)
data['SP500_Future3'] = get_future_returns(data['SP500_real'], 3)
data['Gold_Future3'] = get_future_returns(data['Gold_real'], 3)
data['10_yr_T_Future3'] = get_future_returns(data['10_yr_T_real'], 3)
data['3_yr_CAPE'] = get_future_cape(data['CAPE'], 3)

cape_30 = data[data['CAPE'] > 30][['Year','CAPE','SP500','SCV','Gold','10_yr_T']]

print(data[['SP500','SCV','Gold','10_yr_T']].corr())

# plots 1
sns.regplot(data=data, x='CAPE', y='SCV_Future10', color='red')
sns.regplot(data=data, x='CAPE', y='SP500_Future10', color='blue')
sns.regplot(data=data, x='CAPE', y='Gold_Future10', color='gold')
sns.regplot(data=data, x='CAPE', y='10_yr_T_Future10', color='green')
plt.ylabel('Asset Growth (1 = 100%)')
plt.title('10-year Future Total Returns')
plt.show()

fig, axs = plt.subplots(2, 2, figsize = (9,11))
sns.regplot(data=data, x='CAPE', y='SCV_Future10', ax=axs[0,0], color='red').set_title('SCV vs CAPE')
sns.regplot(data=data, x='CAPE', y='SP500_Future10', ax=axs[0,1], color='blue').set_title('SP500 vs CAPE')
sns.regplot(data=data, x='CAPE', y='Gold_Future10', ax=axs[1,0], color='gold').set_title('Gold vs CAPE')
sns.regplot(data=data, x='CAPE', y='10_yr_T_Future10', ax=axs[1,1], color='green').set_title('10yr vs CAPE')
fig.suptitle('10 Year Future')
plt.show()

fig, axs = plt.subplots(2, 2, figsize = (9,11))
sns.regplot(data=data, x='CAPE', y='SCV_Future7', ax=axs[0,0], color='red').set_title('SCV vs CAPE')
sns.regplot(data=data, x='CAPE', y='SP500_Future7', ax=axs[0,1], color='blue').set_title('SP500 vs CAPE')
sns.regplot(data=data, x='CAPE', y='Gold_Future7', ax=axs[1,0], color='gold').set_title('Gold vs CAPE')
sns.regplot(data=data, x='CAPE', y='10_yr_T_Future7', ax=axs[1,1], color='green').set_title('10yr vs CAPE')
fig.suptitle('7 Year Future')
plt.show()

fig, axs = plt.subplots(2, 2, figsize = (9,11))
sns.regplot(data=data, x='CAPE', y='SCV_Future5', ax=axs[0,0], color='red').set_title('SCV vs CAPE')
sns.regplot(data=data, x='CAPE', y='SP500_Future5', ax=axs[0,1], color='blue').set_title('SP500 vs CAPE')
sns.regplot(data=data, x='CAPE', y='Gold_Future5', ax=axs[1,0], color='gold').set_title('Gold vs CAPE')
sns.regplot(data=data, x='CAPE', y='10_yr_T_Future5', ax=axs[1,1], color='green').set_title('10yr vs CAPE')
fig.suptitle('5 Year Future')
plt.show()

fig, axs = plt.subplots(2, 2, figsize = (9,11))
sns.regplot(data=data, x='CAPE', y='SCV_Future3', ax=axs[0,0], color='red').set_title('SCV vs CAPE')
sns.regplot(data=data, x='CAPE', y='SP500_Future3', ax=axs[0,1], color='blue').set_title('SP500 vs CAPE')
sns.regplot(data=data, x='CAPE', y='Gold_Future3', ax=axs[1,0], color='gold').set_title('Gold vs CAPE')
sns.regplot(data=data, x='CAPE', y='10_yr_T_Future3', ax=axs[1,1], color='green').set_title('10yr vs CAPE')
fig.suptitle('3 Year Future')
plt.show()

# plots 2
#fig, ax = plt.subplots(2,2, figsize=(9,11))
#sns.regplot(data=data, x='CPI', y='Gold_real', color='gold', ax=ax[0,0]).set_title('Gold vs CPI')
#sns.regplot(data=data, x='CPI', y='SP500_real', color='blue', ax=ax[0,1]).set_title('SP500 vs CPI')
#sns.regplot(data=data, x='CPI', y='SCV_real', color='red', ax=ax[1,1]).set_title('SCV vs CPI')
#sns.regplot(data=data, x='CPI', y='10_yr_T_real', color='green', ax=ax[1,0]).set_title('10_yr_T vs CPI')
#fig.suptitle('Returns vs CPI')
#plt.show()

# plots 3

data_1972 = data[data['Year'] > 1971]

fig, ax = plt.subplots(2,2, figsize=(9,11))
sns.regplot(data=data_1972, x='CPI', y='Gold_real', color='gold', ax=ax[0,0]).set_title('Gold vs CPI')
sns.regplot(data=data_1972, x='CPI', y='SP500_real', color='blue', ax=ax[0,1]).set_title('SP500 vs CPI')
sns.regplot(data=data_1972, x='CPI', y='SCV_real', color='red', ax=ax[1,1]).set_title('SCV vs CPI')
sns.regplot(data=data_1972, x='CPI', y='10_yr_T_real', color='green', ax=ax[1,0]).set_title('10_yr_T vs CPI')
fig.suptitle('Returns vs CPI - post 1971')
plt.show()

# plots 4

sns.regplot(data=data, x='CAPE', y='CPI').set_title('CPI vs CAPE')
plt.show()


# Regressions 1
data['10_yr_CAPE_diff'] = data['10_yr_CAPE'] - data['CAPE']

for column in data[['SP500_Future10','SCV_Future10','Gold_Future10','10_yr_T_Future10']].columns:
    X = data[['10_yr_CAPE_diff', 'CAPE']].dropna()
    Y = data[column].dropna()

    model = OLS(Y,X)
    results = model.fit()
    print(f"{column}:")
    print(results.params)

fig, ax = plt.subplots(2,2)
sns.regplot(data=data, x='10_yr_CAPE_diff', y='SP500_Future10', ax=ax[0,0], color='blue')
sns.regplot(data=data, x='10_yr_CAPE_diff', y='SCV_Future10', ax=ax[0,1], color='red')
sns.regplot(data=data, x='10_yr_CAPE_diff', y='Gold_Future10', ax=ax[1,0], color='gold')
sns.regplot(data=data, x='10_yr_CAPE_diff', y='10_yr_T_Future10', ax=ax[1,1], color='green')
fig.suptitle('Changes in SP500 Valuation')
plt.show()


# Monte Carlo Ex

data['CAPE_perc'] = data['CAPE'].divide(data['CAPE'].cummax(axis=0))
our_cape_range = 0.75

SP500 = []
SCV = []
Gold = []
Ten_yr_T = []
CAPE_perc = []
for i in range(3000):
    sample = data[['SP500_Future10','SCV_Future10','Gold_Future10','10_yr_T_Future10', 'CAPE_perc']].sample(n=10, replace=True)
    sample_avg = sample.agg('mean').to_dict()
    SP500.append(sample_avg['SP500_Future10'])
    SCV.append(sample_avg['SCV_Future10'])
    Gold.append(sample_avg['Gold_Future10'])
    Ten_yr_T.append(sample_avg['10_yr_T_Future10'])
    CAPE_perc.append(sample_avg['CAPE_perc'])

data_samples_full = {'SP500': SP500, 'SCV': SCV, 'Gold': Gold, '10_yr_T': Ten_yr_T, 'CAPE_perc': CAPE_perc}
data_samples_full = pd.DataFrame.from_dict(data_samples_full)
data_samples_filtered = data_samples_full[data_samples_full['CAPE_perc'] >= our_cape_range]

fig, ax = plt.subplots(2,1, figsize=(8,8), sharey=True, sharex=True)
sns.kdeplot(data=data_samples_filtered, x='SP500', ax=ax[0]).set_title('Filtered for High CAPE')
sns.kdeplot(data=data_samples_filtered, x='SCV', ax=ax[0], color='red').set_title('Filtered for High CAPE')
sns.kdeplot(data=data_samples_filtered, x='Gold', ax=ax[0], color='gold').set_title('Filtered for High CAPE')
sns.kdeplot(data=data_samples_filtered, x='10_yr_T', ax=ax[0], color='green').set_title('Filtered for High CAPE')

sns.kdeplot(data=data_samples_full, x='SP500', ax=ax[1]).set_title('Unfiltered')
sns.kdeplot(data=data_samples_full, x='SCV', ax=ax[1], color='red').set_title('Unfiltered')
sns.kdeplot(data=data_samples_full, x='Gold', ax=ax[1], color='gold').set_title('Unfiltered')
sns.kdeplot(data=data_samples_full, x='10_yr_T', ax=ax[1], color='green').set_title('Unfiltered')
plt.show()

fig, ax = plt.subplots(2,4, figsize=(12,12), sharey=True)
sns.boxplot(data=data_samples_filtered, y='SP500', ax=ax[0,0]).set_title('Filtered for High CAPE')
sns.boxplot(data=data_samples_filtered, y='SCV', ax=ax[0,1], color='red').set_title('Filtered for High CAPE')
sns.boxplot(data=data_samples_filtered, y='Gold', ax=ax[0,2], color='gold').set_title('Filtered for High CAPE')
sns.boxplot(data=data_samples_filtered, y='10_yr_T', ax=ax[0,3], color='green').set_title('Filtered for High CAPE')

sns.boxplot(data=data_samples_full, y='SP500', ax=ax[1,0]).set_title('Unfiltered')
sns.boxplot(data=data_samples_full, y='SCV', ax=ax[1,1], color='red').set_title('Unfiltered')
sns.boxplot(data=data_samples_full, y='Gold', ax=ax[1,2], color='gold').set_title('Unfiltered')
sns.boxplot(data=data_samples_full, y='10_yr_T', ax=ax[1,3], color='green').set_title('Unfiltered')
plt.show()


# Monte Carlos 1

def monte_carlo_10(sp_perc, scv_perc, g_perc, ty_perc):
    l=[]
    for i in range(3000):
        sp_sample = data_samples_filtered['SP500'].sample()
        scv_sample = data_samples_filtered['SCV'].sample()
        g_sample = data_samples_filtered['Gold'].sample()
        ty_sample = data_samples_filtered['10_yr_T'].sample()

        l.append(sp_sample.iloc[0]*sp_perc + scv_sample.iloc[0]*scv_perc + g_sample.iloc[0]*g_perc + ty_sample.iloc[0]*ty_perc)
    return l

P1 = pd.Series(monte_carlo_10(0.25, 0.25, 0.25, 0.25))
P2 = pd.Series(monte_carlo_10(0.5, 0.5, 0, 0))
P3 = pd.Series(monte_carlo_10(0.3, 0.3, 0.2, 0.2))
P4 = pd.Series(monte_carlo_10(0.4, 0.25, 0.25, 0.1))

for p in [P1, P2, P3, P4]:
    print(f"{p}:")
    print(p.describe())

# Monte Carlos 2
P1 = pd.Series(monte_carlo_10(0, 1, 0, 0))
P2 = pd.Series(monte_carlo_10(0.25, 0.75, 0, 0))
P3 = pd.Series(monte_carlo_10(0.5, 0.5, 0, 0))
P4 = pd.Series(monte_carlo_10(0.75, 0.25, 0, 0))
P5 = pd.Series(monte_carlo_10(1, 0, 0, 0))

for p in [P1, P2, P3, P4, P5]:
    print(f"{p}:")
    print(p.describe())