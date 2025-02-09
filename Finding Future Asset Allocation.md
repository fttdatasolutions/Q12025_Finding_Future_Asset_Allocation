## Introduction

The CAPE Ratio (Cyclically Adjusted Price to Earnings Ratio) is a valuation measurement, developed by Robert Shiller, that compares an asset’s current price to the average of the asset’s annualized earnings over the prior 10 years. Here is a visual for the formula:

 - CAPE = Price / (SUM(10 prior years of earnings) / 10 years)

Shiller’s purpose for developing, tracking, and analyzing this metric was/is to find a data-driven answer to a problematic question in finance: Does an asset’s price definitively impact the asset’s future returns? One should read Shiller’s papers and the deluge of subsequent academic studies, for further answers to that question. In this project, I will use Shiller’s data and build on his and others’ work to answer a different question: What asset allocation should I use, in the 1st quarter of 2025, to optimize my portfolio’s return over the next 10 years?

The project will answer this question with a Bayesian approach, meaning:

    1. The answer will be a probability distribution.
    2. New data will update the answer.

The project will not use strict Bayesian modeling, but will use Bayesian principles as a foundational guide. The project will use Python as a scripting language and will use source data from Robert Shiller’s data repository, Aswath Damodoran’s data repository, and portfoliovizualizer.com (all of whose citations for their data sources are on each respective website). The data covers annual returns, CAPE valuation at end-of-year, and annual inflation from the beginning of 1928 through the end of 2024:



```python
# S&P, Gold, Treasury, SC pre-1972, Inflation data from https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histretSP.html
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
```

The project will focus on 4 asset classes:

    1. S&P 500, as representing US equities, tilted towards large-cap growth (abbr. SP500)
    2. US small-cap value equities (abbr. SCV)
    3. Gold
    4. 10-year US Treasury notes (abbr. 10_yr_T)

I chose these asset classes, due to ease of access to data, ease of access for purchasing by retail investors, and the large disparity in correlation between their respective annual returns:


```python
# Get real returns for each asset (annual return minus annual inflation)
data['SCV_real'] = data['SCV'] - data['CPI']
data['SP500_real'] = data['SP500'] - data['CPI']
data['Gold_real'] = data['Gold'] - data['CPI']
data['10_yr_T_real'] = data['10_yr_T'] - data['CPI']

# Calculate Future 10-year cumulative returns for each year (2015+ will be null for 10-year, 2018+ null for 7-year, and so forth)
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

# Get asset correlations
print(data[['SP500','SCV','Gold','10_yr_T']].corr())
```

In the above correlation matrix, a result of:
 - 1 means that the assets are 100% correlated, meaning they have identical returns each year. Thus, each asset is 100% correlated with itself,  above.
 - 0 means that the 2 assets are 0% correlated, meaning one asset’s return in no way predicts the return of the other asset.
 - $-$1 means that the assets’ returns will be completely opposite.

We see that SP500 and SCV are almost completely uncorrelated with Gold and 10_yr_T, Gold and 1_yr_T are almost completely uncorrelated with each other, and SP500 is quite positively correlated with SCV, but far from 100% (this makes sense, as they are both US equity classes, but typically have different economic sector distributions). In future formulas, I will refer to these correlations as simply “C.”

It is important to note that the CAPE ratio we are using is the CAPE ratio for SP500. This is intentional, for two reasons:
 
    1. SP500 is a popular choice, and one in which I currently have a high percentage of my portfolio.
    2. SP500 currently has a relatively high CAPE ratio (37.77 at time of writing).

Given that Shiller posited and demonstrated that high CAPE ratios can reduce an asset’s future return prospects, reason demands that I look for alternatives or additions to SP500 in my portfolio, in order to achieve a better return over the next 10 years. Thus, I will refine the project’s aim: to measure subsequent distribution of possible 10-year performances of our 4 asset classes, given correlations “C”, the average return of each asset class, and the data provided above.
	Speaking of average returns, here are the summary statistics for each asset class:


```python
print(data['SP500'].describe()
print(data['SCV'].describe()
print(data['Gold'].describe()
print(data['10_yr_T'].describe()
```

We can infer from these summary statistics that all classes have a total and average positive return. Equities (SP500 and SCV) have the highest average and high volatility, Gold is very streaky, and 10_yr_T is usually slow and steady.

## Visualizing our Assets vs CAPE and CPI

Now, let us return to our primary question: for a given SP500 CAPE ratio, correlation “C,” and our data, what return distributions can we expect for each asset class over the next 10 years? We will begin by visualizing previous returns, vs SP500 CAPE ratios, following these steps:

    1. Calculate real returns, meaning returns minus inflation. All future return calculations and visualizations will use real returns, unless specified that the return is “nominal.”
    2. Calculate forward 10-year earnings and forward 10-year CAPE for each year (2015+ will be null, since the end of 2014 was 10 years ago).
    3. We will also calculate forward 3-year, 5-year, and 7-year returns and CAPEs.
    4. Plot linear regression charts for each asset class vs SP500 CAPE. This will rearrange the data to show returns relative to CAPE, in ascending CAPE order on the x axis; it is a scatter plot of the data that includes the regression’s slope and confidence interval.

We will follow this color schema, throughout the project, for clarity:
 - SP500 = blue
 - SCV = red
 - Gold = gold
 - 10_yr_T = green


```python
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
```

In “10-year Future Total Returns,” we see stark downward slopes for SP500 and SCV, indicating that as SP500 CAPE trends higher future returns trend lower. The slope for 10_yr_T is almost flat, indicating no CAPE effect. Gold has a significant positive slope, indicating that future returns trend higher, as CAPE trends higher. “10 year Future” breaks out each asset class into its own chart. “7 Year Future” shows similar results. In “5 Year Future,” we see that the slope for 10_yr_T increased, but the slope of Gold decreased. In “3 year Future,” Gold and 10_yr_T are flat, indicating no effect from CAPE.

We can solidly decide that SP500 CAPE has minimal to no effect on 10_yr_T returns. CAPE also appears to have no near-term effect on Gold, but longer-term Gold slopes are distinctly positive. Since Gold is a physical commodity, I expect Gold to perform better when inflation is high (measured by Consumer Price Index - CPI) . Let’s check:


```python
# plots 2
fig, ax = plt.subplots(2,2, figsize=(9,11))
sns.regplot(data=data, x='CPI', y='Gold_real', color='gold', ax=ax[0,0]).set_title('Gold vs CPI')
sns.regplot(data=data, x='CPI', y='SP500_real', color='blue', ax=ax[0,1]).set_title('SP500 vs CPI')
sns.regplot(data=data, x='CPI', y='SCV_real', color='red', ax=ax[1,1]).set_title('SCV vs CPI')
sns.regplot(data=data, x='CPI', y='10_yr_T_real', color='green', ax=ax[1,0]).set_title('10_yr_T vs CPI')
fig.suptitle('Returns vs CPI')
plt.show()
```

Well, the regression slope seems to show Gold returns relative to inflation as flat (at least, that conclusion is well within the margin of error). Equities are downward sloping to flat. 10_yr_T is heavily impacted by inflation!

One interesting problem with our analysis is that the price of gold was temporarily fixed at $35/oz by the Bretton-Woods agreement, which lasted from 1944 to 1971. That’s a large portion of our dataset, with a confounding variable for gold. Let’s redo the above test with data 1972 to present:


```python
# plots 3
data_1972 = data[data['Year'] > 1971]

fig, ax = plt.subplots(2,2, figsize=(9,11))
sns.regplot(data=data_1972, x='CPI', y='Gold_real', color='gold', ax=ax[0,0]).set_title('Gold vs CPI')
sns.regplot(data=data_1972, x='CPI', y='SP500_real', color='blue', ax=ax[0,1]).set_title('SP500 vs CPI')
sns.regplot(data=data_1972, x='CPI', y='SCV_real', color='red', ax=ax[1,1]).set_title('SCV vs CPI')
sns.regplot(data=data_1972, x='CPI', y='10_yr_T_real', color='green', ax=ax[1,0]).set_title('10_yr_T vs CPI')
fig.suptitle('Returns vs CPI - post 1971')
plt.show()
```

The results illustrate the above problem. Gold reacts much more positively to inflation post Bretton-Woods, while the other assets don’t show significant impact to their slopes. Let’s answer one more question, before moving on from visualization: Is inflation impacted by SP500 CAPE?


```python
# plots 4

sns.regplot(data=data, x='CAPE', y='CPI').set_title('CPI vs CAPE')
plt.show()
```

The results don’t show much effect.

## What about changes in CAPE?

One might notice that we calculated variables above, that we did not use in the visualizations: the future CAPE ratio, to accompany each future return calculation. We calculated this, so we can now account for changes in valuation, for each point that we measure. We will use that, now in some linear regressions, to see if an increase or decrease in CAPE explains the returns that we see (we’ll refer to this as “CAPE delta”, for simplicity, moving forward).


```python
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
```

We’re not shocked to see that an increase in SP500 valuation (represented by positive CAPE delta) leads to increased SP500 returns $-$ that was a pretty safe assumption. What’s more interesting, is that 10_yr_T is quite positively affected by CAPE delta, but with a wide error margin and a small scale (see the scale on the y-axis of the green chart, and compare it to scale of the blue SP500 chart); Gold appears quite negatively impacted by rising CAPE; SCV seems much more affected by starting CAPE than CAPE delta. These are important distinctions to make, since they show the importance of weighting our upcoming calculations, to account for our relatively high starting CAPE ratio (37.77).

## Modeling Future Possibilities

We’re now going to model possible future return distributions, in order to answer our primary objective: to measure subsequent distribution of possible 10-year performances of our 4 asset classes, given correlations “C”, the average return of each asset class, and the data provided above. To achieve this, we will use Monte Carlo simulations. Monte Carlo is a process of 
 - (a) Sampling our data; we’ll use a parameter called “replacement,” meaning we might reuse data points multiple times in each sample. We’ll average those samples in groups of 10 years, and add the averages into a new dataset. 
 - (b) Perform (a) thousands of times
 - (c) Visualize the average returns of each (a), which will let us see a “distribution of distributions.”

We’ll keep a few principles in mind:

    1. Since we’re sampling thousands of times, our model will already account for the average return of each asset class (further reading: the Central Limit Theorem), but retain the randomness of each year’s return.
    2. Correlations “C” are the average correlations of our assets over time. These correlations will be different in each sample and resample. Again, since we’re doing this thousands of times, our Correlations should closely resemble “C.”
    3. We need to account for our starting CAPE being high. We’ll add a CAPE_percentage column for each year, and add preference for when CAPEs were in the top quartile of cumulative valuations (so, in the top quartile, as of the year records - not the top quartile of all time). This point is really important - we’ll visualize these results against unfiltered results, to see the difference that starting CAPE makes.


```python
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
```

We can see our earlier observations illustrated by the kde plots:
 - 10_yr_T has the lowest average returns, but also the most consistency and fewest outliers (lowest variance and shallowest tails). 10_yr_T also appears unaffected by the High CAPE filter.
 - Gold has decent average returns, but is very streaky - see how positively skewed the distribution for gold returns is in both samples. The High CAPE filter positively impacts Gold’s average return.
 - SP500 is negatively impacted by High CAPE, but still has decent, positive average returns.
 - SCV has the most interesting distribution. It has the highest average return in both filtered and unfiltered samples, but its tails are very long.

The box plots show the distributions compared to each other with better visibility for averages and outliers.

If I had to choose only 1 of these asset classes for the coming 10 years, the data suggests SCV would be my best choice, on average. However, there are many historical scenarios where I would end up with below average returns. Thankfully, I don’t have to choose only 1 asset class! I can choose to allocate a percentage to all of them. Let’s do a new Monte Carlo with our above samples, where we use several allocations, with preference for higher allocations of the higher returning, equity asset classes (portfolios abbr P1, P2, P3, P4):

    1. 25% SCV, 25% SP500, 25% Gold, 25% 10_yr_T
    2. 50% SCV, 50% SP500
    3. 30% SCV, 30% SP500, 20% Gold, 20% 10_yr_T
    4. 40% SCV, 25% SP500, 25% Gold, 10% 10_yr_T


```python
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
```

Each time you run the above simulation, the summary statistics will be different, due to random sampling. Regardless, in every simulation I ran, P2 comes out with the highest mean and minimum cumulative returns. Frequently, the risk of a modeled portfolio is measured by its variance or standard deviation; in this scenario, I want to use the minimum for measuring risk, since the minimum measures our worst expected outcome. In the most recent scenario I ran, P2’s minimum was calculated at 1.36362, meaning P2 gained 136.362% on top of its original value (ex: a portfolio that started with \$10,000 would end up with \$23,636.2 in this scenario - an ~8.98% average annual return!). This is quite a good result and much higher than I expected; I expected lower returns, when measuring portfolios that started with a relatively high SP 500 CAPE ratio. Since P2 is a 50/50 split between SP500 and SCV, let’s test different percentage compositions of those 2 asset classes:

    1. 100% SCV
    2. 75% SCV, 25% SP500
    3. 50% SCV, 50% SP500
    4. 25% SCV, 75% SP500
    5. 100% SP500


```python
P1 = pd.Series(monte_carlo_10(0, 1, 0, 0))
P2 = pd.Series(monte_carlo_10(0.25, 0.75, 0, 0))
P3 = pd.Series(monte_carlo_10(0.5, 0.5, 0, 0))
P4 = pd.Series(monte_carlo_10(0.75, 0.25, 0, 0))
P5 = pd.Series(monte_carlo_10(1, 0, 0, 0))

for p in [P1, P2, P3, P4, P5]:
    print(f"{p}:")
    print(p.describe())
```

Well, the simulations I ran were quite linear and decisive:
	
 - P1 > P2 > P3 > P4 > P5

SCV’s returns were better across all summary statistics; standard deviations were higher, but so also were the minimums. This answers my question about what asset class I should look into: small-cap value.

## What we don't know from this analysis

I now need to acknowledge the shortcomings and blindspots of this project.

    1. The project does not cover valuations for SCV or 10_yr_T (since gold does not have earnings, it does not have a valuation relative to earnings). SCV CAPE and 10_yr_T yields (or at least the change in yield) probably impact the results that we studied in this project.
    2. This project didn’t look at international securities, at all. I actually do have an allocation to international equities in my portfolio, but that was outside the scope of this project. Data for international asset classes is hard for me to access, so I limited this project to those we used.
    3. We can’t predict the future. Markets experience new “1sts” all the time. We might get a huge crash 9 years from now, that completely derails my 10-year return. Since I have no way of predicting this, I cannot “put all my eggs in one basket.”

## Conclusion

I will not buy 100% SCV, remembering that I can’t predict the future. I will, however, very much use an “overweight” SCV portfolio, since models using historical return patterns suggest that will perform best.
