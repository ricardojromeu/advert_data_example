# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 23:15:27 2020

@author: ricro

Looking at Ad-Tech Data Downloaded from Kaggle on 11/3/20
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pymc3 as pm
import arviz as az
import theano.tensor as T
import sklearn

data = pd.read_csv('C:\\Users\\ricro\\Desktop\\Data Science Projects in Python\\Analyze_Data\\advertising.csv')


data.head()

"""
Column Names:
    (1) Daily time spent on Site
    (2) Age
    (3) Area Income
    (4) Daily Internet Usage
    (5) Ad Topic Line
    (6) City
    (7) Male (1), vs Female (0)
    (8) Country 
    (9) Timestamp
    (10) Clicked on Ad (0,1)

"""

# My first thought is, can daily time spent on site differentiate between click or no click?

click = data['Clicked on Ad'] == 1
no_click = data['Clicked on Ad'] == 0

time_click = data.loc[click.values, 'Daily Time Spent on Site']
time_noclick = data.loc[no_click.values, 'Daily Time Spent on Site']

plt.hist(time_click, bins = 30, alpha = 0.5, label = 'click')
plt.hist(time_noclick, bins = 30, alpha = 0.5, label = 'no click')
plt.legend(loc='upper left')
plt.show()

# It seems then that spending MORE time on the site leads to a smaller chance of clicking
# Let's see if there is a significant difference between the two groups using a simple t-test:
    
stat, pval = stats.ttest_ind(time_click, time_noclick)

print("T-test results of checking difference between clicked and non-clicked groups on Daily Time spent on Site")
print([stat, pval])
# Stat = -35.616, with a very, very small p value.
# So it seems there is a significant difference in the amount of time spent on the site
# And in fact, with a negative stat, we see that time_click < time_noclick
# So people who spend LESS time on the site are more likely to click. 

# Does this relationship change if we consider the multivariate (Time spent on site, Daily Internet Usage)

# Let's make a scatterplot of these variables, splitting by action
cols = ['Daily Time Spent on Site', 'Daily Internet Usage']

daily_data = data[cols]

daily_click = np.log(daily_data[click])
daily_noclick = np.log(daily_data[no_click])



plt.scatter(daily_click[cols[0]], daily_click[cols[1]], marker='o',label='click')
plt.scatter(daily_noclick[cols[0]], daily_noclick[cols[1]], marker='o',label='noclick')
plt.legend(loc='upper left')
plt.xlabel('Site Time')
plt.ylabel('Int Usage')
plt.show()


# Just at a glance, it seems like there is a clear distinction between the two sets of groups
# We can check for a significant difference using Hotelling's T^2

# Let's look at the covariance matrices for each group:
    
click_cov = daily_click.cov()
noclick_cov = daily_noclick.cov()

print(click_cov)
print(noclick_cov)    
# The covariance matrices for each group are not that similar
    
# Trying a simple Bayesian logistic regression model to predict click/no click
# using daily time spent on site and Daily internet usage

X1 = data["Daily Time Spent on Site"].values
#X1 = X1 - X1.mean()
X2 = data["Daily Internet Usage"].values
#X2 = X2 - X2.mean()
Y = data["Clicked on Ad"].values

"""
BEGIN BAYESIAN FITTING HERE;
    COMMENT OUT TO SAVE COMPUTATION TIME

"""


# with pm.Model() as adv:
#     beta = pm.Normal('beta', mu = 0, sd = 1, shape = 3)
    
#     mu = pm.Deterministic('mu', beta[0] + beta[1]*X1 + beta[2]*X2)
#     L = pm.Deterministic('L', (1 + T.exp(-mu))**(-1))
    
#     y = pm.Bernoulli('y', p = L, observed = Y)
    
#     trace_adv = pm.sample(2000, tune = 1000, cores = 1)
    

# Summary = az.summary(trace_adv, var_names = ['beta'])
# print(Summary)


# Now try the same thing using sklearn. 

# Check accuracy of model by using a training set and a test set, and check
# standard accuracy measures on logistic classification on test set. 

regressor = sklearn.linear_model.LogisticRegression()
X = data[["Daily Time Spent on Site", "Age", "Daily Internet Usage","Male"]]
Y = data["Clicked on Ad"]

# Some extra questions to ask before we look at regressors, though

# Is the sample biased with either more males or more females?

perc_gender = data["Male"].value_counts()/len(data["Male"])
print("Percentage of Males and Females in Sample:")
perc_gender.index = ["Male", "Female"]
print(perc_gender)
# Slight bias towards men, but not dramatic

# Also need to make sure the regressors don't have too strong of a 
# correlation to avoid collinearity

corr_X = X.corr()

print(corr_X)

# Indeed, we saw this correlation in our joint distribution plots above

# Is gender a disparity between clicked vs non-clicked?
print("Clicked Proportion of M/F")
print(X[Y == 1]["Male"].value_counts()/sum(Y == 1))
# Again, slight disparity in gender here

print("Non-clicked Proportion of M/F")
print(X[Y == 0]["Male"].value_counts()/sum(Y == 0))
# These proportions are exactly 50-50 in this data set.

# Let's split the data into test/train data

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X,
                                                                            Y, 
                                                                            test_size = .20,
                                                                            random_state = 0) 

regressor.fit(X_train, Y_train)

# We can check the coefficients and the intercept:
print("Beta Coefficients and Intercept")
print("Variables: [Time on Site, Age, Internet Usage, Male]")
print(regressor.coef_)
print(regressor.intercept_)

# We can also check how the model performed using mean_sqaured_error
# and r2_score

# First, get predictions from the model
y_pred = regressor.predict(X_test)

print("MSE")
print(sklearn.metrics.mean_squared_error(Y_test, y_pred))
print("R^2")
r2 = sklearn.metrics.r2_score(Y_test, y_pred)
print(r2)

"""
Here are the results thus far:

Beta Coefficients and Intercept

Variables: [Time on Site, Age, Internet Usage, Male]
[[-0.15122578  0.09977007 -0.05878246  0.00195414]]

Intercept:
[17.51655033]

MSE
0.055

R^2
0.7785829307568437

The largest beta coefficients are those related to:
    (1) Daily Time Spent on Site
    (2) Age
    (3) Daily Internet Usage
    
We can see that, for every unit of daily time spent on site, the log-odds
of clicking DECREASE by about .15

For every increase in age, the log-odds of clicking increase by about .10

For every unit of time spent on the internet daily, the log-odds of clicking
DECREASE by about 0.06

It also seems that being Male increases the log-odds of clicking by about 0.002
However, we observed a slight bias in the sample towards sampling men,
and our peak at the data showed that the distribution of males and females
in the clicked vs not clicked groups were not the same; the proportion of men
who clicked was slightly higher than women who clicked. 
Additionally, this effect doesn't appear to be that great here. 

It other words, it seems like the typical person who clicked the ad:
    Tends not to spend as much time on the site
    Tends not to spend as much time on the internet in general
    Tends to be older
    Slight trend towards tending to be male more often than female

"""

# One last thing I'd like to check: What are the counts of Ad Topic Line
# between clicked and non-clicked?

print("Ad Topic Counts for Clicked vs Non-clicked")
print("Clicked:")
print(data[Y == 1]["Ad Topic Line"].value_counts()/sum(Y == 1))

print("Non-Clicked:")
print(data[Y == 0]["Ad Topic Line"].value_counts()/sum(Y == 0))

# Hmm, seems all the ad topic lines were different;
# Seems like the data consisted of profiles for each ad topic line, one
# from someone who clicked and another from someone who didn't click





