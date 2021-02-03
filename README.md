# advert_data_example
I analyzed data related to clicking or not clicking a particular ad. The dataset comes from Kaggle: https://www.kaggle.com/fayomi/advertising

# Goal (What I learned)

My main goal here was to practice regression in both the Bayesian and frequentist ways. The Bayesian way was done using PyMC3, and the frequentist way was using sklearn. For the implementation in sklearn, I practiced a generalization test of the model by splitting the data into a training and a test set, and then analyzed the results. Typically, this is really only useful when one compares multiple models, but for a dataset as simple as this one, I just wanted some practice doing it for one model. 

I must admit, I was a bit disapoointed to learn that sklearn does not easily provide the p-values of the regression coefficients. For that, one needs to use statsmodels. 

I also did some mild data visualization and exploration to get a rough sense of what was available to me, as well as some extra practice using pandas.

# Conclusions from data:

Here are the conclusions I drew, from my comments in the code:

TLDR: While I thought that people spending more time on the site would lead to more clicks, it was actually the opposite. If you spent a small amount of time on the site, and if you spent smaller amounts of time on the Internet in general, then you were _more_ likely to click on the ad. In addition, since there were slightly more males than females in the sample, especially among those who clicked the ad, being male increased the chances of clicking the ad, according to the model. Thus, it seems the ads were most effective towards people who might be considered impulsive or not particularly familiar with the Internet, especially considering that the people who spent the least amount of time on the site also tended to spend less time on the Internet in general. 


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
