#!/usr/bin/env python
# coding: utf-8

# # Linear Regression with scikit learn

# ### Problem Statement
# > ACME Insurance Inc. offers affordable health insurance to thousands of customer all over the United States. As the lead data scientist at ACME, you're tasked with creating an automated system to estimate the annual medical expenditure for new customers, using information such as their age, sex, BMI, children, smoking habits and region of residence.
# 
# > Estimates from your system will be used to determine the annual insurance premium (amount paid every month) offered to the customer. Due to regulatory requirements, you must be able to explain why your system outputs a certain prediction.

# ### Data
# > We are using a [CSV file](https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv) containing verified historical data, consisting of the aforementioned information and the actual medical charges incurred by over 1300 customers.
# > <img src="./Data_snippet.PNG" width="1000">
# 
# > Dataset source: https://github.com/stedy/Machine-Learning-with-R-datasets
# 

# ### Download the data
# Lets download the data using the `urlretrieve` function from `urllib.request`.

# In[1]:


get_ipython().system('pip install pandas-profiling --quiet')


# In[2]:


import opendatasets as od
import os


# In[3]:


dataset_url = 'https://www.kaggle.com/datasets/harshsingh2209/medical-insurance-payout'


# In[4]:


od.download(dataset_url)


# In[5]:


os.listdir('./medical-insurance-payout')


# In[6]:


import pandas as pd
insaurance_df = pd.read_csv('./medical-insurance-payout/expenses.csv')
insaurance_df


# >The dataset contains 1338 rows and 7 columns. Each row of the dataset contains information about one customer. 
# 
# > Our objective is to find a way to estimate the value in the "charges" column using the values in the other columns. If we can do so for the historical data, then we should able to estimate charges for new customers too, simply by asking for information like their age, sex, BMI, no. of children, smoking habits and region.
# 
# > Let's check the data type for each column.

# In[7]:


insaurance_df.info()


# Looks like "age", "children", "bmi" ([body mass index](https://en.wikipedia.org/wiki/Body_mass_index)) and "charges" are numbers, whereas "sex", "smoker" and "region" are strings (possibly categories). None of the columns contain any missing values, which saves us a fair bit of work!
# 
# Here are some statistics for the numerical columns:

# In[8]:


insaurance_df.describe()


# The ranges of values in the numerical columns seem reasonable too (no negative ages!), so we may not have to do much data cleaning or correction. The "charges" column seems to be significantly skewed however, as the median (50 percentile) is much lower than the maximum value.

# ## Exploratory Analysis and Visualization
# 
# Let's explore the data by visualizing the distribution of values in some columns of the dataset, and the relationships between "charges" and other columns.
# 
# We'll use libraries Matplotlib, Seaborn and Plotly for visualization. Follow these tutorials to learn how to use these libraries: 
# 

# In[9]:


get_ipython().system('pip install plotly matplotlib seaborn --quiet')


# In[10]:


import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# The following settings will improve the default style and font sizes for our charts.

# In[11]:


sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


# In[12]:


insaurance_df.age.min()


# ### Age
# 
# Age is a numeric column. The minimum age in the dataset is 18 and the maximum age is 64. Thus, we can visualize the distribution of age using a histogram with 47 bins (one for each year) and a box plot. We'll use plotly to make the chart interactive, but you can create similar charts using Seaborn.

# In[13]:


fig = px.histogram(insaurance_df, x='age', 
                   marginal='box',
                   nbins=(insaurance_df.age.max()-insaurance_df.age.min())+1, 
                   title='Distribution of Age')

fig.update_layout(bargap=0.1)
fig.show()


# The distribution of ages in the dataset is almost uniform, with 20-30 customers at every age, except for the ages 18 and 19, which seem to have over twice as many customers as other ages. The uniform distribution might arise from the fact that there isn't a big variation in the [number of people of any given age](https://www.statista.com/statistics/241488/population-of-the-us-by-sex-and-age/) (between 18 & 64) in the USA.
# 

# ### Body Mass Index
# 
# Let's look at the distribution of BMI (Body Mass Index) of customers, using a histogram and box plot.

# In[14]:


px.histogram(insaurance_df, x='bmi')


# In[15]:


fig = px.histogram(insaurance_df, 
                  x='bmi', 
                  marginal='box', 
                  color_discrete_sequence=['red'], 
                  title='Distribution of BMI (Body Mass Index)')
fig.update_layout(bargap=0.1)
fig.show()


# The measurements of body mass index seem to form a [Gaussian distribution](https://en.wikipedia.org/wiki/Normal_distribution) centered around the value 30, with a few outliers towards the right. Here's how BMI values can be interpreted ([source](https://study.com/academy/lesson/what-is-bmi-definition-formula-calculation.html)):
# 
# ![](https://i.imgur.com/lh23OiY.jpg)

# ### Charges
# 
# Let's visualize the distribution of "charges" i.e. the annual medical charges for customers. This is the column we're trying to predict. Let's also use the categorical column "smoker" to distinguish the charges for smokers and non-smokers.

# In[16]:


fig = px.histogram(insaurance_df,  
                  x='charges', 
                  marginal='box', 
                  color='smoker', 
                  color_discrete_sequence=['blue', 'grey'], 
                  title='Annual Medical Charges - Smokers vs Non Smokers')
fig.update_layout(bargap=0.1)
fig.show()


# We can make the following observations from the above graph:
# 
# * For most customers, the annual medical charges are under \\$10,000. Only a small fraction of customer have higher medical expenses, possibly due to accidents, major illnesses and genetic diseases. The distribution follows a "power law"
# * There is a significant difference in medical expenses between smokers and non-smokers. While the median for non-smokers is \\$7300, the median for smokers is close to \\$35,000.
# 
# 
# > **EXERCISE**: Visualize the distribution of medical charges in connection with other factors like "sex" and "region". What do you observe?

# In[17]:


fig = px.histogram(data_frame=insaurance_df, x='charges', color='sex', 
                  marginal='box', 
                  title='Annual MedicAL Expenses - Male vs Female')
fig.update_layout(bargap=0.1)
fig.show()


# Here the distribution and correlation of charges between male and female is not that different, as the median expense for male is 9,369 and for female its 9,412.

# In[18]:


fig = px.histogram(data_frame=insaurance_df, 
                  x='charges', color='region', 
                  marginal='box', 
                  title='Annual Medical Expenses, Region-wise')
fig.update_layout(bargap=0.1)
fig.show()


# Here the northwest region has the highest median expense per year which amounts to 10,057 whereas southwest has the least median expense per year which amounts to 8,798

# ### Smoker
# 
# Let's visualize the distribution of the "smoker" column (containing values "yes" and "no") using a histogram.

# In[19]:


insaurance_df.smoker.value_counts()


# In[20]:


px.histogram(insaurance_df, x='smoker', color='sex', 
             title='Distribution of smokers among male and female')


# In[21]:


insaurance_df


# Having looked at individual columns, we can now visualize the relationship between "charges" (the value we wish to predict) and other columns.
# 
# ### Age and Charges
# 
# Let's visualize the relationship between "age" and "charges" using a scatter plot. Each point in the scatter plot represents one customer. We'll also use values in the "smoker" column to color the points.

# In[22]:


fig = px.scatter(insaurance_df, 
                x='age', 
                y='charges', 
                color='smoker', 
                opacity=0.8, 
                hover_data=['sex'],
                title='Scatter plot of Age vs. Charges')
fig.update_layout(bargap=0.1)
fig.show()


# We can make the following observations from the above chart:
# 
# * The general trend seems to be that medical charges increase with age, as we might expect. However, there is significant variation at every age, and it's clear that age alone cannot be used to accurately determine medical charges.
# 
# 
# * We can see three "clusters" of points, each of which seems to form a line with an increasing slope:
# 
#      1. The first and the largest cluster consists primary of presumably "healthy non-smokers" who have relatively low medical charges compared to others
#      
#      2. The second cluster contains a mix of smokers and non-smokers. It's possible that these are actually two distinct but overlapping clusters: "non-smokers with medical issues" and "smokers without major medical issues".
#      
#      3. The final cluster consists exclusively of smokers, presumably smokers with major medical issues that are possibly related to or worsened by smoking.

# ### BMI and Charges
# 
# Let's visualize the relationship between BMI (body mass index) and charges using another scatter plot. Once again, we'll use the values from the "smoker" column to color the points.

# In[23]:


fig = px.scatter(insaurance_df, x='bmi', 
                y='charges', 
                color='smoker',
                hover_data=['sex'], 
                title='Scatter plot of BMI vs. charges')
fig.update_layout(bargap=0.1)
fig.show()


# - It appears that for non-smokers, an increase in BMI doesn't seem to be related to an increase in medical charges. However, medical charges seem to be significantly higher for smokers with a BMI greater than 30.
# - There seem to be almost well defined clusters with regards to smokers and non somkers wit a few outliers for non-smoker population.
# 

# In[24]:


px.violin(insaurance_df,x='sex', y='charges', color='smoker')


# From the above violin graph its vident that the smokers regardless of the genders incur a minimum of what majority of the non smokers incur in terms of medicla bills. 

# ### Correlation
# 
# As you can tell from the analysis, the values in some columns are more closely related to the values in "charges" compared to other columns. E.g. "age" and "charges" seem to grow together, whereas "bmi" and "charges" don't.
# 
# This relationship is often expressed numerically using a measure called the _correlation coefficient_, which can be computed using the `.corr` method of a Pandas series.

# In[25]:


insaurance_df.charges.corr(insaurance_df.age)


# In[26]:


insaurance_df.charges.corr(insaurance_df.bmi)


# To compute the correlation for the categorical columns, they must first be converted into numeric columns.

# In[27]:


smoker_values = {'no':0, 'yes':1}
smoker_numeric = insaurance_df.smoker.map(smoker_values)
insaurance_df.charges.corr(smoker_numeric)


# In[28]:


insaurance_df


# 
# 
# 
# Here's how correlation coefficients can be interpreted ([source](https://statisticsbyjim.com/basics/correlations)):
# 
# * **Strength**: The greater the absolute value of the correlation coefficient, the stronger the relationship.
# 
#     * The extreme values of -1 and 1 indicate a perfectly linear relationship where a change in one variable is accompanied by a perfectly consistent change in the other. For these relationships, all of the data points fall on a line. In practice, you won’t see either type of perfect relationship.
# 
#     * A coefficient of zero represents no linear relationship. As one variable increases, there is no tendency in the other variable to either increase or decrease.
#     
#     * When the value is in-between 0 and +1/-1, there is a relationship, but the points don’t all fall on a line. As r approaches -1 or 1, the strength of the relationship increases and the data points tend to fall closer to a line.
# 
# 
# * **Direction**: The sign of the correlation coefficient represents the direction of the relationship.
# 
#     * Positive coefficients indicate that when the value of one variable increases, the value of the other variable also tends to increase. Positive relationships produce an upward slope on a scatterplot.
#     
#     * Negative coefficients represent cases when the value of one variable increases, the value of the other variable tends to decrease. Negative relationships produce a downward slope.
# 
# Here's the same relationship expressed visually ([source](https://www.cuemath.com/data/how-to-calculate-correlation-coefficient/)):
# 
# <img src="https://i.imgur.com/3XUpDlw.png" width="360">
# 
# The correlation coefficient has the following formula:
# 
# <img src="https://i.imgur.com/unapugP.png" width="360">
# 
# You can learn more about the mathematical definition and geometric interpretation of correlation here: https://www.youtube.com/watch?v=xZ_z8KWkhXE
# 
# Pandas dataframes also provide a `.corr` method to compute the correlation coefficients between all pairs of numeric columns.

# In[29]:


insaurance_df.corr(numeric_only=True)


# The result of `.corr` is called a correlation matrix and is often visualized using a heatmap.

# In[30]:


sns.heatmap(insaurance_df.corr(numeric_only=True), cmap='Reds', annot=True)
plt.title('Correlation matrix');


# **Correlation vs causation fallacy:** Note that a high correlation cannot be used to interpret a cause-effect relationship between features. Two features $X$ and $Y$ can be correlated if $X$ causes $Y$ or if $Y$ causes $X$, or if both are caused independently by some other factor $Z$, and the correlation will no longer hold true if one of the cause-effect relationships is broken. It's also possible that $X$ are $Y$ simply appear to be correlated because the sample is too small. 
# 
# While this may seem obvious, computers can't differentiate between correlation and causation, and decisions based on automated system can often have major consequences on society, so it's important to study why automated systems lead to a given result. Determining cause-effect relationships requires human insight.

# ## Linear Regression using a Single Feature
# 
# We now know that the "smoker" and "age" columns have the strongest correlation with "charges". Let's try to find a way of estimating the value of "charges" using the value of "age" for non-smokers. First, let's create a data frame containing just the data for non-smokers.

# In[31]:


non_smoker_df = insaurance_df[insaurance_df['smoker'] == 'no']
non_smoker_df


# In[32]:


plt.title('age vs. charges')
sns.scatterplot(non_smoker_df, x='age', y='charges', alpha=0.7, s=15);


# Apart from a few exceptions, the points seem to form a line. We'll try and "fit" a line using these points, and use the line to predict charges for a given age. A line on the X&Y coordinates has the following formula:
# 
# $y = wx + b$
# 
# The line is characterized two numbers: $w$ (called "slope") and $b$ (called "intercept"). 
# 
# ### Model
# 
# In the above case, the x axis shows "age" and the y axis shows "charges". Thus, we're assuming the following relationship between the two:
# 
# $charges = w \times age + b$
# 
# We'll try determine $w$ and $b$ for the line that best fits the data. 
# 
# * This technique is called _linear regression_, and we call the above equation a _linear regression model_, because it models the relationship between "age" and "charges" as a straight line. 
# 
# * The numbers $w$ and $b$ are called the _parameters_ or _weights_ of the model.
# 
# * The values in the "age" column of the dataset are called the _inputs_ to the model and the values in the charges column are called "targets". 
# 
# Let define a helper function `estimate_charges`, to compute $charges$, given $age$, $w$ and $b$.
# 

# In[33]:


def estimate_charges(age, w, b):
    return w*age + b


# The `estimate_charges` function is our very first _model_.
# 
# Let's _guess_ the values for $w$ and $b$ and use them to estimate the value for charges.

# In[34]:


w = 50
b = 100


# In[35]:


ages = non_smoker_df.age
estimated_charges = estimate_charges(ages, w, b)


# We can plot the estimated charges using a line graph.

# In[36]:


plt.plot(ages, estimated_charges, 'r--o')
plt.xlabel('Age')
plt.ylabel('Estimated charges')


# As expected, the points lie on a straight line. 
# 
# We can overlay this line on the actual data, so see how well our _model_ fits the _data_.

# In[37]:


target = non_smoker_df.charges

plt.plot(ages, estimated_charges, 'r', alpha=0.9)
plt.scatter(ages, target, s=8, alpha=0.8)
plt.xlabel('Age')
plt.ylabel('Charges')
plt.legend(['Estimated Charges', 'Actual Charges']);


# Clearly, the our estimates are quite poor and the line does not "fit" the data. However, we can try different values of $w$ and $b$ to move the line around. Let's define a helper function `try_parameters` which takes `w` and `b` as inputs and creates the above plot.

# In[38]:


def try_parameters(w, b):
    ages = non_smoker_df.age
    estimated_charges = estimate_charges(ages, w, b)
    
    target = non_smoker_df.charges

    plt.plot(ages, estimated_charges, 'r', alpha=0.9)
    plt.scatter(ages, target, s=8, alpha=0.8)
    plt.xlabel('Age')
    plt.ylabel('Charges')
    plt.legend(['Estimated Charges', 'Actual Charges']);


# In[39]:


try_parameters(20, 100)


# In[40]:


try_parameters(130, 10)


# In[41]:


try_parameters(90, 5000)


# we can clearly see that the changes made to the w has significantely more effect on the line than b

# As we change the values, of $w$ and $b$ manually, trying to move the line visually closer to the points, we are _learning_ the approximate relationship between "age" and "charges". 
# 
# Wouldn't it be nice if a computer could try several different values of `w` and `b` and _learn_ the relationship between "age" and "charges"? To do this, we need to solve a couple of problems:
# 
# 1. We need a way to measure numerically how well the line fits the points.
# 
# 2. Once the "measure of fit" has been computed, we need a way to modify `w` and `b` to improve the the fit.
# 
# If we can solve the above problems, it should be possible for a computer to determine `w` and `b` for the best fit line, starting from a random guess.

# ### Loss/Cost Function
# 
# We can compare our model's predictions with the actual targets using the following method:
# 
# * Calculate the difference between the targets and predictions (the differenced is called the "residual")
# * Square all elements of the difference matrix to remove negative values.
# * Calculate the average of the elements in the resulting matrix.
# * Take the square root of the result
# 
# The result is a single number, known as the **root mean squared error** (RMSE). The above description can be stated mathematically as follows: 
# 
# <img src="https://i.imgur.com/WCanPkA.png" width="360">
# 
# Geometrically, the residuals can be visualized as follows:
# 
# <img src="https://i.imgur.com/ll3NL80.png" width="420">
# 
# Let's define a function to compute the RMSE.

# In[42]:


get_ipython().system('pip install numpy --quiet')


# In[43]:


import numpy as np


# In[44]:


def rmse(targets, predictions):
    return np.sqrt(np.mean(np.square(targets - predictions)))


# Let's compute the RMSE for our model with a sample set of weights

# In[45]:


w = 50
b = 100


# In[46]:


try_parameters(w, b)


# In[47]:


targets = non_smoker_df.charges
predicted = estimate_charges(non_smoker_df.age, w, b)
rmse(targets, predicted)


# Here's how we can interpret the above number: *On average, each element in the prediction differs from the actual target by \\$8461*. 
# 
# The result is called the *loss* because it indicates how bad the model is at predicting the target variables. It represents information loss in the model: the lower the loss, the better the model.
# 
# Let's modify the `try_parameters` functions to also display the loss.

# In[48]:


def try_parameters(w, b):
    ages = non_smoker_df.age
    predictions = estimate_charges(ages, w, b)
    
    target = non_smoker_df.charges

    plt.plot(ages, predictions, 'r', alpha=0.9)
    plt.scatter(ages, target, s=8, alpha=0.8)
    plt.xlabel('Age')
    plt.ylabel('Charges')
    plt.legend(['Predictions', 'Actual']);
    loss = rmse(target, predictions)
    print(f'The RMSE Loss is {loss}')


# In[49]:


try_parameters(160, 2400)


# ### Optimizer
# 
# Next, we need a strategy to modify weights `w` and `b` to reduce the loss and improve the "fit" of the line to the data.
# 
# * Ordinary Least Squares: https://www.youtube.com/watch?v=szXbuO3bVRk (better for smaller datasets)
# * Stochastic gradient descent: https://www.youtube.com/watch?v=sDv4f4s2SB8 (better for larger datasets)
# 
# Both of these have the same objective: to minimize the loss, however, while ordinary least squares directly computes the best values for `w` and `b` using matrix operations, while gradient descent uses a iterative approach, starting with a random values of `w` and `b` and slowly improving them using derivatives. 
# 
# Here's a visualization of how gradient descent works:
# 
# ![](https://miro.medium.com/max/1728/1*NO-YvpHHadk5lLxtg4Gfrw.gif)
# 
# Doesn't it look similar to our own strategy of gradually moving the line closer to the points?
# 
# 

# ### Linear Regression using Scikit-learn
# 
# In practice, you'll never need to implement either of the above methods yourself. You can use a library like `scikit-learn` to do this for you. 

# In[50]:


get_ipython().system('pip install scikit-learn --quiet')


# Let's use the `LinearRegression` class from `scikit-learn` to find the best fit line for "age" vs. "charges" using the ordinary least squares optimization technique.

# In[51]:


from sklearn.linear_model import LinearRegression


# First we create a new model object.

# In[52]:


model = LinearRegression()


# Next, we can use the `fit` method of the model to find the best fit line for the inputs and targets.

# In[53]:


help(model.fit)


# Note that the input `X` must be a 2-d array, so we'll need to pass a dataframe, instead of a single column.

# In[54]:


inputs = non_smoker_df[['age']]
targets = non_smoker_df.charges
print(f'inputs.shape: {inputs.shape}')
print(f'targets.shape: {targets.shape}')


# Let's fit the model to the data.

# In[55]:


model.fit(inputs, targets)


# In[56]:


model.predict(np.array([[23]]))


# Do these values seem reasonable? Compare them with the scatter plot above.
# 
# Let compute the predictions for the entire set of inputs

# In[57]:


predictions = model.predict(inputs)


# In[58]:


predictions


# Let's compute the RMSE loss to evaluate the model.

# In[59]:


rmse(targets, predictions=predictions)


# Seems like our prediction is off by $4000 on average, which is not too bad considering the fact that there are several outliers.

# The parameters of the model are stored in the `coef_` and `intercept_` properties.

# In[60]:


#w
model.coef_


# In[61]:


#b
model.intercept_


# Are these parameters close to your best guesses?
# 
# Let's visualize the line created by the above parameters.

# In[62]:


try_parameters(model.coef_, model.intercept_)


# Indeed the line is quite close to the points. It is slightly above the cluster of points, because it's also trying to account for the outliers. 
# 
# > **EXERCISE**: Use the [`SGDRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html) class from `scikit-learn` to train a model using the stochastic gradient descent technique. Make predictions and compute the loss. Do you see any difference in the result?

# In[63]:


from sklearn.linear_model import SGDRegressor
model_n_sgd = SGDRegressor()
model_n_sgd.fit(inputs, targets)


# In[64]:


predictions = model_n_sgd.predict(inputs)


# In[65]:


model_n_sgd.coef_


# In[66]:


model_n_sgd.intercept_


# In[67]:


try_parameters(model_n_sgd.coef_, model_n_sgd.intercept_)


# In[68]:


rmse(targets=targets, predictions=predictions)


# Here we have implemented the Stochastic Gradient Descent model, which uses.
# the gradient of the loss is estimated each sample at a time and the model is updated along the way with a decreasing strength schedule (aka learning rate)

# ## Let us have a look at the smokers data now

# In[69]:


smokers_df = insaurance_df[insaurance_df['smoker']=='yes']


# ### Apply Linear regression model

# In[70]:


smokers_df


# In[71]:


inputs_smokers = smokers_df[['age']]
targets_smokers = smokers_df.charges
model_s_lr = LinearRegression()
model_s_lr.fit(inputs_smokers, targets_smokers)
predictions_lr = model_s_lr.predict(inputs_smokers)


# In[72]:


print(f'coefficient: {model_s_lr.coef_}, intercept: {model_s_lr.intercept_}')


# In[73]:


loss_s_lr = rmse(targets_smokers, predictions_lr)
loss_s_lr


# ### Now let's try stochastic gradient descent

# In[74]:


inputs_smokers


# In[75]:


targets_smokers


# In[76]:


model_s_sgd = SGDRegressor()


# In[77]:


model_s_sgd.fit(inputs_smokers, targets_smokers)


# In[78]:


print(f'Number of weight updates performed during training: {model_s_sgd.t_}')
print(f'number of iterations computed: {model_s_sgd.n_iter_}')
print(f'coefficient: {model_s_sgd.coef_}, intercept: {model_s_sgd.intercept_}')


# In[79]:


predictions_sgd = model_s_sgd.predict(inputs_smokers)
predictions_sgd


# In[80]:


loss_s_sdg = rmse(targets_smokers, predictions_sgd)
loss_s_sdg


# ## Very important thing here..
# 
# > I am unable to debug why the coefficient and intercepts are so high when calculating with `SDGRegressor()`!!

# ### Machine Learning
# 
# Congratulations, you've just trained your first _machine learning model!_ Machine learning is simply the process of computing the best parameters to model the relationship between some feature and targets. 
# 
# Every machine learning problem has three components:
# 
# 1. **Model**
# 
# 2. **Cost Function**
# 
# 3. **Optimizer**
# 
# We'll look at several examples of each of the above in future tutorials. Here's how the relationship between these three components can be visualized:
# 
# <img src="https://i.imgur.com/oiGQFJ9.png" width="480">

# As we've seen above, it takes just a few lines of code to train a machine learning model using `scikit-learn`.

# In[81]:


# create inputs and targets
inputs, targets = non_smoker_df[['age']], non_smoker_df['charges']

# Create and train the model
model = LinearRegression().fit(inputs, targets)

# Generate predictions
predictions = model.predict(inputs)

# Compute loss to evaluate the model
loss = rmse(targets, predictions)
print('Loss: ', loss)


# ## Linear Regression using Multiple Features
# 
# So far, we've used on the "age" feature to estimate "charges". Adding another feature like "bmi" is fairly straightforward. We simply assume the following relationship:
# 
# $charges = w_1 \times age + w_2 \times bmi + b$
# 
# We need to change just one line of code to include the BMI.

# In[82]:


# create inputs and targets
inputs, targets = non_smoker_df[['age', 'bmi']], non_smoker_df.charges

# Create and train the model
model_mlr = LinearRegression()
model_mlr.fit(X=inputs, y=targets)

# Generate predictions
predictions = model_mlr.predict(inputs)

# Compute loss to evaluate the model
loss = rmse(targets, predictions)
print(f'Loss: {loss}')


# As you can see, adding the BMI doesn't seem to reduce the loss by much, as the BMI has a very weak correlation with charges, especially for non smokers.

# In[83]:


non_smoker_df.charges.corr(non_smoker_df.bmi)


# In[84]:


fig = px.scatter(non_smoker_df, x='bmi', y='charges', title='BMI vs. Charges')
fig.update_traces(marker_size=5)
fig.show()


# We can also visualize the relationship between all 3 variables "age", "bmi" and "charges" using a 3D scatter plot.

# In[85]:


fig = px.scatter_3d(non_smoker_df, x='age', y='bmi', z='charges')
fig.update_traces(marker_size=3, marker_opacity=0.5)
fig.show()


# You can see that it's harder to interpret a 3D scatter plot compared to a 2D scatter plot. As we add more features, it becomes impossible to visualize all feature at once, which is why we use measures like correlation and loss. 
# 
# Let's also check the parameters of the model.

# In[86]:


model_mlr.coef_, model_mlr.intercept_


# Clearly, BMI has a much lower weightage, and you can see why. It has a tiny contribution, and even that is probably accidental. This is an important thing to keep in mind: you can't find a relationship that doesn't exist, no matter what machine learning technique or optimization algorithm you apply. 

# Let's go one step further, and add the final numeric column: "children", which seems to have some correlation with "charges".
# 
# $charges = w_1 \times age + w_2 \times bmi + w_3 \times children + b$

# In[87]:


non_smoker_df.charges.corr(non_smoker_df.children)


# In[88]:


fig = px.strip(non_smoker_df, x='children', y='charges', title='children vs. charges')
fig.update_traces(marker_size=4, marker_opacity=0.7)
fig.show()


# In[89]:


# Create inputs and targets
inputs, targets = non_smoker_df[['age', 'bmi', 'children']], non_smoker_df['charges']

# Create and train the model
model = LinearRegression()
model.fit(X=inputs, y=targets)

#Gererate Predictions
predictions = model.predict(inputs)

# Compute loss and evaluate the model
loss = rmse(targets, predictions)
print(f'Loss: {loss}')


# Once again, we don't see a big reduction in the loss, even though it's greater than in the case of BMI.

# ### For smokers_df

# In[90]:


# Create inputs and targets
inputs, targets = smokers_df[['age', 'bmi', 'children']], smokers_df['charges']

#Create and train the Linear model
model = LinearRegression()
model.fit(inputs, targets)

# Generate Predictions
predictions = model.predict(inputs)

# Calculate loss and evaluate the model
loss = rmse(targets, predictions)
print(f'Loss: {loss}')


# In[91]:


# Create inputs and targets
inputs, targets = insaurance_df[['age', 'bmi', 'children']], insaurance_df['charges']

#Create and train the Linear model
model = LinearRegression()
model.fit(inputs, targets)

# Generate Predictions
predictions = model.predict(inputs)

# Calculate loss and evaluate the model
loss = rmse(targets, predictions)
print(f'Loss: {loss}')


# ## Using Categorical Features for Machine Learning
# 
# So far we've been using only numeric columns, since we can only perform computations with numbers. If we could use categorical columns like "smoker", we can train a single model for the entire dataset.
# 
# To use the categorical columns, we simply need to convert them to numbers. There are three common techniques for doing this:
# 
# 1. If a categorical column has just two categories (it's called a binary category), then we can replace their values with 0 and 1.
# 2. If a categorical column has more than 2 categories, we can perform one-hot encoding i.e. create a new column for each category with 1s and 0s.
# 3. If the categories have a natural order (e.g. cold, neutral, warm, hot), then they can be converted to numbers (e.g. 1, 2, 3, 4) preserving the order. These are called ordinals
# 
# 
# 

# ## Binary Categories
# 
# The "smoker" category has just two values "yes" and "no". Let's create a new column "smoker_code" containing 0 for "no" and 1 for "yes".
# 

# In[94]:


sns.barplot(data=insaurance_df, x='smoker', y='charges', hue='sex')


# In[95]:


smoker_codes = {'no' : 0, 'yes' : 1}
insaurance_df['smoker_code'] = insaurance_df.smoker.map(smoker_codes)


# In[98]:


insaurance_df.charges.corr(insaurance_df.smoker_code)


# In[101]:


insaurance_df


# We can now use the `smoker_code` column for linear regression.
# 
# $charges = w_1 \times age + w_2 \times bmi + w_3 \times children + w_4 \times smoker + b$

# In[104]:


# Create the inputs and targets
inputs, targets = insaurance_df[['age', 'bmi', 'children', 'smoker_code']], insaurance_df.charges

# Create the model and fit the data to it
model = LinearRegression()
model.fit(inputs, targets)

# Calculate the predictions
predictions = model.predict(inputs)

# Error Calculation and model evaluation
loss = rmse(targets, predictions)
print(f'Loss: {loss}')


# The loss reduces from `11355` to `6056`, almost by 50%! This is an important lesson: never ignore categorical data.
# 
# 
# Let's try adding the "sex" column as well.
# 
# $charges = w_1 \times age + w_2 \times bmi + w_3 \times children + w_4 \times smoker + w_5 \times sex + b$

# In[105]:


sns.barplot(data=insaurance_df, x='sex', y='charges')


# In[106]:


sex_codes = {'female' : 0, 'male' : 1}


# In[107]:


insaurance_df['sex_code'] = insaurance_df.sex.map(sex_codes)


# In[109]:


insaurance_df.charges.corr(insaurance_df.sex_code)


# In[110]:


# Create the inputs and targets
inputs, targets = insaurance_df[['age', 'bmi', 'children', 'smoker_code', 'sex_code']], insaurance_df.charges

# Create the model and fit the data onto it
model = LinearRegression()
model.fit(inputs, targets)

# Make predictions using the trained model on the inputs
predictions = model.predict(inputs)

# Calculate the loss and evaluate the model
loss = rmse(targets, predictions)
print(f'Loss: {loss}')


# As seen from above there isn't significant drop in the error here!!

# 
# ### One-hot Encoding
# 
# The "region" column contains 4 values, so we'll need to use hot encoding and create a new column for each region.
# 
# ![](https://i.imgur.com/n8GuiOO.png)
# 

# In[111]:


sns.barplot(data=insaurance_df, x='region', y='charges')


# As you can see that there are 4 disctint values for the column region

# In[112]:


from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()
enc.fit(insaurance_df[['region']])
enc.categories_


# In[114]:


one_hot = enc.transform(insaurance_df[['region']]).toarray()
one_hot


# In[115]:


insaurance_df[['northeast', 'northwest', 'southeast', 'southwest']] = one_hot


# In[116]:


insaurance_df


# Let's include the region columns into our linear regression model.
# 
# $charges = w_1 \times age + w_2 \times bmi + w_3 \times children + w_4 \times smoker + w_5 \times sex + w_6 \times region + b$

# In[122]:


# Create the inputs and targets
inputs, targets = insaurance_df[['age', 'bmi', 'children', 'smoker_code', 'sex_code', 'northeast', 'northwest', 'southeast', 'southwest']], insaurance_df.charges

# Create the model and fit the data onto it
model_full_df = LinearRegression()
model_full_df.fit(inputs, targets)

# Make predictions based on the trained model
predictions = model_full_df.predict(inputs)

# Calculate loss anf evaluate the model
loss = rmse(targets, predictions)
print(f'Loss: {loss}')


# Once again, this leads to a fairly small reduction in the loss. 
# 
# > **EXERCISE**: Are two separate linear regression models, one for smokers and one of non-smokers, better than a single linear regression model? Why or why not? Try it out and see if you can justify your answer with data.

# ### smoker_new_df and non_smoker_new_df

# In[119]:


smoker_new_df = insaurance_df[insaurance_df['smoker'] == 'yes']
non_smoker_new_df = insaurance_df[insaurance_df['smoker'] == 'no']


# In[120]:


# Create the inputs and targets for smoker_new_df
inputs, targets = smoker_new_df[['age', 'bmi', 'children', 'smoker_code', 'sex_code', 'northeast', 'northwest', 'southeast', 'southwest']], smoker_new_df.charges

# Create the model and fit the data onto it
model = LinearRegression()
model.fit(inputs, targets)

# Make predictions based on the trained model
predictions = model.predict(inputs)

# Calculate loss anf evaluate the model
loss = rmse(targets, predictions)
print(f'Loss: {loss}')


# In[130]:


# Create the inputs and targets for non_smoker_new_df
inputs, targets = non_smoker_new_df[['age', 'bmi', 'children', 'smoker_code', 'sex_code', 'northeast', 'northwest', 'southeast', 'southwest']], non_smoker_new_df.charges
print(inputs.columns)
# Create the model and fit the data onto it
model = LinearRegression()
model.fit(inputs, targets)

# Make predictions based on the trained model
predictions = model.predict(inputs)

# Calculate loss anf evaluate the model
loss = rmse(targets, predictions)
print(f'Loss: {loss}')


# - Separately when we analyze the error is smaller in each case when compared to the entire dataset.

# ## Model Improvements
# 
# Let's discuss and apply some more improvements to our model.
# 
# ### Feature Scaling
# 
# Recall that due to regulatory requirements, we also need to explain the rationale behind the predictions our model. 
# 
# $charges = w_1 \times age + w_2 \times bmi + w_3 \times children + w_4 \times smoker + w_5 \times sex + w_6 \times region + b$
# 
# To compare the importance of each feature in the model, our first instinct might be to compare their weights. 

# In[123]:


model_full_df.coef_


# In[124]:


model_full_df.intercept_


# In[128]:


insaurance_df.columns


# In[133]:


weights_df = pd.DataFrame({
    'feature': np.append(inputs.columns, 1), 
    'weight': np.append(model_full_df.coef_, model_full_df.intercept_)
})
weights_df


# While it seems like BMI and the "northeast" have a higher weight than age, keep in mind that the range of values for BMI is limited (15 to 40) and the "northeast" column only takes the values 0 and 1.
# 
# Because different columns have different ranges, we run into two issues:
# 
# 1. We can't compare the weights of different column to identify which features are important
# 2. A column with a larger range of inputs may disproportionately affect the loss and dominate the optimization process.
# 
# For this reason, it's common practice to scale (or standardize) the values in numeric column by subtracting the mean and dividing by the standard deviation.
# 
# ![](https://i.imgur.com/dT5fLFI.png)
# 
# We can apply scaling using the StandardScaler class from `scikit-learn`.

# In[136]:


from sklearn.preprocessing import StandardScaler


# In[137]:


numeric_cols = ['age', 'bmi', 'children']
scaler = StandardScaler()
scaler.fit(insaurance_df[numeric_cols])


# In[138]:


scaler.mean_


# In[139]:


scaler.var_


# We can now scale the data as follows:

# In[145]:


scaled_inputs = scaler.transform(insaurance_df[numeric_cols])
scaled_inputs


# - These can now be combined with the categorical data

# In[146]:


cat_cols = ['smoker_code', 'sex_code', 'northeast', 'northwest', 'southeast', 'southwest']
categorical_data = insaurance_df[cat_cols].values
categorical_data


# -  Now you can concatinate the numerical scaled data (with the shape`(1338, 3)`) with the categorical data (with the shape`(1338,6 )`) along the axis 1 which is along the columns

# In[148]:


# Create inputs and targets
inputs = np.concatenate((scaled_inputs, categorical_data), axis=1)
targets = insaurance_df.charges

# Create the model and fit it to the data
model =  LinearRegression()
model.fit(inputs, targets)

# Create the predictions
predictions = model.predict(inputs)

# Loss calculation and model evaluation
loss = rmse(targets, predictions)
print(f'Loss: {loss}')


# We can now compare the weights in the formula:
# 
# $charges = w_1 \times age + w_2 \times bmi + w_3 \times children + w_4 \times smoker + w_5 \times sex + w_6 \times region + b$

# In[150]:


weights_df = pd.DataFrame({
    'features':np.append(numeric_cols + cat_cols, 1), 
    'weights': np.append(model.coef_, model.intercept_)
})
weights_df.sort_values('weights', ascending=False)


# As you see the 3 most important feature are
# 1. Smoker
# 2. Age
# 3. BMI
# 

# ### Creating a Test Set - cross validation step.
# 
# Models like the one we've created in this tutorial are designed to be used in the real world. It's common practice to set aside a small fraction of the data (e.g. 10%) just for testing and reporting the results of the model.

# In[176]:


from sklearn.model_selection import train_test_split


# In[177]:


inputs = np.concatenate((scaled_inputs, categorical_data), axis=1)
targets = insaurance_df.charges


# In[178]:


inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.1)


# In[180]:


# Create the model and fit the data to it
model = LinearRegression()
model.fit(inputs_train, targets_train)

# Make predictions using the test and train data
predictions_test = model.predict(inputs_test)
predictions_train = model.predict(inputs_train)

# Loss calculation and model evaluation
loss_test = rmse(targets_test, predictions_test)
loss_train = rmse(targets_train, predictions_train)

print(f'Train loss: {loss_train} \n Testing loss: {loss_test}')


# ### How to Approach a Machine Learning Problem
# 
# Here's a strategy you can apply to approach any machine learning problem:
# 
# 1. Explore the data and find correlations between inputs and targets
# 2. Pick the right model, loss functions and optimizer for the problem at hand
# 3. Scale numeric variables and one-hot encode categorical data
# 4. Set aside a test set (using a fraction of the training set)
# 5. Train the model
# 6. Make predictions on the test set and compute the loss
# 
# We'll apply this process to several problems in future.

# In[ ]:




