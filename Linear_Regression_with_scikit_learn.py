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

# In[113]:


get_ipython().system('pip install pandas-profiling --quiet')


# In[114]:


import opendatasets as od
import os


# In[115]:


dataset_url = 'https://www.kaggle.com/datasets/harshsingh2209/medical-insurance-payout'


# In[116]:


od.download(dataset_url)


# In[117]:


os.listdir('./medical-insurance-payout')


# In[118]:


import pandas as pd
insaurance_df = pd.read_csv('./medical-insurance-payout/expenses.csv')
insaurance_df


# >The dataset contains 1338 rows and 7 columns. Each row of the dataset contains information about one customer. 
# 
# > Our objective is to find a way to estimate the value in the "charges" column using the values in the other columns. If we can do so for the historical data, then we should able to estimate charges for new customers too, simply by asking for information like their age, sex, BMI, no. of children, smoking habits and region.
# 
# > Let's check the data type for each column.

# In[119]:


insaurance_df.info()


# Looks like "age", "children", "bmi" ([body mass index](https://en.wikipedia.org/wiki/Body_mass_index)) and "charges" are numbers, whereas "sex", "smoker" and "region" are strings (possibly categories). None of the columns contain any missing values, which saves us a fair bit of work!
# 
# Here are some statistics for the numerical columns:

# In[120]:


insaurance_df.describe()


# The ranges of values in the numerical columns seem reasonable too (no negative ages!), so we may not have to do much data cleaning or correction. The "charges" column seems to be significantly skewed however, as the median (50 percentile) is much lower than the maximum value.

# ## Exploratory Analysis and Visualization
# 
# Let's explore the data by visualizing the distribution of values in some columns of the dataset, and the relationships between "charges" and other columns.
# 
# We'll use libraries Matplotlib, Seaborn and Plotly for visualization. Follow these tutorials to learn how to use these libraries: 
# 

# In[121]:


get_ipython().system('pip install plotly matplotlib seaborn --quiet')


# In[122]:


import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# The following settings will improve the default style and font sizes for our charts.

# In[123]:


sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


# In[124]:


insaurance_df.age.min()


# ### Age
# 
# Age is a numeric column. The minimum age in the dataset is 18 and the maximum age is 64. Thus, we can visualize the distribution of age using a histogram with 47 bins (one for each year) and a box plot. We'll use plotly to make the chart interactive, but you can create similar charts using Seaborn.

# In[125]:


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

# In[126]:


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

# In[127]:


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

# In[128]:


fig = px.histogram(data_frame=insaurance_df, x='charges', color='sex', 
                  marginal='box', 
                  title='Annual MedicAL Expenses - Male vs Female')
fig.update_layout(bargap=0.1)
fig.show()


# Here the distribution and correlation of charges between male and female is not that different, as the median expense for male is 9,369 and for female its 9,412.

# In[129]:


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

# In[130]:


insaurance_df.smoker.value_counts()


# In[131]:


px.histogram(insaurance_df, x='smoker', color='sex', 
             title='Distribution of smokers among male and female')


# In[132]:


insaurance_df


# Having looked at individual columns, we can now visualize the relationship between "charges" (the value we wish to predict) and other columns.
# 
# ### Age and Charges
# 
# Let's visualize the relationship between "age" and "charges" using a scatter plot. Each point in the scatter plot represents one customer. We'll also use values in the "smoker" column to color the points.

# In[133]:


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

# In[134]:


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

# In[135]:


px.violin(insaurance_df,x='sex', y='charges', color='smoker')


# From the above violin graph its vident that the smokers regardless of the genders incur a minimum of what majority of the non smokers incur in terms of medicla bills. 

# ### Correlation
# 
# As you can tell from the analysis, the values in some columns are more closely related to the values in "charges" compared to other columns. E.g. "age" and "charges" seem to grow together, whereas "bmi" and "charges" don't.
# 
# This relationship is often expressed numerically using a measure called the _correlation coefficient_, which can be computed using the `.corr` method of a Pandas series.

# In[136]:


insaurance_df.charges.corr(insaurance_df.age)


# In[137]:


insaurance_df.charges.corr(insaurance_df.bmi)


# To compute the correlation for the categorical columns, they must first be converted into numeric columns.

# In[138]:


smoker_values = {'no':0, 'yes':1}
smoker_numeric = insaurance_df.smoker.map(smoker_values)
insaurance_df.charges.corr(smoker_numeric)


# In[139]:


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

# In[140]:


insaurance_df.corr(numeric_only=True)


# The result of `.corr` is called a correlation matrix and is often visualized using a heatmap.

# In[142]:


sns.heatmap(insaurance_df.corr(numeric_only=True), cmap='Reds', annot=True)
plt.title('Correlation matrix');


# **Correlation vs causation fallacy:** Note that a high correlation cannot be used to interpret a cause-effect relationship between features. Two features $X$ and $Y$ can be correlated if $X$ causes $Y$ or if $Y$ causes $X$, or if both are caused independently by some other factor $Z$, and the correlation will no longer hold true if one of the cause-effect relationships is broken. It's also possible that $X$ are $Y$ simply appear to be correlated because the sample is too small. 
# 
# While this may seem obvious, computers can't differentiate between correlation and causation, and decisions based on automated system can often have major consequences on society, so it's important to study why automated systems lead to a given result. Determining cause-effect relationships requires human insight.

# ## Linear Regression using a Single Feature
# 
# We now know that the "smoker" and "age" columns have the strongest correlation with "charges". Let's try to find a way of estimating the value of "charges" using the value of "age" for non-smokers. First, let's create a data frame containing just the data for non-smokers.

# In[ ]:




