# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:29:15 2019

@author: Huong Pham
"""

# Batch imports of text processing libraries
import scipy as sp
import nltk
import string
global string
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd # Import pandas library
# Import the clean csv file 
path = r'C:\Users\Huong Pham\Documents\Graduate School\Winter 2019\Independent Study\\'
data = 'new_hotels.csv'  
df= pd.read_csv(path+data)
print (df.shape)

# create a new DataFrame that only contains the rounded numbers of Reviewer Scores
df1 = df[(df.Reviewer_Score==3) | (df.Reviewer_Score==4) | (df.Reviewer_Score==5) | (df.Reviewer_Score==6) | (df.Reviewer_Score==7) | (df.Reviewer_Score==8) | (df.Reviewer_Score==9) | (df.Reviewer_Score==10)] 
print df1.shape
list(df1.columns.values)

# Figure 1. Counts of Hotel Locations
plt.figure(figsize=(15,7))
plt.subplot(121)
toptrader_imp = df1.Country.value_counts(normalize=True)
toptrader_imp.head(30).plot(kind='bar', fontsize=10)
plt.title('Hotel Locations', fontsize=15)
plt.xticks(rotation=0)

# Figure 2. Reviewer Score by Number of Reviewers
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
plt.style.use('fivethirtyeight')
plt.figure(figsize=(10,7))
plt.hist(df1['Reviewer_Score'],bins=20)
plt.ylabel('Number_Reviewers',fontsize=16)
plt.xlabel('Reviewer Score',fontsize=16)
plt.title('Reviewer Score accross Users',fontsize=16)
plt.axvline(df1['Reviewer_Score'].mean(), color='k', linestyle='dashed', linewidth=1)
plt.savefig('Ratings_user.png')


# Revier Score by Hotels
plt.figure(figsize=(10,7))
plt.hist(df1['Average_Score'],bins=20)
plt.ylabel('Number Hotels',fontsize=16)
plt.xlabel('Average Score',fontsize=16)
plt.title('Average Score accross Hotels',fontsize=16)
plt.axvline(df1['Average_Score'].mean(), color='k', linestyle='dashed', linewidth=1)
plt.savefig('Ratings_hotel.png')

# Figure 4. Histogram of Text Legnth Distribution
df1['text length'] = df1['Review'].apply(len)
df.head()
g = sns.FacetGrid(data=df1, col='Reviewer_Score')
g.map(plt.hist, 'text length', bins=50)

# Boxplot of text legnth for each score
sns.boxplot(x='Reviewer_Score', y='text length', data=df1)
