
# Batch imports of text processing libraries
import numpy as np
import scipy as sp
import nltk
import string
global string
import scipy.sparse as sp

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm

import pandas as pd #Import pandas library
#import the clean csv file 
path = r'C:\Users\Huong Pham\Documents\Graduate School\Winter 2019\4 classes\\'
data = 'clean_data.csv'
df2= pd.read_csv(path+data)
print (df2.shape) #Dataframe df1 contains only reviewer score 3 & 10
print df2.head()

from textblob import TextBlob
from textblob import Word
from nltk.stem.snowball import SnowballStemmer

# Write a function to create sentimental scores for the Reviews
def sentiment_func(review):
    try:
        return TextBlob(review).sentiment.polarity
    except:
       return None

# Apply the "sentiment_func" function. Add a new column 'Sentiment' to the data frame for the polarity scores    
df2['Sentiment'] = df2['Review'].apply(sentiment_func)
print df2['Sentiment']

# box plot of sentiment grouped by stars
df2.boxplot(column='Sentiment', by='Reviewer_Score')

# reviews with most positive sentiment
df2[df2.Sentiment == 1].Review.head()

# reviews with most negative sentiment
df2[df2.Sentiment == -1].Review.head()

# reviews with most negative sentiment
df2[df2.Sentiment == 0].Review.head()

# remove index column
#print df2.shape
df2 = df2.reset_index()
df2=df2.drop(['index'],axis=1)
print df2.shape

# repalace inf by NaN values
df2.replace([np.inf, -np.inf], np.nan) # This code doesn't seem to work. I still get error when I run my models

# Let's replace all Nan values with 1234, and export to a new CSV file to examine each 1234 value now
df_cleaned = df2.fillna(1234)
print df_cleaned.shape  

# Export to CSV file
import os
path_d = r'C:\Users\Huong Pham\Documents\Graduate School\Winter 2019\4 classes\\'
df_cleaned.to_csv(os.path.join(path_d,'testing.csv')) # Cleaned up the NaN values in Excel, then renamed 'testing' to 'clean_data_1'

# Re-import the clean csv file 
import pandas as pd #Import pandas library
path = r'C:\Users\Huong Pham\Documents\Graduate School\Winter 2019\4 classes\\'
data = 'clean_data_1.csv'
df3= pd.read_csv(path+data)
print (df3.shape) 
list(df3.columns.values)

# CREATE LABEL VECTOR AND PREDICTORS
# Create the outcome/lable vector (y)
y = df3['Reviewer_Score']
print y.head()

# Creating X which is the selected attributes as dummy data because jupyter get read nominal attributes
feature_cols = ['Additional_Number_of_Scoring','Sentiment', 'Reviewer_Nationality', 'Country', 'Average_Score', 'Total_Number_of_Reviews']
feat = df3[feature_cols]
X = feat
df3 = pd.get_dummies(feat)
X = df3

df3 = df3.reset_index(drop=True)
print df3.shape
list(df3.columns.values)

#Define test_train data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Before splitting
print X.shape

# After splitting
print X_train.shape
print X_test.shape

# KNN MODEL
'''
Step 1: decide what an appropriate "N" is for our model. This was determined to be "6" based on running through options of "N" and comparing accuracy.
'''
#import the class
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt
#Now, let's iterate through potential values of K to find an optimal value for our KNN model
k_range = range(1,10)

models = []

for k in k_range: 
    knn = KNeighborsClassifier(n_neighbors=k)
    k_scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    models.append(k_scores.mean())
print(models)

plt.plot(k_range, models)
plt.xlabel('KNN Value')
plt.ylabel('Accuracy Score')

'''
Step 2: Run KNN with Test Train Split
Now that we know we want to us 6 as our KNN value, we can run the KNN Model using Test, Train, Split on our data and print the confusion matrix.
'''
#import the class
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split #import the model
'''
Step 2a: create test, train datasets
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=99)

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)

'''
Step 2b: test the model on the testing dataset and evaluate how accurate the model is,
#based on the model trained on the training dataset 
'''
from sklearn import metrics
y_pred_class = knn.predict(X_test)
print ("KNN Score Accurancy Score: ",metrics.accuracy_score(y_test, y_pred_class))
print ("KNN Confusion Matrix: ",metrics.confusion_matrix(y_test, y_pred_class))

'''
Step 3: Run KNN with Test Cross Validation
'''
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

knn = KNeighborsClassifier(n_neighbors=10)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy') #cv is the cross-validation parameter
print ("The average score : " , scores.mean())
print ("KNN Score Accurancy Score: ",metrics.accuracy_score(y_test, y_pred_class))

# Evaluate the accuracy score of KNN model 
from sklearn.metrics import confusion_matrix, classification_report
KNN_confusion = metrics.confusion_matrix(y_test, y_pred_class)
print KNN_confusion
print("Confusion Matrix", classification_report(y_test, y_pred_class))

# save confusion matrix and slice into four pieces - KNN Model
KNN_TP = KNN_confusion[1][1]
KNN_TN = KNN_confusion[0][0]
KNN_FP = KNN_confusion[0][1]
KNN_FN = KNN_confusion[1][0]   

print 'True Positives:', KNN_TP
print 'True Negatives:', KNN_TN
print 'False Positives:',KNN_FP
print 'False Negatives:',KNN_FN

# Plot the confusion matrix for KNN Model
labels = ['3','5','7','8']
conf_m1 = metrics.confusion_matrix(y_test, y_pred_class)
print ("Confusion Matrix: ",conf_m1)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_m1)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# NAIVE BAY MODEL
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
GaussianNB()
from sklearn.model_selection import cross_val_score
scores = cross_val_score(nb, X,y, cv=2, scoring = 'accuracy')
print(scores).mean()

from sklearn.model_selection import train_test_split #import the model
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=99)
nb.fit(X_train, y_train)
y_pred_class = nb.predict(X_test)

# Print score, confusion matrix, classification report for Naive Bay model
from sklearn.metrics import confusion_matrix, classification_report
print ("NB Score Accurancy Score: ",metrics.accuracy_score(y_test, y_pred_class))
print ("NB Confusion Matrix: ",metrics.confusion_matrix(y_test, y_pred_class))
print('\n')
print("Confusion Matrix", classification_report(y_test, y_pred_class))

# save confusion matrix and slice into four pieces - Naive Bay Model
NB_confusion = metrics.confusion_matrix(y_test, y_pred_class)
NB_TP = NB_confusion[1][1]
NB_TN = NB_confusion[0][0]
NB_FP = NB_confusion[0][1]
NB_FN = NB_confusion[1][0]   

print 'True Positives:', NB_TP
print 'True Negatives:', NB_TN
print 'False Positives:',NB_FP
print 'False Negatives:',NB_FN

# LOGISTIC MODEL
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e9)
logreg.fit(X_train, y_train)       #Fit data on Logistic regression model
zip(feature_cols, logreg.coef_[0]) #calculate the degree of correlation

# calculate classification accuracy
from sklearn import metrics
l_pred_class = logreg.predict(X_test)  #predicted probabilities
print metrics.accuracy_score(y_test, l_pred_class)

# print confusion matrix for Logistic Regression
print metrics.confusion_matrix(y_test, l_pred_class)
print("Confusion Matrix", classification_report(y_test, l_pred_class))

# save confusion matrix and slice into four pieces
log_confusion = metrics.confusion_matrix(y_test, l_pred_class)
log_TP = log_confusion[1][1]
log_TN = log_confusion[0][0]
log_FP = log_confusion[0][1]
log_FN = log_confusion[1][0]   

print 'True Positives:', log_TP
print 'True Negatives:', log_TN
print 'False Positives:', log_FP
print 'False Negatives:', log_FN