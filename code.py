# -*- coding: utf-8 -*-
"""
Created on Fri Nov 03 19:43:56 2017

@author: Dzaky
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

df_train = pd.read_csv('D:\AnalythicVidhya Test\Train.csv')
df_test = pd.read_csv('D:\AnalythicVidhya Test\Test.csv')

for column in df_train:
    status = df_train[column].isnull().values.any()
    if status:
        df_train.drop(column, axis=1, inplace=True)
        
for column in df_test:
    status = df_test[column].isnull().values.any()
    if status:
        df_test.drop(column, axis=1, inplace=True)

X_train = df_train.iloc[:,144:172]
y_train = df_train.iloc[:,-1]
X_test = df_test.iloc[:,144:172]
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(y_pred.shape)
print(y_pred)
#print("\n")
#print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))
#print("\n")
#print(classification_report(y_test, y_pred))