
# coding: utf-8

# In[1]:

import pandas as pd
import csv
import numpy as np
from sklearn import preprocessing
from sklearn import tree
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
plt.rc("font", size=14)
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


# In[21]:

df_train = pd.read_csv('Train.csv')
df_test = pd.read_csv('Test.csv')


# In[32]:

# DROP COLUMN 1
df_test.drop(['city'], axis=1, inplace=True)
df_train.drop(['city'], axis=1, inplace=True)

# DROP COLUMN 2
df_test.drop(['dependents'], axis=1, inplace=True)
df_train.drop(['dependents'], axis=1, inplace=True)

# DROP COLUMN 3
df_test.drop(['Charges_cnt_PrevQ1_N'], axis=1, inplace=True)
df_train.drop(['Charges_cnt_PrevQ1_N'], axis=1, inplace=True)

# DROP COLUMN 4
df_test.drop(['FRX_PrevQ1_N'], axis=1, inplace=True)
df_train.drop(['FRX_PrevQ1_N'], axis=1, inplace=True)

# DROP COLUMN 5
df_test.drop(['zip'], axis=1, inplace=True)
df_train.drop(['zip'], axis=1, inplace=True)

# DROP COLUMN 6
df_test.drop(['ATM_C_prev1'], axis=1, inplace=True)
df_train.drop(['ATM_C_prev1'], axis=1, inplace=True)

# DROP COLUMN 7
df_test.drop(['ATM_D_prev1'], axis=1, inplace=True)
df_train.drop(['ATM_D_prev1'], axis=1, inplace=True)

# DROP COLUMN 9
df_test.drop(['Billpay_Active_PrevQ1_N'], axis=1, inplace=True)
df_train.drop(['Billpay_Active_PrevQ1_N'], axis=1, inplace=True)

# DROP COLUMN 10
df_test.drop(['Billpay_Reg_ason_Prev1_N'], axis=1, inplace=True)
df_train.drop(['Billpay_Reg_ason_Prev1_N'], axis=1, inplace=True)

# DROP COLUMN 10
df_test.drop(['vintage'], axis=1, inplace=True)
df_train.drop(['vintage'], axis=1, inplace=True)


# In[46]:

# REPLACE VALUE 1
df_train['OCCUP_ALL_NEW'] = df_train['OCCUP_ALL_NEW'].map({'HOUSEWIFE': 1, 'SALARIED': 2, 'SELF_EMPLOYED': 3, 'INDIVIDUAL': 4, 'RETIRED': 5, 'STUDENT': 6, 'NON_INDIVIDUA': 7, 'MISSING': 0})
df_test['OCCUP_ALL_NEW'] = df_test['OCCUP_ALL_NEW'].map({'HOUSEWIFE': 1, 'SALARIED': 2, 'SELF_EMPLOYED': 3, 'INDIVIDUAL': 4, 'RETIRED': 5, 'STUDENT': 6, 'NON_INDIVIDUA': 7, 'MISSING': 0})

df_train['OCCUP_ALL_NEW'].fillna(0, inplace=True)
df_test['OCCUP_ALL_NEW'].fillna(0, inplace=True)

# REPLACE VALUE 2
df_train['HNW_CATEGORY'] = df_train['HNW_CATEGORY'].map({'1_Imperia': 1, '2_Preferred': 2, '3_Classic': 3})
df_test['HNW_CATEGORY'] = df_test['HNW_CATEGORY'].map({'1_Imperia': 1, '2_Preferred': 2, '3_Classic': 3})

# REPLACE VALUE 3
df_train['FINAL_WORTH_prev1'] = df_train['FINAL_WORTH_prev1'].map({'HIGH': 1, 'MEDIUM': 2, 'LOW': 3})
df_test['FINAL_WORTH_prev1'] = df_test['FINAL_WORTH_prev1'].map({'HIGH': 1, 'MEDIUM': 2, 'LOW': 3})

df_train['FINAL_WORTH_prev1'].fillna(1, inplace=True)
df_test['FINAL_WORTH_prev1'].fillna(1, inplace=True)

# REPLACE VALUE 4
df_train['ENGAGEMENT_TAG_prev1'] = df_train['ENGAGEMENT_TAG_prev1'].map({'HIGH': 3, 'MEDIUM': 2, 'LOW': 1, 'NO': 0})
df_test['ENGAGEMENT_TAG_prev1'] = df_test['ENGAGEMENT_TAG_prev1'].map({'HIGH': 3, 'MEDIUM': 2, 'LOW': 1, 'NO': 0})

df_train['ENGAGEMENT_TAG_prev1'].fillna(1, inplace=True)
df_test['ENGAGEMENT_TAG_prev1'].fillna(1, inplace=True)

# REPLACE VALUE 5
df_train['RBI_Class_Audit'] = df_train['RBI_Class_Audit'].map({'SEMI-URBAN': 1, 'METROPOLITAN': 2, 'URBAN': 3, 'RURAL': 4})
df_test['RBI_Class_Audit'] = df_test['RBI_Class_Audit'].map({'SEMI-URBAN': 1, 'METROPOLITAN': 2, 'URBAN': 3, 'RURAL': 4})

# REPLACE VALUE 6
df_train['gender_bin'] = df_train['gender_bin'].map({'Male': 1, 'Female': 2, 'Missin': 0})
df_test['gender_bin'] = df_test['gender_bin'].map({'Male': 1, 'Female': 2, 'Missin': 0})

# REPLACE VALUE 7
df_train['Recency_of_BRANCH_TXN'].fillna(df_train['Recency_of_BRANCH_TXN'].median(), inplace=True)
df_test['Recency_of_BRANCH_TXN'].fillna(df_test['Recency_of_BRANCH_TXN'].median(), inplace=True)

# REPLACE VALUE 8
df_train['Recency_of_DR_TXN'].fillna(df_train['Recency_of_DR_TXN'].median(), inplace=True)
df_test['Recency_of_DR_TXN'].fillna(df_test['Recency_of_DR_TXN'].median(), inplace=True)

# REPLACE VALUE 9
df_train['Recency_of_CR_TXN'].fillna(df_train['Recency_of_CR_TXN'].median(), inplace=True)
df_test['Recency_of_CR_TXN'].fillna(df_test['Recency_of_CR_TXN'].median(), inplace=True)

# REPLACE VALUE 10
df_train['Recency_of_DR_TXN'].fillna(df_train['Recency_of_DR_TXN'].median(), inplace=True)
df_test['Recency_of_DR_TXN'].fillna(df_test['Recency_of_DR_TXN'].median(), inplace=True)

# REPLACE VALUE 11
df_train['Recency_of_Activity'].fillna(df_train['Recency_of_Activity'].median(), inplace=True)
df_test['Recency_of_Activity'].fillna(df_test['Recency_of_Activity'].median(), inplace=True)


# In[64]:

df_train = pd.read_csv('Train_clean.csv')
df_test = pd.read_csv('Test_clean.csv')

for column in df_train:
    status = df_train[column].isnull().sum() > 0
    if status:
        df_train[column].fillna(df_train[column].median(), inplace=True)
        if column != 'Responders':
            df_test[column].fillna(df_test[column].median(), inplace=True)


# In[67]:

X_train = df_train.iloc[:,3:-1]
y_train = df_train.iloc[:,-1]
X_test = df_test.iloc[:,3:]
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

classifier = tree.DecisionTreeClassifier(min_samples_split=300, min_samples_leaf=20)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
np.savetxt("hasil.csv", y_pred, delimiter=",", fmt="%i")
#print("\n")
#print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))
#print("\n")
#print(classification_report(y_test, y_pred))

