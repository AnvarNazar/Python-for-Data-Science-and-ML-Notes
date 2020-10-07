#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression


# Read training data
df = pd.read_csv('train2-14.csv')
df.head()

X_train = df[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9',
       'X10', 'X11', 'X12']]
y_train = df['X..language']

sns.heatmap(X_train.dropna(how='any'), yticklabels=False, cbar=False, cmap='viridis')

X_train.mean()

# Fill nan values with column mean
X_train = X_train.apply(lambda x: x.fillna(x.mean()))

lm = LogisticRegression()
lm.fit(X_train, y_train)

X_test = pd.read_csv('test2-14.csv')
X_test.info()

X_test.rename(columns={'X..X1': 'X1'}, inplace=True)

X_test = X_test.apply(lambda x: x.fillna(x.mean()))

y_pred = lm.predict(X_train)
y_test = lm.predict(X_test)

# write y_test to csv file
np.savetxt('question2.csv', y_test)

from sklearn.metrics import accuracy_score

print('Accuracy score on training data: ', accuracy_score(y_train, y_pred))

