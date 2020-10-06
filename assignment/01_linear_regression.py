#!/usr/bin/env python
# coding: utf-8
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression


# Load training data
df = pd.read_csv("train1-24.csv")
X_train = df[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12']]
y_train = df['X..language']


# Create linear regression model
linearModel = LinearRegression()
linearModel.fit(X_train, y_train)


# Load data for testing
X_test = pd.read_csv('test1-24.csv')
y_test = pd.read_csv('labels1-24.csv')
X_test.rename(columns={'X..X1': 'X1'}, inplace=True)


# Run the model
predictions = linearModel.predict(X_test)


# Save output to question1.csv
np.savetxt('question1.csv', predictions, fmt='%.18e', delimiter=' ',
           newline='\n', header='', footer='', comments='#', encoding=None)

# Calculate accuracy
accuracy_score(y_test['X..language'], predictions.round())

