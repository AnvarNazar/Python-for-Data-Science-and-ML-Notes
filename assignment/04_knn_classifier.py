#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

df_train = pd.read_csv('train3-035.csv')
df_test = pd.read_csv('test3-035.csv')

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(df_train.drop('X..language', axis=1))

scaled_features = scaler.transform(df_train.drop('X..language', axis=1))

df_train.head()

df_feat = pd.DataFrame(scaled_features, columns=df_train.columns[1:])

knn = KNeighborsClassifier(1)
knn.fit(df_feat, df_train['X..language'])

pred = knn.predict(df_feat)

print(classification_report(df_train['X..language'], pred))

df_test.rename(columns={'X..X1': 'X1'}, inplace=True)

scaler.fit(df_test)

df_test = pd.DataFrame(scaler.transform(df_test), columns=df_test.columns)

pred = knn.predict(df_test)

np.savetxt('question3.csv', pred)

