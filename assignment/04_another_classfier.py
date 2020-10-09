#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

from sklearn.metrics import classification_report
from sklearn.svm import SVC


df = pd.read_csv('train3-035.csv')


svcmodel = SVC()


X_train = df.drop('X..language', axis=1)
y_train = df['X..language']

svcmodel.fit(X_train, y_train)


X_test = pd.read_csv('test3-035.csv')


X_test.rename(columns={'X..X1':'X1'}, inplace=True)

pred = svcmodel.predict(X_train)


print(classification_report(y_train, pred))


y_pred = svcmodel.predict(X_test)


np.savetxt('question4.csv', y_pred)




