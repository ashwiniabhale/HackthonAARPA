# Call the library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import xgboost as xgb
data = pd.read_csv('result.csv')
# Output Variable
y = data.result
# Input Variable
X = data.iloc[:,[2,3,4,5,6,7,8,9]].values

# Output is in form of a string (string is not used for calculation, convert it into numeric value)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Feature Scaling on input data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Develop Random Forest
from sklearn.ensemble import RandomForestClassifier
cls = RandomForestClassifier()
cls.fit(X_train, y_train)

'''
# Develop Random Forest
from sklearn.tree import DecisionTreeClassifier
cls = DecisionTreeClassifier()
cls.fit(X_train, y_train)
'''

# Finding out the Predicted value
#y_pred = cls.predict(X_test)
'''
# Finding out accuracy
from sklearn.metrics import accuracy_score
print("Accuracy of our model is :", accuracy_score(y_test, y_pred))
'''