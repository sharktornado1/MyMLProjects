import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn import preprocessing
from sklearn.metrics import r2_score

data = pd.read_csv("Real estate.csv")
X = data.iloc[:,[1,2,3,4,5,6]].values
Y = data.iloc[:,7].values

X_train,X_test,Y_train,Y_test = train_test_split(X,Y)
model = LinearRegression()
model.fit(X_train,Y_train)
predictions = model.predict(X_test)
r2 = r2_score(Y_test,predictions)
print("R-squared accuracy: ",r2*100,"%")
print('mean_squared_error : ', mean_squared_error(Y_test, predictions))
print('mean_absolute_error : ', mean_absolute_error(Y_test, predictions))