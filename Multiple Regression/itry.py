import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error,mean_squared_error
data = pd.read_csv("Real estate.csv")
X = data.iloc[:,[1,2,3,4,5,6]].values
Y = data.iloc[:,7].values
m = np.array([0,0,0,0,0,0])
c= 0
#Performing feature scaling
X_normalised = (X - np.mean(X,axis=0))/(np.std(X,axis=0)) #Applying Z Score Normalisation, axis=0 means doing the operation column wise

n = float(len(Y))
costlist=[]
alpha = 0.5
epochs = 20
for i in range(epochs):
       Yi = np.dot(X_normalised,m) + c
       #print(Yi.shape)
       #print(Yi)
       cost = (1/(2*n))*(np.sum(Y-Yi,axis=0)**2)
       costlist.append(cost)
       m_temp = (-1/n)*np.sum((Y-Yi).reshape(-1,1)*X_normalised,axis=0)
       c_temp = (-1/n)*(np.sum(Y-Yi,axis=0))
       m=m-alpha*m_temp
       c=c-alpha*c_temp
plt.plot(list(range(epochs)),costlist)
plt.show()
print(m)
print(c)
Ypred = np.dot(X_normalised,m)+c
r2 = r2_score(Y,Ypred)
print("R-squared accuracy: ",r2*100,"%")
print('mean_squared_error : ', mean_squared_error(Y, Ypred))
print('mean_absolute_error : ', mean_absolute_error(Y, Ypred))
