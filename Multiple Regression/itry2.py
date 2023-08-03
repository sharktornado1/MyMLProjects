import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
data = pd.read_csv("housing.csv")
data = data.dropna()
X = data.iloc[:,[1,2,3,4,5,6,7]].values
Y = data.iloc[:,8].values

m = np.array([0,0,0,0,0,0,0])
c= 0
#Performing feature scaling
X_normalised = (X - np.mean(X,axis=0))/(np.std(X,axis=0)) #Applying Z Score Normalisation, axis=0 means doing the operation column wise
X_train, X_test, Y_train, Y_test = train_test_split(X_normalised,Y)
n = float(len(Y))
costlist=[]
alpha = 0.1
epochs = 1000
for i in range(epochs):
       Yi = np.dot(X_train,m) + c
       #print(Yi)
       cost = (1/(2*n))*(np.sum(Y_train-Yi,axis=0)**2)
       costlist.append(cost)
       for j in range(X_train.shape[1]):
              weightCost = (np.multiply(np.subtract(Yi,Y_train),X_train[:,j])).mean()
              m[j] = m[j] - (alpha*weightCost)
              
       #m_temp = (-1/n)*np.sum((Y_train-Yi)*X_train,axis=0)
       c_temp = (-1/n)*(np.sum(Y_train-Yi,axis=0))
       #m=m-alpha*m_temp
       c=c-alpha*c_temp
plt.plot(list(range(epochs)),costlist)
plt.show()
print(m)
print(c)
Ypred = np.dot(X_test,m)+c
r2 = r2_score(Y_test,Ypred)
print("R-squared accuracy: ",r2*100,"%")
print(np.dot(X_normalised[0],m)+c)
