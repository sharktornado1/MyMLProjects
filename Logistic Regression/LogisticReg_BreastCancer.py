import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics

data = pd.read_csv("brca.csv")
L=[]
for i in range(1,31):
       L.append(i)
X = data.iloc[:,L].values #importing features
Y = data.iloc[:,31].values #importing Y
Y = np.where(Y == 'B',0,1) #if y is b, set it to 0 else 1

X_normalised = (X - np.mean(X,axis=0))/(np.std(X,axis=0)) #Z-score Normalisation
X_train, X_test, Y_train, Y_test = train_test_split(X_normalised,Y) #splitting into training and testing data
m = np.zeros(X_normalised.shape[1]) #making a numpy array of 0s for m
c=0 #initialising c
n = len(Y) #no of features
costlist=[]
alpha = 0.005
epochs = 4000

for i in range(epochs): 
       f = 1/(1+np.exp(-1*(np.dot(X_train,m)+c))) #f(x) for logisitc regression
       cost = (-1/n)*np.sum(np.log(f)*Y_train + np.log(1-f)*(1-Y_train),axis=0) #J(Cost) calculation
       costlist.append(cost) #appending it to a list so that we can plot it later
       for j in range(len(m)): #updating values of m, i couldnt do it through vectorisation idk why
              gradient = np.mean(np.multiply(f-Y_train, X_train[:, j]))
              m[j] = m[j] - alpha * gradient
       c_temp = (-1/n)*(np.sum(Y_train-f,axis=0)) #updating c value
       c=c-alpha*c_temp
plt.plot(list(range(epochs)),costlist) #plotting cost function
plt.show()
temp = np.dot(X_test,m)+c
f_pred = 1/(1+np.exp(-1*temp)) #getting probabilities(f(X)) that the it is M or B
Y_pred = []
for i in f_pred:
       if i >= 0.5: #if the probability is >0.5 then set it to 1 or M
              Y_pred.append(1)
       elif i<0.5: #if the probability is <0.5 then set it to 0 or B
              Y_pred.append(0)
Y_pred = np.array(Y_pred) #turning into numpy array
correct=0
wrong=0
print("Actual results: ")
print(Y_test)
print("Prediciton results: ")
print(Y_pred)
for i in range(len(Y_test)): #traversing through the Y_pred array and Y_test array to see if they match
       if Y_pred[i]==Y_test[i]:
              correct=correct+1 #if they match increment correct counter
       else:
              wrong = wrong +1 #else increment wrong counter
total = correct + wrong #total number of cases
print("Correct: ",correct)
print("Wrong: ",wrong)
print("Total: ",total)
print("Accuracy: ",accuracy_score(Y_test,Y_pred)*100,"%") #testing accuracy percentage using scikit method

cnf_matrix = metrics.confusion_matrix(Y_test,Y_pred)
print(cnf_matrix)