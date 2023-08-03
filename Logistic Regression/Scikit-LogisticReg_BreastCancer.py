import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

data = pd.read_csv("brca.csv")
L=[]
for i in range(1,31):
       L.append(i)
X = data.iloc[:,L].values #importing features
Y = data.iloc[:,31].values #importing Y
Y = np.where(Y == 'B',0,1) #if y is b, set it to 0 else 1
X_normalised = (X - np.mean(X,axis=0))/(np.std(X,axis=0))
X_train, X_test, Y_train, Y_test = train_test_split(X_normalised,Y)
logreg = LogisticRegression()
logreg.fit(X_train,Y_train)
Y_pred = logreg.predict(X_test)

print("Accuracy: ",accuracy_score(Y_test,Y_pred)*100,"%")
cnf_matrix = metrics.confusion_matrix(Y_test,Y_pred)
print(cnf_matrix)