import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("mydata.csv")
X = data.iloc[:,0]
Y = data.iloc[:,1]
#plt.scatter(X,Y)
#plt.show()

m=0
c=0
n=float(len(X))
alpha = 0.01
costlist = []
epochs =4
for i in range(epochs):
       Yi = m*X + c
       cost = (1/(2*n))*(sum(Y-Yi)**2)
       #print(cost)
       costlist.append(cost)
       m_temp = (-1/n)*sum(X*(Y-Yi))
       c_temp = (-1/n)*(sum(Y-Yi))
       m=m-alpha*m_temp
       c=c-alpha*c_temp
print(m,c)
Yi = m*X + c
#plt.scatter(X,Y)
plt.plot(list(range(epochs)),costlist)
#plt.plot([min(X),max(X)],[min(Yi),max(Yi)])
plt.show()


