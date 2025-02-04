import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('home.txt',names=["size","bedroom","price"])

#print(data.head())

data = (data - data.mean())/data.std()
#print(data.head())

X = data.iloc[:,0:2]
ones = np.ones([X.shape[0],1])
X = np.concatenate((ones,X),axis=1)
y = data.iloc[:,2:3].values
theta = np.zeros([1,3])

alpha = 0.01
iters = 1000

def computeCost(X,y,theta):
    tobesummed = np.power(((X @ theta.T)-y),2)
    return np.sum(tobesummed)/(2 * len(X))

#print(computeCost(X,y,theta))

def gradientDescent(X, y, theta, iters, alpha):
    cost = np.zeros(iters)
    for i in range(iters):
        theta = theta - (alpha / len(X)) * np.sum(X * (X @ theta.T - y), axis=0)
        cost[i] = computeCost(X, y, theta)

    return theta, cost

g,cost = gradientDescent(X,y,theta,iters,alpha)
print(g)

finalCost = computeCost(X,y,g)
print(finalCost)

fig, ax = plt.subplots()
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()
