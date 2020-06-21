import matplotlib.pyplot as plt
import numpy as np
data = np.genfromtxt('data.csv', delimiter=',')
X = data[:, 0].reshape(-1,1)
ones = np.ones([X.shape[0], 1])
X = np.concatenate([ones, X],1)
y = data[:, 1].reshape(-1,1)

plt.scatter(data[:, 0].reshape(-1,1), y)
plt.show()

alpha = 0.0001
iters = 1000
theta = np.array([[1.0, 1.0]])

def computeCost(X, y,theta):
    inner = np.power(((X @ theta.T) - y), 2)
    return np.sum(inner)/(2 * len(X))

#print(computeCost(X, y, theta))

def gradientDescent(X, y, theta, alpha, iters):
    for i in range(iters):
        theta = theta - (alpha/len(X)) * np.sum((X @ theta.T - y) * X, axis=0)
        cost = computeCost(X, y, theta)
        #debugging step
        #if i %10 == 0;
        #   print(cost)
    return (theta, cost)

g, cost = gradientDescent(X, y, theta, alpha, iters)
print(g, cost)


plt.scatter(data[:, 0].reshape(-1,1), y)
axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = g[0][0] + g[0][1]* x_vals #the line equation
plt.plot(x_vals, y_vals, '--')
plt.show()