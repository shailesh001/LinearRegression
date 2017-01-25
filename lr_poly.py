import numpy as np
import matplotlib.pyplot as plt

# load the data
X = []
Y = []
for line in open('data_poly.csv'):
    x, y = line.split(',')
    x = float(x)
    X.append([1, x, x*x])
    Y.append(float(y))

# convert to numpy arrays
X = np.array(X)
Y = np.array(Y)

# Do a quick plot
plt.scatter(X[:,1], Y)
plt.show()

# calculate weights
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X, w)

# plot it again
plt.scatter(X[:,1], Y)
plt.plot(sorted(X[:,1]), sorted(Yhat))
plt.show()

d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("r-squared:",r2)


