import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []
for line in open('data_1d.csv'):
    x,y = line.split(',')
    X.append(float(x))
    Y.append(float(y))

# Turn X and Y into numpy arrays
X = np.array(X)
Y = np.array(Y)

# Plot it to see what it looks like
#plt.scatter(X,Y)
#plt.show()

# Calculate a and b
# Note: X.dot(X) is the sum of the products of the corresponding entries of the two sequences of numbers
denominator = X.dot(X) - X.mean() * X.sum()

print("X.dot(X) - %f X.mean() - %f X.sum() - %f" % (X.dot(X), X.mean(), X.sum()))

a = (X.dot(Y) - Y.mean() * X.sum()) / denominator
b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denominator

# calculate the predicted Y i.e. Algebraic Equation for a Line y = mx + b
Yhat = a*X + b
print(Yhat)

# Plot Everything
plt.scatter(X,Y)
plt.plot(X,Yhat)
plt.show()

# Calculate R-squared to measure the accuracy of the model. Should be between 0 and 1, with 1 being 100% accurate.
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)

print("Accuracy - %f" % r2)