# The data (X1, X2, X3) are for each patient
# X1 = systolic blood pressure
# X2 = age in years
# X3 = weight in pounds

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Create Pandas Data Frame which just a 2D matrix
df = pd.read_excel('mlr02.xls')
X = df.as_matrix()
print(df)

plt.scatter(X[:,1], X[:,0])
plt.show()

plt.scatter(X[:,2], X[:,0])
plt.show()

# Add a column of 1's called 'ones'
df['ones'] = 1
print(df)

# Extract column X1 into Data Frame Y
Y = df['X1']
print(Y)

# Create Data Frame X with Columns X2, X3, ones
X = df[['X2', 'X3', 'ones']]
print(X)

X2only = df[['X2', 'ones']]
X3only = df[['X3', 'ones']]

def get_r2(X,Y):
    w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
    Yhat = X.dot(w)

    d1 = Y - Yhat
    d2 = Y - Y.mean()
    r2 = 1 - d1.dot(d1) / d2.dot(d2)
    return r2

print("r2 for X2 only:", get_r2(X2only, Y))
print("r2 for x3 only:", get_r2(X3only, Y))
print("r2 for both:", get_r2(X,Y))


