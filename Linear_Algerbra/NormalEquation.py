import numpy as np
import matplotlib.pyplot as plt

#data
x = np.array([1,2,3,4,5])
y = np.array([2,4,5,4,5])

X = np.c_[np.ones(x.shape[0]), x]

theta = np.linalg.inv(X.T @ X) @ X.T @ y
print("Theta", theta)

y_pred = X @ theta

plt.scatter(x, y, color='red', label='Data points')
plt.plot(x, y_pred, color='blue', label='Fitted line')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
