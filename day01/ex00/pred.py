import numpy as np

def predict_(theta, X):
    if isinstance(theta, np.ndarray) == 1 and isinstance(theta, np.ndarray) == 1:
        if len(X[0]) == len(theta) - 1:
            new = np.full((len(X),1),1)
            X = np.hstack([new, X])
            return (X.dot(theta))
        else:
            print("\nx's columns is not theta's line - 1. \n")
    else:
        print("theta or X is not a np.ndarray. Incompatible.\n")

X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
theta1 = np.array([[2.], [4.]])
b = predict_(theta1, X1)
print (b)

X2 = np.array([[1], [2], [3], [5], [8]])
theta2 = np.array([[2.]])
b = predict_(theta2, X2)
print (b)

X3 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8.,
80.]])
theta3 = np.array([[0.05], [1.], [1.], [1.]])
b = predict_(theta3, X3)
print (b)
