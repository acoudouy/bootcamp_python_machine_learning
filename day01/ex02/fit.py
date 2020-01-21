import numpy as np
from vec_gradient import gradient
from pred import predict_

def fit_(theta, X, Y, alpha, n_cycle):
    if isinstance(X, np.ndarray) == 1 and isinstance(theta, np.ndarray) == 1 and isinstance(Y, np.ndarray) == 1:
        if len(X[0]) == len(theta) - 1 and len(X) == len(Y):
            if isinstance(alpha, float) == 1 and isinstance(n_cycle, int) == 1:
                new = np.full((len(X),1),1.)
                X = np.hstack([new, X])
                for i in range(n_cycle):
                    theta = theta - alpha * gradient(X,Y,theta)
                return (theta)
            else:
                print("alpha is not a float or n_cycle not an int")
        else:
            print("\nx's columns is not theta's line - 1 or dim(X) and dim(Y) different \n")
    else:
        print("theta or X is not a np.ndarray. Incompatible.\n")

X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
Y1 = np.array([[2.], [6.], [10.], [14.], [18.]])
theta1 = np.array([[1.], [1.]])
theta1 = fit_(theta1, X1, Y1, alpha=0.01, n_cycle=2000)
print(theta1)
print(predict_(theta1,X1))
print()
X2 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8.,
80.]])
Y2 = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
theta2 = np.array([[42.], [1.], [1.], [1.]])
theta2 = fit_(theta2, X2, Y2, alpha = 0.0005, n_cycle=42000)
print(theta2)
print(predict_(theta2, X2))
