import numpy as np

class MyLinearRegression():
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """
    def __init__(self, theta):
        if isinstance(theta, np.ndarray) == 1:
           self.theta = theta 
        else:
            print("theta is not a np.ndarray")

    def predict_(self, X):
        if isinstance(self.theta, np.ndarray) == 1 and isinstance(X, np.ndarray) == 1:
            if len(X[0]) == len(self.theta) - 1:
                new = np.full((len(X),1),1)
                X = np.hstack([new, X])
                return (X.dot(self.theta))
            else:
                print("\nx's columns is not theta's line - 1. \n")
        else:
            print("theta or X is not a np.ndarray. Incompatible.\n")


    def cost_elem_(self, X, Y):
        if len(X[0]) == len(self.theta) - 1 and len(X) == len(Y):
            new = np.full((len(X),1),1.)
            X = np.hstack([new, X])
            J = np.full((len(X), 1),0.0)
            for i in range(len(X)):
                J[i] = ((np.dot(X[i], self.theta) - Y[i]) ** 2)
                J[i] = J[i] / (len(X) * 2)
            return (J)
        else:
            print("Incompatibily in X, Y and theta's dimensions.\n")

    def cost_(self, X, Y):
        if len(X[0]) == len(self.theta) - 1 and len(X) == len(Y):
            new = np.full((len(X),1),1)
            X = np.hstack([new, X])
            J = 0.0
            for i in range(len(X)):
                J += (np.dot(X[i], self.theta) - Y[i]) ** 2
            J = J / ((i + 1) * 2)
            return (float(J))
        else:
            print("Incompatibily in X, Y and theta's dimensions.\n")

    def gradient(self, x, y):
        if isinstance(y, np.ndarray) == 1 and isinstance(x, np.ndarray) == 1 and isinstance(self.theta, np.ndarray) == 1:
            if len(x) == len(y):
                if len(x[0]) == len(self.theta):
                    x1 = np.delete(x,0,1)
                    res = np.dot(x.T, (x.dot(self.theta) - y)) / len(x)
                    return (res)
                else:
                    print("x's columns is not equel to theta's lines")
            else:
                print("x and y do not have the same number of lines")
        else:
            print("At least one argument is not a np.ndaray")


    def fit_(self, X, Y, alpha, n_cycle):
        if isinstance(X, np.ndarray) == 1 and isinstance(self.theta, np.ndarray) == 1 and isinstance(Y, np.ndarray) == 1:
            if len(X[0]) == len(self.theta) - 1 and len(X) == len(Y):
                if isinstance(alpha, float) == 1 and isinstance(n_cycle, int) == 1:
                    new = np.full((len(X),1),1.)
                    X = np.hstack([new, X])
                    for i in range(n_cycle):
                        self.theta = self.theta - alpha * self.gradient(X,Y)
                    return (self.theta)
                else:
                    print("fit: alpha is not a float or n_cycle not an int")
            else:
                print("fit: x's columns is not theta's line - 1 or dim(X) and dim(Y) different \n")
        else:
            print("fit: theta or X is not a np.ndarray. Incompatible.\n")

    def mse_(self, y, y_hat):
        if isinstance(y, np.ndarray) == 1 and isinstance(y_hat, np.ndarray) == 1:
            if len(y) == len(y_hat):
                if y_hat.ndim == 2 and y.ndim == 2:
                    res = 0.0
                    for i in range(len(y)):
                        res += (y_hat[i][0] - y[i][0]) ** 2
                    res = res / (i + 1)
                    return (res)
                else:
                    print("2 args are not only composed with one column")
            else:
                print("2 args do not have the same size")
        else:
            print("2 args are not np.ndarray")
        return (None)
