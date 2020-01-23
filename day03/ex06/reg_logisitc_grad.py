import numpy as np
from sigmoid import sigmoid_

def reg_logistic_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of three non-empty numpy.ndarray, with two for-loop. The three arrays must have compatible dimensions."""

    if isinstance(y, np.ndarray) == 1 and isinstance(x, np.ndarray) == 1 and isinstance(theta, np.ndarray) == 1:
        if isinstance(lambda_, (int, float)) == 1:
            if len(x) == len(y):
                if len(x[0]) == len(theta):
                    y_pred = sigmoid_(np.dot(x, theta))
                    res = np.zeros(len(x[0]))
                    res[0] = sum((y_pred - y) * (x[:,0])) / len(x)
                    for j in range(1,len(theta)):
                        res[j] = sum((y_pred - y) * (x[:,j]) + lambda_ * theta[j]) / len(x)
                    return(res)
                else:
                    print("reg_log_grad: x's columns is not equel to theta's lines")
            else:
                print("reg_log_grad: lamda not a float / int")
        else:
            print("reg_log_grad: x and y do not have the same number of lines")
    else:
        print("reg_log_grad: At least one argument is not a np.ndaray")

X = np.array([
 [ -6, -7, -9],
 [ 13, -2, 14],
 [ -7, 14, -1],
 [ -8, -4, 6],
 [ -5, -9, 6],
 [ 1, -5, 11],
 [ 9, -11, 8]])
Y = np.array([1,0,1,1,1,0,0])
Z = np.array([1.2,0.5,-0.32])

print(reg_logistic_grad(Y, X, Z, 1))

print(reg_logistic_grad(Y, X, Z, 0.5))

print(reg_logistic_grad(Y, X, Z, 0.0))

