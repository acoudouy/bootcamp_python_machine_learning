import numpy as np
from sigmoid import sigmoid_

def vec_reg_logistic_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of three non-empty numpy.ndarray, without any for-loop. The three arrays must have compatible dimensions."""
    if isinstance(x, np.ndarray) == 1 and isinstance(y, np.ndarray) == 1 and isinstance(theta, np.ndarray) == 1:
        if isinstance(lambda_, (int, float)) == 1:
            if len(y) == len(x) and len(x[0]) == len(theta):
                y_pred = sigmoid_(np.dot(x, theta))
                temp = np.dot(x.T, (y_pred - y)) / len(x)
                res = np.dot(x.T, (y_pred - y)) / len(x) + np.dot(lambda_ / len(x), theta)
                res[0] = temp[0]
                return (res)
            else:
                print("vec_reg_log_gradient: error in size of x, y or theta")
        else:
            print("vec_reg_log_gradient: lamda is not a scalar")
    else:
        print("vec_reg_log_gradient: error in type of x, y or theta")

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

print(vec_reg_logistic_grad(Y, X, Z, 1))
print(vec_reg_logistic_grad(Y, X, Z, 0.5))
print(vec_reg_logistic_grad(Y, X, Z, 0.0))
