import numpy as np

def vec_reg_linear_grad(x, y, theta, lambda_):
    """Computes the regularized linear gradient of three non-empty numpy.ndarray, with two for-loop. The three arrays must have compatible dimensions."""

    if isinstance(y, np.ndarray) == 1 and isinstance(x, np.ndarray) == 1 and isinstance(theta, np.ndarray) == 1:
        if isinstance(lambda_, (int, float)) == 1:
            if len(x) == len(y):
                if len(x[0]) == len(theta):
                    res = np.zeros(len(x[0]))
                    res = np.dot(x.T, (x.dot(theta) - y)) / len(x) + np.dot(lambda_/len(x), theta) ###
                    temp = np.dot(x.T, (x.dot(theta) - y)) / len(x) ###
                    res[0] = temp[0]
                    return (res)
                else:
                    print("vec_reg_lin_grad: x's columns is not equel to theta's lines")
            else:
                print("vec_reg_lin_grad: lamda not a float / int")
        else:
            print("vec_reg_lin_grad: x and y do not have the same number of lines")
    else:
        print("vec_reg_lin_grad: At least one argument is not a np.ndaray")

X = np.array([
 [ -6, -7, -9],
 [ 13, -2, 14],
 [ -7, 14, -1],
 [ -8, -4, 6],
 [ -5, -9, 6],
 [ 1, -5, 11],
 [ 9, -11, 8]])
Y = np.array([2, 14, -13, 5, 12, 4, -19])
Z = np.array([3,10.5,-6])
print(vec_reg_linear_grad(X, Y, Z, 1))

print(vec_reg_linear_grad(X, Y, Z, 0.1))

print(vec_reg_linear_grad(X, Y, Z, 0.0))
