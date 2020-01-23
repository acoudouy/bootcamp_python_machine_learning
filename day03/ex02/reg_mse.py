import numpy as np

def reg_mse(x, y, theta, lamda_):
    """Computes the regularized mean squared error of three non-empty numpy.ndarray, without any for-loop. The three arrays must have compatible dimensions."""
    if isinstance(y, np.ndarray) == 1 and isinstance(x, np.ndarray) == 1:
        if isinstance(theta, np.ndarray) == 1 and isinstance(lamda_, (int, float)) == 1:
            if len(y) == len(x) and len(x[0]) == len(theta):
                xdt_y = np.dot(x, theta) - y
                res = (1 / len(x)) * (np.dot((xdt_y).T,xdt_y) + lamda_ * np.dot(theta, theta))
                return (res)

            else:
                print("reg_mse: problem with size of y, x or theta")
        else:
            print("reg_mse: problem with theta or lamda type")
    else:
        print("reg_mse: problem with y or x type")


X = np.array([
 [ -6, -7, -9],
 [ 13, -2, 14],
 [ -7, 14, -1],
 [ -8, -4, 6],
 [ -5, -9, 6],
 [ 1, -5, 11],
 [ 9, -11, 8]])
Y = np.array([2, 14, -13, 5, 12, 4, -19])
Z = np.array([3,0.5,-6])
print(reg_mse(X, Y, Z, 0))
print(reg_mse(X, Y, Z, 0.1))
print(reg_mse(X, Y, Z, 0.5))
