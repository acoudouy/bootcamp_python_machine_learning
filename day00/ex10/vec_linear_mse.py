import numpy as np

def vec_linear_mse(x,y,theta):
    if isinstance(y, np.ndarray) == 1 and isinstance(x, np.ndarray) == 1 and isinstance(theta, np.ndarray) == 1:
        if len(x) == len(y):
            if len(x[0]) == len(theta):
                res = (x.dot(theta) - y).T * (x.dot(theta) - y) / len(y)
                return(res.sum())
            else:
                print("x's columns is not equel to theta's lines")
        else:
            print("x and y do not have the same number of lines")
    else:
        print("At least one argument is not a np.ndaray")

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
W = np.array([0,0,0])
print(vec_linear_mse(X,Y,Z))
print()
print(vec_linear_mse(X,Y,W))
