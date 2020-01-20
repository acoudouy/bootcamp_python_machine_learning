import numpy as np

def linear_mse(x,y,theta):
    if isinstance(x, np.ndarray) == 1 and isinstance(y, np.ndarray) == 1 and isinstance(theta, np.ndarray) == 1:
        if len(x) == len(y):
            if len(x[0]) == len(theta):
                res = 0.0
                for i in range(len(x)):
                    res += (np.dot(x[i], theta) - y[i]) ** 2
                res = res / (i + 1)
                return (res)
            else:
                print("x's columns is not equal to theta's lines")
        else:
            print("x and y do not have the same number of lines")
    else:
        print("At least one arg is not a np.ndarray")



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

print(linear_mse(X,Y,Z))
print()
W = np.array([0,0,0])
print(linear_mse(X,Y,W))
