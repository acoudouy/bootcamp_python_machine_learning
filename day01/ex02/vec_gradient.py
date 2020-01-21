import numpy as np

def gradient(x,y,theta):
    if isinstance(y, np.ndarray) == 1 and isinstance(x, np.ndarray) == 1 and isinstance(theta, np.ndarray) == 1:
        if len(x) == len(y):
            if len(x[0]) == len(theta):
                res = np.dot(x.T, (x.dot(theta) - y)) / len(x)
                return (res)
            else:
                print("x's columns is not equel to theta's lines")
        else:
            print("x and y do not have the same number of lines")
    else:
        print("At least one argument is not a np.ndaray")
