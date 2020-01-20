import numpy as np

def dot(x,y):
    res = 0.0
    if isinstance(x, np.ndarray) == 1 and isinstance(y, np.ndarray) == 1:
        if len(x) == len(y):
            for i in range(len(x)):
                res += x[i] * y[i]
            return (res)
        else:
            print("The 2 args do not have the same dimension")
            return (None)
    else:
        print("The 2 args are not np.ndarray")
        return (None)


X = np.array([0, 15, -9, 7, 12, 3, -21])
Y = np.array([2, 14, -13, 5, 12, 4, -19])

R = dot(Y,Y)
print(R)
