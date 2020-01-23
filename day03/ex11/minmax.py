import numpy as np

def minmax(x):
    if isinstance(x, np.ndarray) == 1:
        y = x
        mn = np.amin(x)
        mx = np.amax(x)
        for i in range(len(x)):
            y[i] = (x[i] - mn) / (mx - mn)
        return (y)
    else:
        print("x not a np.ndarray")

X = np.array([0., 15., -9., 7., 12., 3., -21.])
print(minmax(X))

Y = np.array([2., 14., -13., 5., 12., 4., -19.])
print(minmax(Y))
