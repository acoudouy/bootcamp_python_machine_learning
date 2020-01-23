import numpy as np
import math

def mean(x):
    res = 0.0
    if isinstance(x, np.ndarray) == 1:
        for i,el in enumerate(x):
            res += el
        res = res / len(x)
        return (res)
    else:
        print("x is not a np.ndarray")
        return (None)

def variance(x):
    res = 0.0
    if isinstance(x, np.ndarray) == 1:
        for i, el in enumerate(x):
            res += (el - mean(x)) ** 2
        res = res / (len(x))
        return (math.sqrt(res))
    else:
        print("x is not a np.ndarray")
        return (None)

def zscore(x):
    """Computes the normalized version of a non-empty numpy.ndarray using the z-score standardization."""
    if isinstance(x, np.ndarray) == 1:
        y = x
        moy = mean(x)
        var = variance(x)
        for i in range(len(x)):
            y[i] = (x[i] - moy) / (var)
        return (y)
    else:
        print("x is not a np.ndarray")

X = np.array([0., 15., -9., 7., 12., 3., -21.])
print(zscore(X))

Y = np.array([2., 14., -13., 5., 12., 4., -19.])
print(zscore(Y))
