import numpy as np

def mean(x):
    res = 0.0
    if isinstance(x, np.ndarray) == 1:
        for i,el in enumerate(x):
            res += el
        res = res / (i + 1)
        return (res)
    else:
        print("x is not a np.ndarray")
        return (None)


a = np.array([0,15,-9,7,12,3,-21])
b = mean(a ** 2)
print(b)
