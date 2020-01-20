import numpy as np
import types

def sum_(x, f):
    res = 0.0
    if isinstance(x, np.ndarray) == 1 and isinstance(f, types.FunctionType) == 1:
        for el in x:
            res += f(el)
        return (res)
    elif isinstance(f, types.FunctionType) == 0:
        print("f is not a function")
        return (None)
    else:
        print("x is not a np.ndarray")
        return (None)


a = np.array([0,15,-9,7,12,3,-21])
b = sum_(a,lambda x: x**2)
print(b)
