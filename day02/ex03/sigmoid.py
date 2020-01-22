import math
import numpy as np

def sigmoid_(x):
    if isinstance(x, (list, np.ndarray)) == 1:
        X = x
        for i in range(len(x)):
            X[i] = 1 / (1 + math.exp(-x[i]))
        return (X)
    elif isinstance(x, (int, float)) == 1:
        return(1 / (1 + math.exp(-x)))
    else:
        print("sigmoid: x is neither a scalar nor a list")
