import math

def sigmoid_(x):
    if isinstance(x, list) == 1:
        X = x
        for i in range(len(x)):
            X[i] = 1 / (1 + math.exp(-x[i]))
        return (X)
    elif isinstance(x, float) == 1 or isinstance(x, int) == 1:
        return(1 / (1 + math.exp(-x)))
    else:
        print("x is neither a scalar nor a list")
