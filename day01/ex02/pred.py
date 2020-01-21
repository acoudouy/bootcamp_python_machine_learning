import numpy as np

def predict_(theta, X):
    if isinstance(theta, np.ndarray) == 1 and isinstance(theta, np.ndarray) == 1:
        if len(X[0]) == len(theta) - 1:
            new = np.full((len(X),1),1)
            X = np.hstack([new, X])
            return (X.dot(theta))
        else:
            print("\nx's columns is not theta's line - 1. \n")
    else:
        print("theta or X is not a np.ndarray. Incompatible.\n")

