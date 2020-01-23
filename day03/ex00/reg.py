import numpy as np

def regularization(theta, lambda_):
    """Computes the regularization term of a non-empty numpy.ndarray, with a for-loop."""
    if isinstance(theta, np.ndarray) == 1 and isinstance(lambda_, (int, float)) == 1:
        res = 0.0
        for i in range(len(theta)):
            res += lambda_ * theta[i] ** 2
        return (res)
    else:
        print("regularization: problems with args type")

X = np.array([0, 15, -9, 7, 12, 3, -21])

print(regularization(X, 0.3))

print(regularization(X, 0.01))

print(regularization(X, 0))

