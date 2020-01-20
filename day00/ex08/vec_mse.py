import numpy as np

def vec_mse(y, y_hat):
    if isinstance(y, np.ndarray) == 1 and isinstance(y_hat, np.ndarray) == 1:
        if len(y) == len(y_hat):
            if y_hat.ndim == 1 and y.ndim == 1:
     #           res = np.square(np.subtract(y_hat,y)).mean() 
                res = (y_hat - y).dot(y_hat - y) / len(y)
                return (res)
            else:
                print("2 args are not only composed with one column")
        else:
            print("2 args do not have the same size")
    else:
        print("2 args are not np.ndarray")
    return (None)

X = np.array([0, 15, -9, 7, 12, 3, -21])
Y = np.array([2, 14, -13, 5, 12, 4, -19])

print(vec_mse(X,Y))
