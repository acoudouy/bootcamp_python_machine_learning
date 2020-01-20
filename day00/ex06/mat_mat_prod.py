import numpy as np

def mat_mat_prod(x ,y):
    if isinstance(x, np.ndarray) == 1 and isinstance(y, np.ndarray) == 1:
        if len(x[0]) == len(y):
            res = np.zeros((len(x), len(y[0])))
            for i in range(len(x)):
                for j in range(len(y[0])):
                    for k in range(len(x[0])):
                        res[i][j] += x[i][k] * y[k][j]
            return (res)
        else:
            print("x and y do not share compatible dimensions.")
            return (None)
    else:
        print("x or y is not a np.ndarray")
        return (None)


W = np.array([
 [ -8, 8, -6, 14, 14, -9, -4],
 [ 2, -11, -2, -11, 14, -2, 14],
 [-13, -2, -5, 3, -8, -4, 13],
 [ 2, 13, -14, -15, -14, -15, 13],
 [ 2, -1, 12, 3, -7, -3, -6]])
Z = np.array([
 [ -6, -1, -8, 7, -8],
 [ 7, 4, 0, -10, -10],
 [ 7, -13, 2, 2, -11],
 [ 3, 14, 7, 7, -4],
 [ -1, -3, -8, -4, -14],
 [ 9, -14, 9, 12, -7],
 [ -9, -4, -10, -3, 6]])

A = np.array([
 [1,2,3],
 [1,1,1]])

B = np.array([
 [4,1,1,1],
 [5,1,1,1],
 [6,1,1,1]])
print(mat_mat_prod(W,Z))
print ()
print(W.dot(Z))
print ()
print(mat_mat_prod(Z,W))
print ()
print(Z.dot(W))
