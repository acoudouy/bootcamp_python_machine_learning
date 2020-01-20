import numpy as np

def dot(x,y):
    res = 0.0
    if isinstance(x, np.ndarray) == 1 and isinstance(y, np.ndarray) == 1:
        if len(x) == len(y):
            for i in range(len(x)):
                res += x[i] * y[i]
            return (res)
        else:
            print("The 2 args do not have the same dimension")
            return (None)
    else:
        print("The 2 args are not np.ndarray")
        return (None)

def mat_vec_prod(x ,y):
    if isinstance(x, np.ndarray) == 1 and isinstance(y, np.ndarray) == 1:
        if len(x[0]) == len(y):
            res = np.ndarray((len(x),1))
            for i in range((len(x))):
                res[i] = dot(x[i],y)
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
X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((7,1))
Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((7,1))

a = mat_vec_prod(W, X)
print("a")
print(a)

b = W.dot(X)
print('b')
print(b)
print()
print(mat_vec_prod(W, Y))
print()
print(W.dot(Y))

