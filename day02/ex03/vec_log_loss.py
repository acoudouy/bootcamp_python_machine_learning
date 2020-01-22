import numpy as np
import math
from sigmoid import sigmoid_

def vec_log_loss_(y_true, y_pred, m, eps=1e-15):
    if isinstance(y_true, np.ndarray) == 1 and isinstance(y_pred, np.ndarray) == 1:
        if len(y_true) == len(y_pred) and m == len(y_true):
            a = (np.dot(y_true, np.log(y_pred + eps)))
            b = (np.dot((1 - y_true), np.log(1 - y_pred + eps)))
            J = - (1 / m) * (a + b)
            return (J)
        else:
            print("vec_log_loss: y_true and y_red are not the same size")
    elif isinstance (y_true, (int, float)) == 1 and isinstance(y_pred, (int, float)) == 1:
        J = - (1 / m) * (y_true * math.log(y_pred + eps) + (1 - y_true) * math.log(1 - y_pred + eps))
        return (J)
    else:
        print("vec_log_loss: error with var type")

print("Test n.1")
x = 4
y_true = 1
theta = 0.5
y_pred = sigmoid_(x * theta)
m = 1 # length of y_true is 1
print(vec_log_loss_(y_true, y_pred, m))
# 0.12692801104297152

print("Test n.2")
x = np.array([1, 2, 3, 4])
y_true = 0
theta = np.array([-1.5, 2.3, 1.4, 0.7])
y_pred = sigmoid_(np.dot(x, theta))
m = 1
print(vec_log_loss_(y_true, y_pred, m))
# 10.100041078687479

print("Test n.3")
x_new = np.arange(1, 13).reshape((3, 4))
y_true = np.array([1, 0, 1])
theta = np.array([-1.5, 2.3, 1.4, 0.7])
y_pred = sigmoid_(np.dot(x_new, theta))
m = len(y_true)
print(vec_log_loss_(y_true, y_pred, m))
