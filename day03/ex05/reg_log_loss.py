from sigmoid import sigmoid_
import math
import numpy as np

def reg_log_loss_(y_true, y_pred, m, theta, lambda_, eps=1e-15):
    """Compute the logistic loss value."""
    if isinstance(y_true, np.ndarray) == 1 and isinstance(y_pred, np.ndarray) == 1:
        if len(y_true) == len(y_pred) and m == len(y_true):
            a = (np.dot(-y_true, np.log(y_pred + eps)))
            b = (np.dot((1 - y_true), np.log(1 - y_pred + eps)))
            J = (1 / m) * (a - b  +  (lambda_ *np.dot(theta, theta)))
            return (J)
        else:
            print("vec_log_loss: y_true and y_red are not the same size")
    elif isinstance (y_true, (int, float)) == 1 and isinstance(y_pred, (int, float)) == 1:
        J = - (1 / m) * (y_true * math.log(y_pred + eps) + (1 - y_true) * math.log(1 - y_pred + eps))
        return (J)
    else:
        print("vec_log_loss: error with var type")

print("Test n.1")
x_new = np.arange(1, 13).reshape((3, 4))
y_true = np.array([1, 0, 1])
theta = np.array([-1.5, 2.3, 1.4, 0.7])
y_pred = sigmoid_(np.dot(x_new, theta))
m = len(y_true)
print(reg_log_loss_(y_true, y_pred, m, theta, 0.0))

# Test n.2
print("Test n.2")
x_new = np.arange(1, 13).reshape((3, 4))
y_true = np.array([1, 0, 1])
theta = np.array([-1.5, 2.3, 1.4, 0.7])
y_pred = sigmoid_(np.dot(x_new, theta))
m = len(y_true)
print(reg_log_loss_(y_true, y_pred, m, theta, 0.5))

# Test n.3
print("Test n.3")
x_new = np.arange(1, 13).reshape((3, 4))
y_true = np.array([1, 0, 1])
theta = np.array([-1.5, 2.3, 1.4, 0.7])
y_pred = sigmoid_(np.dot(x_new, theta))
m = len(y_true)
print(reg_log_loss_(y_true, y_pred, m, theta, 1))

# Test n.4
print("Test n.4")
x_new = np.arange(1, 13).reshape((3, 4))
y_true = np.array([1, 0, 1])
theta = np.array([-5.2, 2.3, -1.4, 8.9])
y_pred = sigmoid_(np.dot(x_new, theta))
m = len(y_true)
print(reg_log_loss_(y_true, y_pred, m, theta, 1))

# Test n.5
print("Test n.5")
x_new = np.arange(1, 13).reshape((3, 4))
y_true = np.array([1, 0, 1])
theta = np.array([-5.2, 2.3, -1.4, 8.9])
y_pred = sigmoid_(np.dot(x_new, theta))
m = len(y_true)
print(reg_log_loss_(y_true, y_pred, m, theta, 0.3))

# Test n.6
print("Test n.6")
x_new = np.arange(1, 13).reshape((3, 4))
y_true = np.array([1, 0, 1])
theta = np.array([-5.2, 2.3, -1.4, 8.9])
y_pred = sigmoid_(np.dot(x_new, theta))
m = len(y_true)
print(reg_log_loss_(y_true, y_pred, m, theta, 0.9))
