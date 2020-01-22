from sigmoid import sigmoid_
import math

def log_loss_(y_true, y_pred, m , eps=1e-15):
    if isinstance(y_true, list) == 1 and isinstance(y_pred, list) == 1:
        if len(y_true) == len(y_pred):
            somme = 0.0
            for i in range(m):
                somme += (y_true[i] * math.log(y_pred[i] + eps) + (1 - y_true[i]) * math.log(1 - y_pred[i] + eps))
            J = - (1 / m) * somme
            return (J)
        else:
            print("log_loss: y_true and y_red are not the same size")
    elif isinstance (y_true, (int, float)) == 1 and isinstance(y_pred, (int, float)) == 1:
        J = - (1 / m) * (y_true * math.log(y_pred + eps) + (1 - y_true) * math.log(1 - y_pred + eps))
        return (J)
    else:
        print("log_loss: error with var type")

# Test n.1
x = 4
y_true = 1
theta = 0.5
y_pred = sigmoid_(x * theta)
m = 1 # length of y_true is 1
print(log_loss_(y_true, y_pred, m))

# Test n.2
x = [1, 2, 3, 4]
y_true = 0
theta = [-1.5, 2.3, 1.4, 0.7]
x_dot_theta = sum([a*b for a, b in zip(x, theta)])
y_pred = sigmoid_(x_dot_theta)
m = 1
print(log_loss_(y_true, y_pred, m))

# Test n.3
x_new = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
y_true = [1, 0, 1]
theta = [-1.5, 2.3, 1.4, 0.7]
x_dot_theta = []
for i in range(len(x_new)):
    my_sum = 0
    for j in range(len(x_new[i])):
        my_sum += x_new[i][j] * theta[j]
    x_dot_theta.append(my_sum)
y_pred = list(sigmoid_(x_dot_theta))
m = len(y_true)
print(log_loss_(y_true, y_pred, m))
