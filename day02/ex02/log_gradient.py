import numpy as np
from sigmoid import sigmoid_

def log_gradient_(x,y_true, y_pred):
    if isinstance(x, list) == 1 and isinstance(y_true, (int, float)) == 1 and isinstance(y_pred, (int, float)) == 1:
        res = x
        for i in range(len(x)):
            res[i] = (y_pred - y_true) * x[i]
        return (res)

    elif isinstance(x, list) == 1 and isinstance(y_true, list) == 1 and isinstance(y_pred, list) == 1:
        if len(x) == len(y_true)  and len(y_true) == len(y_pred):
            res = [0.] * len(x[0])
            for j in range(len(x[0])):
                for i in range(len(x)):
                    res[j] += (y_pred[i] - y_true[i]) * x[i][j]
            return (res)
        else:
            print("log_gradient: x and y do not have the same number of lines")
    else:
        print("log_gradient: error in params")
    
print("Test n.1")
x = [1, 4.2] # 1 represent the intercept
y_true = 1
theta = [0.5, -0.5]
x_dot_theta = sum([a*b for a, b in zip(x, theta)])
y_pred = sigmoid_(x_dot_theta)
print(log_gradient_(x, y_true, y_pred))

print("Test n.2")
x = [1, -0.5, 2.3, -1.5, 3.2]
y_true = 0
theta = [0.5, -0.5, 1.2, -1.2, 2.3]
x_dot_theta = sum([a*b for a, b in zip(x, theta)])
y_pred = sigmoid_(x_dot_theta)
print(y_pred)
print(log_gradient_(x, y_true, y_pred))

print("Test n.3")
x_new = [[1, 2, 3, 4, 5], [1, 6, 7, 8, 9], [1, 10, 11, 12, 13]]
# first column of x_new are intercept values initialized to 1
y_true = [1, 0, 1]
theta = [0.5, -0.5, 1.2, -1.2, 2.3]
x_new_dot_theta = []
for i in range(len(x_new)):
    my_sum = 0
    for j in range(len(x_new[i])):
        my_sum += x_new[i][j] * theta[j]
    x_new_dot_theta.append(my_sum)

y_pred = sigmoid_(x_new_dot_theta)
print(log_gradient_(x_new, y_true, y_pred))
