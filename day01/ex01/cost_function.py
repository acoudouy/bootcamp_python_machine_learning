import numpy as np

def cost_elem_(theta, X, Y):
    if len(X[0]) == len(theta) - 1 and len(X) == len(Y):
        new = np.full((len(X),1),1.)
        X = np.hstack([new, X])
        J = np.full((len(X), 1),0.0)
        for i in range(len(X)):
            J[i] = ((np.dot(X[i], theta) - Y[i]) ** 2)
            J[i] = J[i] / (len(X) * 2)
        return (J)
    else:
        print("Incompatibily in X, Y and theta's dimensions.\n")

def cost_(theta, X, Y):
    if len(X[0]) == len(theta) - 1 and len(X) == len(Y):
       new = np.full((len(X),1),1)
       X = np.hstack([new, X])
       J = 0.0
       for i in range(len(X)):
           J += (np.dot(X[i], theta) - Y[i]) ** 2
       J = J / ((i + 1) * 2)
       return (float(J))
    else:
        print("Incompatibily in X, Y and theta's dimensions.\n")

X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
theta1 = np.array([[2.], [4.]])
Y1 = np.array([[2.], [7.], [12.], [17.], [22.]])
print(cost_elem_(theta1, X1, Y1))
print(cost_(theta1, X1, Y1))


X2 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8.,
80.]])
theta2 = np.array([[0.05], [1.], [1.], [1.]])
Y2 = np.array([[19.], [42.], [67.], [93.]])
print(cost_elem_(theta2, X2, Y2))
print(cost_(theta2, X2, Y2))

X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
Y1 = np.array([[2.], [6.], [10.], [14.], [18.]])
theta1 = np.array([[1.], [1.]])
print(cost_elem_(theta1, X1, Y1))
print(cost_(theta1, X1, Y1))
