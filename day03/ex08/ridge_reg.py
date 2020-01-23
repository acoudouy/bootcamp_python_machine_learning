from mylinearregression import MyLinearRegression as MyLR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MyRidge(MyLR):
    """A class similar to sklearn.linear_model.Ridge. """
    
    def vec_reg_linear_grad(self, x, y, theta, lambda_):
        """Computes the regularized linear gradient of three non-empty numpy.ndarray, with two for-loop. The three arrays must have compatible dimensions."""

        if isinstance(y, np.ndarray) == 1 and isinstance(x, np.ndarray) == 1 and isinstance(theta, np.ndarray) == 1:
            if isinstance(lambda_, (int, float)) == 1:
                if len(x) == len(y):
                    if len(x[0]) == len(theta):
                        res = np.zeros(len(x[0]))
                  #      res = np.dot(x.T, (x.dot(theta) - y)) / len(x) + np.dot(lambda_/len(x), theta) ###
                        res = np.dot(x.T, (x.dot(theta) - y)) / len(x) + lambda_/len(x) * theta ###
                        temp = np.dot(x.T, (x.dot(theta) - y)) / len(x) ###
                        res[0] = temp[0]
                        return (res)
                    else:
                        print("vec_reg_lin_grad: x's columns is not equel to theta's lines")
                else:
                    print("vec_reg_lin_grad: lamda not a float / int")
            else:
                print("vec_reg_lin_grad: x and y do not have the same number of lines")
        else:
            print("vec_reg_lin_grad: At least one argument is not a np.ndaray")


    def fit_(self, X, Y, alpha, lambda_=1.0, n_cycle=1000, tol=0.001):
        """  Fit the linear model by performing Ridge regression (Tikhonov regularization)."""
        if isinstance(lambda_, (int, float)) == 1 and isinstance(n_cycle, int) == 1 and isinstance(tol, float) == 1:
            new = np.full((len(X),1),1.)
            X = np.hstack([new, X])
            for i in range(n_cycle):
                self.theta = self.theta - alpha * self.vec_reg_linear_grad(X,Y, self.theta, lambda_)
            return (self.theta)
        else:
            print("MyRidge: lambda, max_iter or tol are not int or float")

data = pd.read_csv("../resources/data.csv")
data['x1.2'] = data['# x1'] ** 3
data['x2.2'] = data['x2'] ** 2
Y = np.array(data[['y']])

X1 = np.array(data[['# x1','x2']])
theta1 = np.array([[-5],[0],[0]])

X2 = np.array(data[['# x1','x1.2','x2','x2.2']])
theta2 = np.array([[0.72],[0.5],[0.],[0.1],[-0.6]])


#Myr1 = MyRidge(theta1)
Myr1 = MyRidge(theta2)
Myr2 = MyRidge(theta2)

#Myr1.fit(X1, Y, alpha=0.0005, n_cycle=4000)
#Y_new1 = Myr1.predict_(X1)
Myr1.fit(X2, Y, alpha=0.0005, n_cycle=4000)
Y_new1 = Myr1.predict_(X2)

Myr2.fit_(X2, Y, alpha=0.0005, lambda_=10,n_cycle=4000)
Y_new2 = Myr2.predict_(X2)

print("MSE2=")
a = 0.0
for i in range(55):
    a += 1
    Myr2.fit_(X2, Y, alpha=0.0005, lambda_=a,n_cycle=4000)
    Y_new2 = Myr2.predict_(X2)
    print("Lambda = " + str(a) + "  MSE= " +str(Myr2.mse_(Y, Y_new2)))
    print(Myr2.theta)
print("MSE1=")
print(Myr1.mse_(Y, Y_new1))

plt.scatter(data['# x1'], Y, color='black')
plt.scatter(data['# x1'], Y_new1, color='green')
plt.scatter(data['# x1'], Y_new2, color='red')
plt.title('My model : day03ex08')
plt.xlabel('x1')
plt.ylabel('y')
plt.show()
