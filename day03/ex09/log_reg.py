import pandas as pd
import numpy as np
import math
from statistics import mean

class LogisticRegressionBatchGd:
    
    def __sigmoid_(self, x):
        if isinstance(x, (list, np.ndarray)) == 1:
            return (1 / (1 + np.exp(-x)))
        elif isinstance(x, (int, float)) == 1:
            return(1 / (1 + math.exp(-x)))
        else:
            print("sigmoid: x is neither a scalar nor a list")     
    
    def __vec_log_loss_(self, y_true, y_pred, m, eps=1e-15):
        """ Comput the logistic loss value. """
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

    
    def __vec_log_gradient_(self, x ,y_true, y_pred):
        """ Comput the gradient. """
        if isinstance(x, np.ndarray) == 1 and isinstance(y_true, np.ndarray) == 1 and isinstance(y_pred, np.ndarray) == 1:
            res = np.dot((y_pred.T - y_true), x)
            return (res)
        elif isinstance(x, np.ndarray) == 1 and isinstance(y_true, (int, float)) == 1 and isinstance(y_pred, (int, float)) == 1:
            res = np.dot((y_pred - y_true), x)
            return (res)
        else:
            print("log_gradient: error in params")

    def vec_reg_logistic_grad(self, y, x, lambda_):
        """Computes the regularized linear gradient of three non-empty numpy.ndarray, without any for-loop. The three arrays must have compatible dimensions."""
        if isinstance(x, np.ndarray) == 1 and isinstance(y, np.ndarray) == 1 and isinstance(self.theta, np.ndarray) == 1:
            if isinstance(lambda_, (int, float)) == 1:
                if len(y) == len(x) and len(x[0]) == len(self.theta):
                    y_pred = self.__sigmoid_(np.dot(x, self.theta))
                    temp = np.dot(x.T, (y_pred - y)) / len(x)
                    res = np.dot(x.T, (y_pred - y)) / len(x) + np.dot(lambda_ / len(x), self.theta)
                    res[0] = temp[0]
                    return (res)
                else:
                    print("vec_reg_log_gradient: error in size of x, y or theta")
            else:
                print("vec_reg_log_gradient: lamda is not a scalar")
        else:
            print("vec_reg_log_gradient: error in type of x, y or theta")

    def __init__(self, theta, alpha=0.001, max_iter=1000, verbose=False, learning_rate='constant', penalty='l2'):
        if penalty != 'l2' and penalty != 'none':
            print("init class LRGB: penalty error 404")
        self.alpha = alpha
        self.max_iter = max_iter
        self.verbose = verbose
        self.theta = theta
        self.penalty = penalty # with ou without regression : l2 or None
        self.learning_rate = learning_rate # can be 'constant' or 'invscaling'
        self.loss_list = []

    def predict(self, x_train):
        """ Predict class labels for samples in x_train."""
        if isinstance(self.theta, np.ndarray) == 1 and isinstance(x_train, np.ndarray) == 1:
            new = np.full((len(x_train),1),1.)
            x_train = np.hstack([new, x_train])
            y_pred = self.__sigmoid_(x_train.dot(self.theta))
            y_pred = np.where(y_pred > 0.5, 1, 0)
            return(y_pred)
        else:
            print("score: theta or X is not a np.ndarray. Incompatible.\n")


    def fit(self, x_train, y_train, lambda_=0.0):
        """ Fit the model according to the given training data."""
        if isinstance(x_train, np.ndarray) == 1 and isinstance(self.theta, np.ndarray) == 1 and isinstance(y_train, np.ndarray) == 1:
            if len(x_train[0]) == len(self.theta) - 1 and len(x_train) == len(y_train):
                if isinstance(self.alpha, float) == 1 and isinstance(self.max_iter, int) == 1:
                    new = np.full((len(x_train),1),1.)
                    x_train = np.hstack([new, x_train])
                    for i in range(self.max_iter):
                        y_pred = self.__sigmoid_(x_train.dot(self.theta))
                        y_pred = np.where(y_pred > 0.5, 1, 0)
                        if self.penalty == "l2":
                            self.theta = self.theta - self.alpha * self.vec_reg_logistic_grad(y_train, x_train, lambda_)
                        elif self.penalty == 'none':
                            self.theta = self.theta - self.alpha * self.__vec_log_gradient_(x_train, y_train, y_pred)
                        loss = self.__vec_log_loss_(y_train, y_pred, len(y_pred))
                        self.loss_list.append(loss)
                        if i % 1500  == 0:
                            print("epoch " + str(i) + " : loss " + str(loss))
             #           print(str(self.theta[0]) + "epoch " + str(i) )
                    return (self.theta)
                else:
                    print("fit: alpha is not a float or n_cycle not an int")
            else:
                print("fit: x's columns is not theta's line - 1 or dim(X) and dim(Y) different \n")
        else:
            print("fit: theta or X is not a np.ndarray. Incompatible.\n") 


    def score(self, x_train, y_train):
        """ Returns the mean accuracy on the given test data and labels. """
        y_pred = model.predict(x_train)
        return ((y_pred == y_train).mean())

# We load and prepare our train and test dataset into x_train, y_train and x_test, y_test
df_train = pd.read_csv('../../day02/resources/train_dataset_clean.csv', delimiter=',', header=None, index_col=False)
x_train, y_train = np.array(df_train.iloc[:, 1:82]), np.array(df_train.iloc[:, 0])
df_test = pd.read_csv('../../day02/resources/test_dataset_clean.csv', delimiter=',', header=None, index_col=False)
x_test, y_test = np.array(df_test.iloc[:, 1:82]), np.array(df_test.iloc[:, 0])
theta = np.full(82, 0)  # compare a avant theta a des lignes et colonnes

# We set our model with our hyperparameters : alpha, max_iter, verbose and learning_rate
model = LogisticRegressionBatchGd(theta, alpha=0.01, max_iter=1501, verbose=True, learning_rate='constant', penalty='none')
model2 = LogisticRegressionBatchGd(theta, alpha=0.01, max_iter=1501, verbose=True, learning_rate='constant', penalty='l2')

# We fit our model to our dataset and display the score for the train and test datasets
model.fit(x_train, y_train) # mise a jour des theta
print(f'Score on train dataset : {model.score(x_train, y_train)}')
a = 0
for i in range(10):
    a += 5
    model2 = LogisticRegressionBatchGd(theta, alpha=0.01, max_iter=1501, verbose=True, learning_rate='constant', penalty='l2')
    model2.fit(x_train, y_train, a) # mise a jour des theta
    print(f'Score on train dataset : {model2.score(x_train, y_train)}')
y_pred = model.predict(x_test)
print(f'Score on test dataset  : {(y_pred == y_test).mean()}')
