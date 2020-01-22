import pandas as pd
import numpy as np
import math
from statistics import mean

class LogisticRegressionBatchGd:
    
    def __sigmoid_(self, x):
        if isinstance(x, (list, np.ndarray)) == 1:
#            X = x
#            for i in range(len(x)):
#                X[i] = 1 / (1 + np.exp(-x[i]))
            X = 1 / (1 + np.exp(-x))
            return (X)
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
           # print(y_pred.T)
            return (res)
        elif isinstance(x, np.ndarray) == 1 and isinstance(y_true, (int, float)) == 1 and isinstance(y_pred, (int, float)) == 1:
            res = np.dot((y_pred - y_true), x)
            return (res)
        else:
            print("log_gradient: error in params")



    def __init__(self, alpha=0.001, max_iter=1000, verbose=False, learning_rate='constant'):
        self.alpha = alpha
        self.max_iter = max_iter
        self.verbose = verbose
        self.learning_rate = learning_rate # can be 'constant' or 'invscaling'
        self.theta = np.full(82, 1)  # compare a avant theta a des lignes et colonnes
        self.loss_list = []

        #Code here = a list of loss for each epochs

    def predict(self, x_train):
        """ Predict class labels for samples in x_train."""
        if isinstance(self.theta, np.ndarray) == 1 and isinstance(x_train, np.ndarray) == 1:
            new = np.full((len(x_train),1),1.)
            x_train = np.hstack([new, x_train])
            if len(x_train[0]) == len(self.theta):
                return (x_train.dot(self.theta))
            else:
                print("predict: x's columns is not theta's line. \n")
        else:
            print("score: theta or X is not a np.ndarray. Incompatible.\n")


    def fit(self, x_train, y_train):
        """ Fit the model according to the given training data."""
        if isinstance(x_train, np.ndarray) == 1 and isinstance(self.theta, np.ndarray) == 1 and isinstance(y_train, np.ndarray) == 1:
            if len(x_train[0]) == len(self.theta) - 1 and len(x_train) == len(y_train):
                if isinstance(self.alpha, float) == 1 and isinstance(self.max_iter, int) == 1:
                    new = np.full((len(x_train),1),1.)
                    x_train = np.hstack([new, x_train])
                    for i in range(self.max_iter):
                        y_pred = self.__sigmoid_(x_train.dot(self.theta))
                        self.theta = self.theta - self.alpha * self.__vec_log_gradient_(x_train,y_train, y_pred)
                        loss = self.__vec_log_loss_(y_train, y_pred, len(y_pred))
                        self.loss_list.append(loss)
                        if i % 150 == 0:
                            print("epoch " + str(i) + " : loss " + str(loss))
                        
                    return (self.theta)
                else:
                    print("fit: alpha is not a float or n_cycle not an int")
            else:
                print("fit: x's columns is not theta's line - 1 or dim(X) and dim(Y) different \n")
        else:
            print("fit: theta or X is not a np.ndarray. Incompatible.\n") 


    def score(self, x_train, y_train):
        """ Returns the mean accuracy on the given test data and labels. """
        return (mean(self.loss_list))





# We load and prepare our train and test dataset into x_train, y_train and x_test, y_test
df_train = pd.read_csv('../resources/train_dataset_clean.csv', delimiter=',', header=None, index_col=False)
x_train, y_train = np.array(df_train.iloc[:, 1:82]), np.array(df_train.iloc[:, 0])
df_test = pd.read_csv('../resources/test_dataset_clean.csv', delimiter=',', header=None, index_col=False)
x_test, y_test = np.array(df_test.iloc[:, 1:82]), np.array(df_test.iloc[:, 0])

# We set our model with our hyperparameters : alpha, max_iter, verbose and learning_rate
model = LogisticRegressionBatchGd(alpha=5e-5, max_iter=3000, verbose=True, learning_rate='constant')

# We fit our model to our dataset and display the score for the train and test datasets
model.fit(x_train, y_train) # mise a jour des theta
print(f'Score on train dataset : {model.score(x_train, y_train)}')
y_pred = model.predict(x_test)
print(f'Score on test dataset  : {(y_pred == y_test).mean()}')



# epoch 0     : loss 2.711028065632692
# epoch 150   : loss 1.760555094793668
# epoch 300   : loss 1.165023422947427
# epoch 450   : loss 0.830808383847448
# epoch 600   : loss 0.652110347325305
# epoch 750   : loss 0.555867078788320
# epoch 900   : loss 0.501596689945403
# epoch 1050  : loss 0.469145216528238
# epoch 1200  : loss 0.448682476966280
# epoch 1350  : loss 0.435197719530431
# epoch 1500  : loss 0.425934034947101
# Score on train dataset : 0.7591904425539756
# Score on test dataset  : 0.7637737239727289

# This is an example with verbose set to True, you could choose to display your loss at the epochs you want.
# Here I choose to only display 11 rows no matter how many epochs I had.
# Your score should be pretty close to mine.
# Your loss may be quite different weither you choose different hyperparameters, if you add an intercept to your x_train
# or if you shuffle your x_train at each epochs (this introduce stochasticity !) etc...
# You might not get a score as good as sklearn.linear_model.LogisticRegression because it uses a different algorithm and
# more optimized parameters that would require more time to implement.








