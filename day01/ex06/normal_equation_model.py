import pandas as pd
import numpy as np
from mylinearregression import MyLinearRegression as MyLR





data = pd.read_csv("../resources/spacecraft_data.csv")
X = np.array(data[['Age','Thrust_power','Terameters']])
Y = np.array(data['Sell_price']).reshape(-1,1)

theta = np.array([[1.], [1.], [1.], [1.]])
#theta2 = np.array([[1.], [1.], [1.], [1.]])

myLR_ne = MyLR(theta)
print(len(myLR_ne.theta))
myLR_lgd = MyLR(theta)
print(len(myLR_lgd.theta))
myLR_lgd.fit_(X,Y, alpha = 5e-5, n_cycle = 10000)
Y_new1 = myLR_lgd.predict_(X)

myLR_ne.normalequation_(X, Y)
Y_new2 = myLR_ne.predict_(X)
print("MSE = ")
print(myLR_lgd.mse_(Y, Y_new1))
print("MSE = ")
print(myLR_ne.mse_(Y, Y_new2))
