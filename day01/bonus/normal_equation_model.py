import pandas as pd
import numpy as np
from mylinearregression import MyLinearRegression as MyLR
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score




data = pd.read_csv("../resources/spacecraft_data.csv")
X = np.array(data[['Age','Thrust_power','Terameters']])
X1 = np.array(data[['Age','Thrust_power','Terameters']])
Y = np.array(data['Sell_price']).reshape(-1,1)

theta1 = np.array([[1.], [1.], [1.], [1.]])
theta2 = np.array([[1.], [1.], [1.], [1.]])

myLR_ne = MyLR(theta1)
myLR_lgd = MyLR(theta2)
myLR_lgd.fit_(X,Y, alpha = 5e-5, n_cycle = 2000)
Y_new1 = myLR_lgd.predict_(X)

myLR_ne.normalequation_(X, Y)
Y_new2 = myLR_ne.predict_(X)

print("LR MSE")
print(myLR_lgd.mse_(Y, Y_new1))
print("LR RMSE")
print(myLR_lgd.rmse_(Y, Y_new1))
print("NE MSE")
print(myLR_ne.mse_(Y, Y_new2))
print("NE RMSE")
print(myLR_ne.rmse_(Y, Y_new2))
print("LR R2SCORE")
print(myLR_lgd.r2score_(Y, Y_new1))
print("LR R2SCORE check")
print(r2_score(Y, Y_new1))
print("NE R2SCORE")
print(myLR_ne.r2score_(Y, Y_new2))
print("NE R2SCORE check")
print(r2_score(Y, Y_new2))

'''
print("MSE = ")
print(myLR_lgd.theta)
print(myLR_lgd.mse_(Y, Y_new1))
print("MSE = ")
print(myLR_ne.theta)
print(myLR_ne.mse_(Y, Y_new2))

plt.scatter(data.Age, Y_new1, color='green')
plt.scatter(data.Age, Y_new2, color='red')
plt.scatter(data.Age, Y, color='blue')
plt.title('Linear Regression vs. Normal Equation Comparaison')
plt.xlabel('Age en annee')
plt.ylabel('Prix')
plt.show()
'''
