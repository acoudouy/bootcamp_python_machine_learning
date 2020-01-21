import numpy as np
import pandas as pd
from mylinearregression import MyLinearRegression as MyLR
import matplotlib.pyplot as plt

data = pd.read_csv("../resources/spacecraft_data.csv")

# For Linear Regression:
'''
#Age vs Price
X = np.array(data['Age']).reshape(-1,1)
Y = np.array(data['Sell_price']).reshape(-1,1)
theta = np.array([[700], [-8.0]])
myLR_age = MyLR(theta)
print("New Theta:")
print(myLR_age.fit_(X, Y, alpha = 2.5e-5, n_cycle = 300000))
Y_new = myLR_age.predict_( X )
plt.scatter(data['Age'],Y_new)
plt.scatter(data['Age'],data['Sell_price'])
plt.title('Age et Sell Price: prediction et dataset')
plt.xlabel('Age (en annees)')
plt.ylabel('Sell Price (en keuros)')
RMSE_age = myLR_age.mse_(Y, Y_new)
print(RMSE_age)

#Power vs Price
X2 = np.array(data['Thrust_power']).reshape(-1,1)
Y2 = np.array(data['Sell_price']).reshape(-1,1)
theta2 = np.array([[0], [9.0]])
myLR_power = MyLR(theta2)
print("New Theta:")
print(myLR_power.fit_(X2, Y2, alpha = 2.5e-5, n_cycle = 100000))
Y2_new = myLR_power.predict_(X2)
RMSE_power = myLR_power.mse_(Y2, Y2_new)
print(RMSE_power)
plt.scatter(X2,Y2_new)
plt.scatter(X2,Y2)
plt.title('Power et Sell Price: prediction et dataset')
plt.xlabel('Power')
plt.ylabel('Sell Price (en keuros)')

#Distance vs Price
X3 = np.array(data['Terameters']).reshape(-1,1)
Y3 = np.array(data['Sell_price']).reshape(-1,1)
theta3 = np.array([[820], [-3]])
myLR_thera = MyLR(theta3)
print("New Theta:")
print(myLR_thera.fit_(X3, Y3, alpha = 2.5e-5, n_cycle = 200000))
Y3_new = myLR_thera.predict_(X3)
RMSE_thera = myLR_thera.mse_(Y3, Y3_new)
print(RMSE_thera)
plt.scatter(X3,Y3_new)
plt.scatter(X3,Y3)
plt.title('Distance et Sell Price: prediction et dataset')
plt.xlabel('Distance')
plt.ylabel('Sell Price (en keuros)')

plt.show()
'''

#For Multilinear Regression
print(data)
X = np.array(data[['Age','Thrust_power','Terameters']])
Y = np.array(data[['Sell_price']])
theta = np.array([[1.0],[-10.],[6.0],[-2.0]])
my_lreg = MyLR(theta)
Y_new = my_lreg.predict_(X)
print("MSE =")
print(my_lreg.mse_(Y, Y_new))

print("New Theta:")
print(my_lreg.fit_(X, Y, alpha = 2.5e-5, n_cycle = 200000))

Y_new = my_lreg.predict_(X)
print("MSE2 =")
print(my_lreg.mse_(Y, Y_new))

plt.scatter(data['Age'],Y_new)
plt.scatter(data['Age'],data['Sell_price'])
plt.title('Age et Sell Price: prediction et dataset')
plt.xlabel('Age (en annees)')
plt.ylabel('Sell Price (en keuros)')
plt.show()
