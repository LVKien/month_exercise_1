import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from macpath import dirname, join



path = join(dirname(__file__), "california_housing_train.csv")
data = pd.read_csv(path)
y = data.loc[:, "median_house_value"].tolist()
X1 = data.loc[:,"housing_median_age"].tolist()
X2 = data.loc[:,"total_rooms"].tolist()
X3 = data.loc[:,"total_bedrooms"].tolist()
X4 = data.loc[:,"population"].tolist()

X=[]
X.append(X1)
X.append(X2)
X.append(X3)
X.append(X4)
X = np.asarray(X)
Y = np.asarray(y)

x_train = X[:,:16000]
y_train = y[:16000]
x_test = X[:, 16000:]
y_test = y[16000:]

ols = linear_model.LinearRegression()
model = ols.fit(x_train.T,y_train)

plt.scatter(x_train[0,:50], y_train[:50], color='red')
plt.title('Median House Value vs Housing Median Age', fontsize=14)
plt.xlabel('Housing Median Age', fontsize=14)
plt.ylabel('Median House Value', fontsize=14)
plt.grid(True)
plt.show()

plt.scatter(x_train[1,:50], y_train[:50], color='green')
plt.title('Median House Value vs Total Rooms', fontsize=14)
plt.xlabel('Total Rooms', fontsize=14)
plt.ylabel('Median House Value', fontsize=14)
plt.grid(True)
plt.show()

plt.scatter(x_train[2,:50], y_train[:50], color='blue')
plt.title('Median House Value vs Total Bedrooms', fontsize=14)
plt.xlabel('Total Bedrooms', fontsize=14)
plt.ylabel('Median House Value', fontsize=14)
plt.grid(True)
plt.show()

plt.scatter(x_train[3,:50], y_train[:50], color='yellow')
plt.title('Median House Value vs Population', fontsize=14)
plt.xlabel('Population', fontsize=14)
plt.ylabel('Median House Value', fontsize=14)
plt.grid(True)
plt.show()


print('Intercept: \n', ols.intercept_)
print('Coefficients: \n', ols.coef_)

print ('\n\n',model.predict(x_test.T))
print(0)