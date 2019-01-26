
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from macpath import dirname, join

import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

path = join(dirname(__file__), "california_housing_train.csv")
data = pd.read_csv(path)

data = data.values

X    = np.c_[data[:,1],data[:,2],data[:,3],data[:,4]]
Y = np.c_[data[:,0]]

plt.scatter(X[:, 0], Y, s=10, color='g', marker='x')
plt.title (" housing_meidan_age and median_house_value", fontsize = 14)
plt.xlabel('housing_median_age', fontsize=14)
plt.ylabel('median_house_value', fontsize=10)
plt.grid(True)
plt.show()

plt.scatter(X[:, 1], Y, s=10, color='g', marker='x')
plt.title (" total_rooms and median_house_value", fontsize = 14)
plt.xlabel('total_rooms', fontsize=14)
plt.ylabel('median_house_value', fontsize=10)
plt.grid(True)
plt.show()

plt.scatter(X[:, 2], Y, s=10, color='g', marker='x')
plt.title (" total_bedrooms and median_house_value", fontsize = 14)
plt.xlabel('total_bedrooms', fontsize=14)
plt.ylabel('median_house_value', fontsize=10)
plt.grid(True)
plt.show()

plt.scatter(X[:, 3], Y, s=10, color='g', marker='x')
plt.title (" population and median_house_value", fontsize = 14)
plt.xlabel('population', fontsize=14)
plt.ylabel('median_house_value', fontsize=10)
plt.grid(True)
plt.show()


model = LinearRegression()
model = model.fit(X,Y)
predict = model.predict(X)

print(predict)
print()
print(model.coef_)
print()
