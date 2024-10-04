"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

X,y = make_regression(n_samples=4, n_features=1, n_informative=1, n_targets=1, noise=80, random_state=13)
plt.scatter(X, y)
# plt.show()

lr.fit(X,y)
print('Slope : ', lr.coef_, 'Intercept : ', lr.intercept_)

b = -100
m = 78.35
lr = 0.1
epochs = 100

for i in range(epochs):
    loss_slope = -2 * np.sum(y - m*X.ravel() - b)
    b = b-(lr * loss_slope)

    y_pred = m * X +b
    print(y_pred)
"""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

lr = LinearRegression()

X, y = make_regression(n_samples=100, n_features=1, n_informative=1, n_targets=1, noise=20, random_state=12)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=2)
lr.fit(X_train, y_train)
print('Slope : ', lr.coef_, 'Intercept : ', lr.intercept_)
y_pred = lr.predict(X_test)
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(r2)


class GDRegressor:

    def __init__(self, learning_rate, epochs):
        self.m = 100
        self.b = -120
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        for i in range(self.epochs):
            loss_slope_b = -2 * np.sum(y - self.m * X.ravel() - self.b)
            loss_slope_m = -2 * np.sum((y - self.m * X.ravel() - self.b) * X.ravel())

            self.b = self.b - (self.lr * loss_slope_b)
            self.m = self.m - (self.lr * loss_slope_m)

        print(self.m, self.b)

    def predict(self, X):
        return self.m * X + self.b


gd = GDRegressor(0.003, 100)
print(gd.fit(X_train, y_train))
y_pred = gd.predict(X_test)
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(r2)
