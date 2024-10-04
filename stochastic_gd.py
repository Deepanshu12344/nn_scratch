import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

X, y = load_diabetes(return_X_y=True)
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2, test_size=0.2)
lr = LinearRegression()
lr.fit(X_train, y_train)

print(lr.coef_)
print(lr.intercept_)

y_pred = lr.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(r2)


class SGDRegressor:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.coef_ = None
        self.intercept_ = None
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self, X_train, y_train):
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1])

        for i in range(self.epochs):
            for j in range(X_train.shape[0]):
                index = np.random.randint(0, X_train.shape[0])

                y_hat = np.dot(X_train[index], self.coef_) + self.intercept_

                intercept_der = -2 * (y_train[index] - y_hat)
                self.intercept_ = self.intercept_ - (self.lr * intercept_der)

                coef_der = -2 * np.dot((y_train[index] - y_hat), X_train[index])
                self.coef_ = self.coef_ - (self.lr * coef_der)

        print(self.coef_, self.intercept_)

    def predict(self, X_test):
        return np.dot(X_test, self.coef_) + self.intercept_


sgd = SGDRegressor(0.01, 50)
sgd.fit(X_train, y_train)

y_pred = sgd.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(r2)
