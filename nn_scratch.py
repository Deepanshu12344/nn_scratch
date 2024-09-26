import numpy as np
import pandas as pd

data = pd.read_csv('train.csv')
data = np.array(data)
m,n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]

def init_params():
    W1 = np.random.randn(10, 784)
    b1 = np.random.randn(10, 1)
    W2 = np.random.randn(10, 10)
    b2 = np.random.randn(10, 1)
    return W1,W2,b1,b2

def ReLU(Z):
    return  np.maximum(0,Z)

def Softmax(Z):
    return np.exp(Z)/np.sum(np.exp(Z))

def forward_prop(W1,W2,b1,b2,X,Y):
    Z1 = np.dot(W1,X) + b1
    A1 = ReLU(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = Softmax(Z2)
    return Z1,Z2,A1,A2


