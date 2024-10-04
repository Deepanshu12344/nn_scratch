# third method
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

X, y = spiral_data(100, 3)


class Layers_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.output = None
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


X, y = spiral_data(samples=100, classes=3)
dense1 = Layers_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layers_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:100])
'''
#first method
import numpy as np
np.random.seed(0)

inputs = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
bias = [2, 3, 0.5]

weights2 = [[0.1, -0.14, 0.5],
           [-0.5, 0.12, -0.33],
           [-0.44, 0.73, -0.13]]
bias2 = [-1, 2, -0.5]

layer1_output = np.dot(inputs, np.array(weights).T) + bias
layer2_output = np.dot(layer1_output, np.array(weights2).T) + bias2

print(layer2_output)
'''

'''
#second method
import numpy as np
np.random.seed(0)

inputs = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
bias = [2, 3, 0.5]

nn_output = []
for neuron_weight, neuron_bias in zip(weights, bias):
    neuron_output = 0
    for input, weight in zip(inputs, neuron_weight):
        neuron_output += input * weight
    neuron_output += neuron_bias
    nn_output.append(neuron_output)
'''

'''
#Custom Dataset
def create_data(points, classes):
    X = np.zeros((points * classes, 2))
    Y = np.zeros(points * classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number*4, (class_number+1) * 4, points) + np.random.randn(points) * 0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        Y[ix] = class_number
    return  X, Y

import matplotlib.pyplot as plt

print('here')
X, Y = create_data(100,3)
plt.scatter(X[:,0], X[:,1])
plt.show()

plt.scatter(X[:,0], X[:,1], c=Y, cmap="brg")
plt.show()
'''
