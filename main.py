# the most basic neural network ever

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivate_sigmoid(x):
    return x * (1 - x)


def feed_forward(inputs, weights):  # also we can call it predict
    return sigmoid(np.dot(inputs, weights))


def rand_weights(nb):
    return np.random.rand(nb)


datasetForTrainingNeurons = [  # even or odd
    {"inputs": [0, 0, 0, 0], "output": 1},
    {"inputs": [0, 1, 0, 1], "output": 0},
    {"inputs": [0, 0, 1, 0], "output": 1},
    {"inputs": [1, 1, 0, 1], "output": 0},
    {"inputs": [1, 1, 0, 0], "output": 1},
    {"inputs": [0, 0, 1, 1], "output": 0},
    {"inputs": [1, 1, 1, 0], "output": 1},
    {"inputs": [1, 1, 0, 1], "output": 0},
]

inputs_as_array = np.array(
    list(map(lambda a: a["inputs"], datasetForTrainingNeurons)))
outputs_as_array = np.array(
    list(map(lambda a: a["output"], datasetForTrainingNeurons)))

weights = rand_weights(4)

print(inputs_as_array)
print(inputs_as_array.T)

for iteration in range(1000):
    predicted = feed_forward(inputs_as_array, weights)
    errors = outputs_as_array - predicted
    delta = errors * derivate_sigmoid(predicted)
    weights += np.dot(inputs_as_array.T, delta)

# it's 6 (even), i need a close to 1 as output from this
final_test = [0, 1, 1, 0]

print(feed_forward(final_test, weights))
