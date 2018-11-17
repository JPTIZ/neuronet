from sys import argv
from typing import List

import numpy as np


def sigmoid(x):
    return 1.0/(1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


class NeuralNetwork:
    def __init__(self, x: List[List[int]], y: List[List[int]]):
        print(f'len(x): {len(x)}\nlen(y): {len(y)}')
        self.input: List[List[int]] = x
        self.weights: List[List[List[int]]] = [
            np.random.rand(self.input.shape[1], len(y)),
            np.random.rand(len(y), 1),
        ]
        self.y: List[List[int]] = y
        self.output: List[int] = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights[0]))
        self.output = sigmoid(np.dot(self.layer1, self.weights[1]))

    def backprop(self):
        d_weights = [
            np.dot(
                self.input.T,
                (np.dot(
                    2*(self.y - self.output) * sigmoid_derivative(self.output),
                    self.weights[1].T
                ) * sigmoid_derivative(self.layer1))
            ),
            np.dot(
                self.layer1.T,
                (2*(self.y - self.output) * sigmoid_derivative(self.output))
            ),
        ]

        self.weights = [
            w + d_w for w, d_w in zip(self.weights, d_weights)
        ]


if __name__ == '__main__':
    try:
        train_file = argv[1]
        trains = argv[2]
    except IndexError:
        print(f'Usage: {argv[0]} <training file> <train iter count>')
        exit(1)

    print('Gathering data...')

    inputs = []
    outputs = []
    with open(train_file) as f:
        labels = f.readline().split(',')
        print(f'Labels: {labels}')

        for line in f.readlines():
            output, *data = line.split(',')
            inputs += [[int(data) for data in data]]
            outputs += [output]

    inputs = np.array(inputs)
    outputs = np.array(outputs)

    print('Creating NeuroNet...')
    neuro = NeuralNetwork(inputs, outputs)

    print(f'Training {trains} times...')
    try:
        for i in range(1500):
            neuro.feedforward()
            neuro.backprop()
    except MemoryError:
        print(f'Failed in iteration {i}')

    print(neuro.output)
