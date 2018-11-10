import numpy as np


def sigmoid(x):
    return 1.0/(1+ np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights = [
            np.random.rand(self.input.shape[1], 4),
            np.random.rand(4, 1),
        ]
        self.y = y
        self.output = np.zeros(y.shape)

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
    inputs = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    results = np.array([[0], [1], [1], [0]])

    neuro = NeuralNetwork(inputs, results)
    for i in range(30000):
        neuro.feedforward()
        neuro.backprop()
    print(neuro.output)
