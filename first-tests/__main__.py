import numpy as np
from defaultlist import defaultlist as dlist

from sys import argv
from typing import Tuple
import csv


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: float) -> float:
    return x * (1 - x)


def make_layer(input_length: int, neurons: int):
    weights = np.random.rand(neurons, input_length) * 2 - 1
    biases = np.random.rand(neurons) * 2 - 1
    return weights, biases


class NeuralNetwork:
    def __init__(
        self,
        input_length,
        hidden_layers,
        neurons_per_layer,
        output_length,
        activate,
        activate_diff,
    ):
        first_layer = make_layer(input_length, neurons_per_layer)[0]
        hidden_layers = (
            make_layer(neurons_per_layer, neurons_per_layer)[0]
            for _ in range(hidden_layers - 1)
        )
        output_layer = make_layer(output_length, neurons_per_layer)[0]

        self.layers = [first_layer, *hidden_layers, output_layer]
        self.activate = activate
        self.activate_diff = activate_diff

    def train(self, input_row, expected_output):
        '''
        HINT: A Perceptron is defined by a weights vector.
        '''
        # Setup
        input_layer, *hidden_layers, output_layer = self.layers

        # Feed forward
        print('Feed me >:) Feeding...')

        # -> Input layer
        a = dlist(lambda *a: dlist(lambda *b: float))
        z = dlist(lambda *a: dlist(lambda *b: float))
        for j, weights in enumerate(input_layer):
            z[0][j] = sum(i*weights[k]
                          for k, i in enumerate(input_row))
            a[0][j] = self.activate(z[0][j])

        # -> Hidden layers
        for l, layer in enumerate(hidden_layers):
            last_layer = self.layers[l]
            for j, weights in enumerate(self.layers[l+1]):
                z[l+1][j] = sum(a*weights[k]
                                for k, a in enumerate(last_layer))
                a[l+1][j] = self.activate(z[l][j])

        # -> Output layer
        for j, weights in enumerate(output_layer):
            _sum = 0
            for k, i in enumerate(a[l][j]):
                print('-'*80)
                print(f'i: {i}')
                print(f'w: {weights[k]}')
                _sum += i*weights[k]
            z[l+1][j] = _sum
            a[l+1][j] = self.activate(z[l+1][j])

        # Backpropagation
        print('Backpropping...')

        # -> C(o)
        print(f'answ: {list(a[-1])}')
        print(f'expc: {expected_output}')
        error = sum(np.square(result - expected)
                    for result, expected in zip(a[-1], expected_output))
        print(f'error: {error}')

        # TODO: Keep going with the backprop, bro :)


def read_dataset(csv_path: str):
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        for label, *pixels in reader:
            pixels = np.array(pixels).astype(np.float) / 255
            # normalized to [0, 1]

            output = np.zeros(10)
            output[int(label)] = 1

            yield pixels, output


def cli():
    usage = (
        f'Usage: {argv[0]} '
        '<training-dataset> '
        '<hidden-layers> '
        '<neurons-per-layer>'
    )

    if '-h' in argv or '--help' in argv:
        print(usage)
        exit(0)

    try:
        _, training_dataset, hidden_layers, neurons_per_layer = argv
        hidden_layers = int(hidden_layers)
        neurons_per_layer = int(neurons_per_layer)
    except (ValueError, TypeError):
        print(usage)
        exit(1)

    return read_dataset(training_dataset), hidden_layers, neurons_per_layer


def main():
    training_dataset, hidden_layers, neurons_per_layer = cli()

    net = NeuralNetwork(
        input_length=784,
        hidden_layers=hidden_layers,
        neurons_per_layer=neurons_per_layer,
        output_length=10,
        activate=sigmoid,
        activate_diff=sigmoid_derivative,
    )

    for pixels, label in training_dataset:
        net.train(pixels, label)


if __name__ == '__main__':
    main()
