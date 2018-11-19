from sys import argv
import csv
import numpy as np

from defaultlist import defaultlist as dlist


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: float) -> float:
    return x * (1 - x)


def make_layer(input_length: int, neurons: int):
    weights = np.random.rand(input_length, neurons) * 2 - 1
    biases = np.random.rand(neurons) * 2 - 1
    return weights, biases


def h(x, thetas):
    return thetas[0] + thetas.T @ x


def cost(htheta, y):
    # V1:    return np.square(htheta - y)/2
    return (
        -np.log(htheta) if y == 1 else
        -np.log(1 - htheta)
    )


def cost_function(thetas, xs, ys, m):
    return (
        (1 / m) *
        sum(
            cost(h(x), y)
            for x, y in zip(xs, ys)
        )
    )


def update_thetas(thetas, x, y, m, alpha=1.0):
    return thetas - (alpha / m) * x.T @ (cost(x*thetas) - y)


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
        first_layer = make_layer(neurons_per_layer, input_length)[0]
        hidden_layers = (
            make_layer(neurons_per_layer, neurons_per_layer)[0]
            for _ in range(hidden_layers - 1)
        )
        output_layer = make_layer(output_length, neurons_per_layer)[0]

        self.layers = [first_layer, *hidden_layers, output_layer]
        self.activate = activate
        self.activate_diff = activate_diff

    def feedforward(self, input_row, expected_output):
        input_layer, *hidden_layers, output_layer = self.layers
        print('# Feed me >:) Feeding...')

        # -> Input layer
        a = dlist(lambda *a: dlist(lambda *b: float))
        z = dlist(lambda *a: dlist(lambda *b: float))
        for j, weights in enumerate(input_layer):
            z[0][j] = sum(i*weights[k]
                          for k, i in enumerate(input_row))
            a[0][j] = self.activate(z[0][j])

        # -> Hidden layers
        for l, layer in enumerate(hidden_layers, start=1):
            for j, weights in enumerate(layer):
                z[l][j] = sum(a*weights[k]
                              for k, a in enumerate(a[l-1]))
                a[l][j] = self.activate(z[l][j])

        # -> Output layer
        lay = len(self.layers) - 1
        for j, weights in enumerate(output_layer):
            z[lay][j] = sum(a*weights[k]
                            for k, a in enumerate(a[lay-1]))
            a[lay][j] = self.activate(z[lay][j])

        print('# I has been fed ^w^')
        return a, z

    def backpropagate(self, a, z, expected_output):
        print('# Backpropping...')

        input_layer, *hidden_layers, output_layer = self.layers

        # -> C(o)
        error = sum(np.square(result - expected)
                    for result, expected in zip(a[-1], expected_output))
        print(f'> general error: {error}')

        # TODO: Keep going with the backprop, bro :)
        # Compute last layer derivate of C in terms of a
        lay = len(self.layers) - 1
        dc_da = dlist(lambda *a: 0.0)
        dc_da[lay] = [
            a_k - y
            for a_k, y in zip(a[lay], expected_output)
        ]

        # Compute derivate of C in terms of w
        # --> Meanwhile, compute derivate of C in terms of a
        dc_dw = dlist(lambda *a: dlist(lambda *b: dlist(lambda *c: 0.0)))
        for _l, layer in enumerate(reversed(self.layers[:-1]), start=2):
            lay = len(self.layers) - _l
            print(f'working on layer {lay}/{len(self.layers)-1}')
            print(f'  |-> nodes: {len(self.layers[lay])}')
            print(f'  |-> weights/node: {len(self.layers[lay][0])}')
            print(f'  `-> activations: {len(a[lay])}')
            for j, weights in enumerate(layer):
                for k, w in enumerate(weights):
                    if lay != len(self.layers) - 1:
                        _theta = self.layers[lay+1][k]
                        zeta = z[lay]
                        s = sum(
                            theta *
                            self.activate_diff(zeta[j]) *
                            dc_da[lay+1]
                            for _j, theta in enumerate(_theta)
                        )
                        dc_da[lay] = s
                    dc_dw[lay][j][k] = (
                        a[lay-1][k]
                        * self.activate_diff(z[lay][j])
                        * dc_da[lay]
                    )

        # Update weights
        for l, layer in enumerate(hidden_layers):
            for j, weights in enumerate(layer):
                for k, w in enumerate(weights):
                    weights[j][k] = weights[j][k] - dc_dw[l][j][k]

    def train(self, input_row, expected_output):
        '''
        HINT: A Perceptron is defined by a weights vector.
        '''
        # Setup
        input_layer, *hidden_layers, output_layer = self.layers

        print(f'''
              .{"-"*55}.
              | Training the following neural network:               {' '* 0} |
              | - Layers: {len(self.layers)}                         {' '*17} |
              |    `-> Hidden: {len(hidden_layers):<3}               {' '*20} |
              | - Neurons in:                                        {' '* 0} |
              |    |-> Input  Layer: {len(input_layer):<3}           {' '*18} |
              |    |-> Hidden Layer: {len(hidden_layers[0]):<3}      {' '*23} |
              |    `-> Output Layer: {len(output_layer):<3}          {' '*19} |
              | - Weights for each neuron in:                        {' '* 0} |
              |    |-> Input  Layer: {len(input_layer[0]):<3}        {' '*21} |
              |    |-> Hidden Layer: {len(hidden_layers[0][0]):<3}   {' '*26} |
              |    `-> Output Layer: {len(output_layer[0]):<3}       {' '*22} |
              `{"-"*55}'
              '''
              )

        # Feed forward
        a, z = self.feedforward(input_row, expected_output)

        # Backpropagation
        self.backpropagate(a, z, expected_output)


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
