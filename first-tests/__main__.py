from sys import argv
import csv
import numpy as np

from defaultlist import defaultlist as dlist


MSG_ME = False


AUTO_LEN = object()
def reversed_enumerate(it, length: int = AUTO_LEN, start=0):
    if length is AUTO_LEN:
        length = len(it)
    length -= start + 1
    for v in it:
        yield length, v
        length -= 1


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: float) -> float:
    return x * (1 - x)


def cost(htheta, y):
    # V1:    return np.square(htheta - y)/2
    return (
        -np.log(htheta) if y == 1 else
        -np.log(1 - htheta)
    )


def cost_function(a, xs, ys, m):
    return (
        (-1 / m) *
        sum(
            cost(a, y)
            for x, y in zip(xs, ys)
        )
    )


def update_thetas(thetas, x, y, m, alpha=1.0, lamb=0.0):
    return (
        thetas - (alpha / m) * x.T @ (cost(x*thetas) - y) +
        (lamb/m) * sum(np.square(theta) for theta in thetas[1:])
    )


def z(j, thetas, a):
    return thetas[j-1] @ a[j - 1]


def make_layer(input_length: int, neurons: int):
    weights = np.random.rand(input_length, neurons) * 2 - 1
    bias = np.random.rand(1) * 2 - 1
    return weights, bias


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
        first_layer = make_layer(neurons_per_layer, input_length)
        hidden_layers = (
            make_layer(neurons_per_layer, neurons_per_layer)
            for _ in range(hidden_layers - 1)
        )
        output_layer = make_layer(output_length, neurons_per_layer)

        self.layers = [first_layer, *hidden_layers, output_layer]
        self.activate = activate
        self.activate_diff = activate_diff
        self._gradients = None

    def prepare_training(self):
        # Setup
        self.print_unecessary_disclaimer()

        self._gradients = [
            np.zeros(len(weights))
            for weights, bias in self.layers
        ]

    def feedforward(self, input_row, expected_output):
        input_layer, *hidden_layers, output_layer = self.layers
        layers = len(self.layers)

        if MSG_ME:
            print('# Feed me >:) Feeding...')

        g = self.activate

        # -> All layers
        z = dlist(lambda *x: None)
        a = dlist(lambda *x: None)
        z[0] = np.array(input_row, dtype=float)
        a[0] = g(np.array(input_row, dtype=float))
        for j, (weights, bias) in enumerate(self.layers):
            z[j+1] = weights @ a[j]
            a[j+1] = g(z[j+1])

        z, a = np.array(z), np.array([*a])

        if MSG_ME:
            print('# I has been fed ^w^')
        return a, z

    def backpropagate(self, a, z, x, y, m=1, lamb=10):
        if MSG_ME:
            print('# Backpropping...')

        if self._gradients is None:
            print('# Oops, no gradients! Preparing for training...')
            self.prepare_training()

        last_layer = cost_function(
            a=a[-1],
            xs=[x],
            ys=y,
            m=m,
        )

        L = len(a) - 1
        delta = dlist(lambda *x: None)
        delta[L] = a[L] - y
        print(f'-> Diff: {delta[L].reshape(10, 1)}')
        for l, (weights, bias) in reversed_enumerate(reversed(self.layers[1:]),
                                                     length=len(self.layers)):
            delta[l] = (weights.T @ delta[l+1]) * a[l] * (1 - a[l])

        for l, _ in enumerate(self.layers[:-1], start=1):
            # TODO: VER QUE FUNÇÃO UTILIZAR PARA RETROPROPAGAÇÃO (HTHETA = G(THETA @ A) ???)
            if MSG_ME:
                print(f'self._gradients[{l}]: {self._gradients[l].shape}')
                print(f'delta[{l+1}]: {delta[l+1].shape}')
                print(f'a[{l}].T: {a[l].T.shape}')
            self._gradients[l] = self._gradients[l] + delta[l+1] @ a[l].T

        for l, (weights, bias) in enumerate(self.layers[1:], start=1):
            self.layers[l] = (
                weights - (1/m) * (self._gradients[l] + lamb * weights),
                bias - (1/m) * (self._gradients[l])
            )

        if MSG_ME:
            print('# Backpropped. Thats it, boys, we\'re done.')
            print(f'# Partial Result: {a[-1]}')

    def train(self, input_row, expected_output, m, lamb=10):
        '''
        HINT: A Perceptron is defined by a weights vector.
        '''
        # Feed forward
        a, z = self.feedforward(input_row, expected_output)

        # Backpropagation
        self.backpropagate(a, z, input_row, expected_output, m=m, lamb=lamb)

        self.a = a

    def print_unecessary_disclaimer(self):
        input_layer, *hidden_layers, output_layer = self.layers

        if MSG_ME:
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

    MAX_FEEDS = 60000
    for i, (pixels, label) in enumerate(training_dataset):
        print(f'# Feed {i}/{MAX_FEEDS}')
        net.train(pixels, label, m=MAX_FEEDS, lamb=0.00001)
        if MAX_FEEDS is not None and i == MAX_FEEDS:
            break

    print(f'Final a[L]:\n {net.a[-1].reshape(10, 1)}')


if __name__ == '__main__':
    main()
