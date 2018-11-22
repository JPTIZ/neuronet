from textwrap import dedent
from sys import argv
from pathlib import Path

import csv
import numpy as np
import os

from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib


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


def main():
    usage = dedent(f'''
    Usage:
        {argv[0]} train <layer-sizes>... <training-dataset> <output-network>
        {argv[0]} test <trained-network> <test-dataset>
    '''.strip())

    if '-h' in argv or '--help' in argv:
        print(usage)

    elif argv[1] == 'train':
        try:
            _, _, *ls, td, on = argv
            layer_sizes = [int(l) for l in ls]
            training_dataset = read_dataset(td)
            output_network = Path(on)
        except (ValueError, TypeError):
            print(usage)
            exit(1)
        else:
            train(
                layer_sizes,
                training_dataset,
                output_network
            )

    elif argv[1] == 'test':
        try:
            _, _, tn, td = argv
            trained_network = Path(tn)
            test_dataset = read_dataset(td)
        except (ValueError, TypeError):
            print(usage)
            exit(1)
        else:
            test(trained_network, test_dataset)

    else:
        print(usage)
        exit(1)


def train(layer_sizes, training_dataset, output_network):
    print('Creating new NN.')
    net = MLPClassifier(
        solver='lbfgs',
        alpha=1e-5,
        hidden_layer_sizes=layer_sizes,
        random_state=1,
    )

    X, Y = zip(*training_dataset)
    print('Fitting...', end='')
    net.fit(X, Y)

    print('Done.\nSaving...', end='')
    joblib.dump(net, output_network)


def test(trained_network, test_dataset):
    net = joblib.load(trained_network)

    print('Testing...', end='')

    right, total = 0, 0
    for (w, o) in test_dataset:
        r = net.predict([w])
        if (r == o).all():
            right += 1
        total += 1

    print(f'Done.\nAccurracy: {right}/{total} ({100*right/total:3.2f}%)')


if __name__ == '__main__':
    main()
