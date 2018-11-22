from sys import argv
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


# TODO: separar em subcomandos: `train` e `test`.
def cli():
    usage = (
        f'Usage: {argv[0]} '
        '<training-dataset> '
        '<hidden-layers> '
        '<neurons-per-layer>'
        '<test-dataset> '
    )

    if '-h' in argv or '--help' in argv:
        print(usage)
        exit(0)

    try:
        (_,
            training_dataset,
            hidden_layers,
            neurons_per_layer,
            test_dataset) = argv
        hidden_layers = int(hidden_layers)
        neurons_per_layer = int(neurons_per_layer)
    except (ValueError, TypeError):
        print(usage)
        exit(1)

    return (
        read_dataset(training_dataset),
        hidden_layers,
        neurons_per_layer,
        read_dataset(test_dataset),
    )


def main():
    training_dataset, hidden_layers, neurons_per_layer, test_dataset = cli()
    TRAINED_FILE = 'trained.net'

    if os.path.exists(TRAINED_FILE):
        print('Using previously trained NN.')
        net = joblib.load(TRAINED_FILE)
    else:
        print('Creating new NN.')
        net = MLPClassifier(
            solver='lbfgs',
            alpha=1e-5,
            hidden_layer_sizes=(10, 10),
            random_state=1,
        )

        X, Y = zip(*training_dataset)
        print('Fitting...', end='')
        net.fit(X, Y)

        print('Done.\nSaving...', end='')
        joblib.dump(net, TRAINED_FILE)
        print('Done.')

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
