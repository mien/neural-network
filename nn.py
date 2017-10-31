import numpy as np


class NeuralNetwork(object):
    """docstring for NeuralNetwork"""

    def __init__(self, layers):
        self.layers = layers
        self.size = len(self.layers)
        self.weights = [np.random.randn(x, y)
                        for x, y in zip(self.layers[1:], self.layers[:-1])]
        self.bais = [np.random.randn(x, 1) for x in self.layers[1:]]

    def SGD(self, training_data, epoch, batch_size, learn_rate, test_data=None):
        for j in xrange(epoch):
            np.random.shuffle(training_data)
            # create mini batches
            mini_batches = [training_data[i:i + batch_size]
                            for i in np.arange(0, len(training_data), batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learn_rate)

    def update_mini_batch(self, mini_batch, learn_rate):
        new_weights = [np.zeros(w_s.shape, dtype=float)
                       for w_s in self.weights]
        new_baises = [np.zeros(b_s.shape, dtype=float) for b_s in self.bais]
        for x, y in mini_batch:
            self.back_prop(x, y)

    def back_prop(self, x, y):
        zs = []
        activations = [x]
        activation = x
        print('weights len ',len(self.weights))
        for b, W in zip(self.bais, self.weights):
            activation = np.dot(W, activation) + b
            zs.append(activation)
            activations.append(sigmoid(zs[-1]))
        delta = (activations[-1] - y) * sigmoid_prime(zs[-1])


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
