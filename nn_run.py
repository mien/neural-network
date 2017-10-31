from nn import NeuralNetwork
from mnist_loader import load_data_wrapper

neural = NeuralNetwork([784,30,10])

train_data, val_data, test_data = load_data_wrapper()

neural.SGD(train_data,30,30,1.0)
