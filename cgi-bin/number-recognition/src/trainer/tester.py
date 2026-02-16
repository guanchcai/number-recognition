import numpy as np
from network import *
from data_loader import loadData, loadMnist, loadWeights

if __name__=="__main__":
    batchSize = 10000
    firstLayer = 28 * 28
    lastLayer = 10
    print("Reading data...")
    X, y = loadData("mnist_test")
    savedNetwork = loadWeights()
    network = Network(-1, 1, batchSize, [firstLayer, 32, lastLayer], savedNetwork)

    result = network.forwardPropagate(X)
    pred = result.argmax(axis=1)
    correct = (pred == y).sum()
    print(f"Successful, score: {correct}/{len(y)}, {correct / len(y)}% confidence (lower the better) {network.cost().mean()}")