import numpy as np
import json
import os

def loadData(fileName):
    resourcePath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../res/")

    trainData = os.path.join(resourcePath, f"{fileName}.csv")

    data = np.loadtxt(trainData, delimiter=",")
    X = resize(data[:, 1:])
    y = data[:, 0].astype(np.int64)
    return X, y

def resize(n):
    return n / 255 - 0.5

def loadMnist(X, y, batchSize):
    indices = np.random.permutation(len(X))
    for i in range(0, len(X), batchSize):
        idx = indices[i:i+batchSize]
        yield X[idx], y[idx]

def storeWeights(network):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../res/weights_and_biases.json")
    data = {
        "size": [len(layer.biases) for layer in network.layers],
        "data": network.getWeightsAndBiases()
    }
    with open(path, "w") as f:
        json.dump(data, f)

def loadWeights(fileName="weights_and_biases"):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"../res/{fileName}.json")
    with open(path, "r") as f:
        return json.load(f)
