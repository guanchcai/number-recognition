import numpy as np

class Layer():

    def __init__(self, minVal, maxVal, layerSize, prevSize, prevLayer, weights=None, biases=None):
        self.prevLayer = prevLayer
        self.nextLayer = None

        # weights: (neurons, previous layer)
        if weights:
            self.weights = np.array(weights)
        else:
            self.weights = np.random.uniform(minVal, maxVal, size=(layerSize, prevSize))
        # biases: (neurons, 1)

        if biases:
            self.biases = np.array(biases)
        else:
            self.biases  = np.random.uniform(minVal, maxVal, size=layerSize)
        # z = a * w + b: (batch size, neurons)
        self.z       = None
        self.values  = None
        
        # zPrime keeps track of dz/dC: (batch size, neurons)
        self.zPrime  = None
        # nablaW: (batch size, neurons)
        self.nablaW  = None
        # nablaB: (batch size, neurons)
        self.nablaB  = None
        if prevSize:
            # Not first layer
            prevLayer.nextLayer = self
    
    def calculate(self, X):
        # X: (batch size, previous layer)
        self.z = X @ self.weights.T + self.biases
        self.values = self.__sigmoid(self.z)
        return self.values # (batch size, neurons)

    def getDeri(self, nextZP):
        # nextZP: (batch size, next layer)
        if self.nextLayer:
            self.zPrime = (nextZP @ self.nextLayer.weights) * self.__sigmoidP(self.z)
        else:
            self.zPrime = nextZP * self.__sigmoidP(self.z)
        self.nablaW = self.prevLayer.values.T @ self.zPrime
        self.nablaB = self.zPrime.sum(axis=0, keepdims=True)
        return self.zPrime

    def getWeightsAndBiases(self):
        data = {
            "W": self.weights.tolist(),
            "b": self.biases.tolist()
        }
        return data

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def __reLU(self, x):
        return np.maximum(0, x)
    
    def __sigmoidP(self, x):
        s = self.__sigmoid(x)
        return s * (1 - s)
    