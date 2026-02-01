from src.trainer.layer import Layer
import numpy as np

class Network():
    def __init__(self, minVal, maxVal, batchSize, layerSizes, load=None):
        prev = 0
        self.layers = []
        self.batchSize = batchSize
        prevLayer = None
        if load:
            if not (load["size"] == layerSizes):
                raise ValueError(f"Size provided doesn't match the size in dataset, expected {load['size']}, got {layerSizes} instead.")

        data = load["data"] if load else None
        for i, layerSize in enumerate(layerSizes):
            # Skip the 0th layer because it is unrecorded
            layerData = data[i - 1] if data and i != 0 else {}
            prevLayer = Layer(minVal, maxVal, layerSize, prev, prevLayer, layerData.get("W"), layerData.get("b"))
            self.layers.append(prevLayer)
            prev = layerSize
    
    def forwardPropagate(self, inputMatrix):
        M = inputMatrix
        self.layers[0].values = inputMatrix
        for layer in self.layers[1:]:
            M = layer.calculate(M)
        
        self.layers[-1].values = self.__softMax(self.layers[-1].z)
        
        return self.layers[-1].values

    def cost(self):
        return self.__crossEnt(self.layers[-1].values)
    
    def backPropagate(self, answerMatrix):
        # Answer matrix: (batch size, last layer)
        prevP = self.layers[-1].values - answerMatrix
        for i in range(len(self.layers) - 1, 0, -1):
            prevP = self.layers[i].getDeri(prevP)
    
    def updateParams(self, eta):
        for i, layer in enumerate(self.layers[1:]):
            layer.weights -= layer.nablaW.T / self.batchSize * eta
            layer.biases  -= layer.nablaB.reshape(-1) / self.batchSize * eta
    
    def getWeightsAndBiases(self):
        return [l.getWeightsAndBiases() for l in self.layers[1:]]

    def __softMax(self, M):
        # M: (batch size, final layer)
        z_shifted = M - np.max(M, axis=1, keepdims=True)  # stability trick
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def __crossEnt(self, Y):
        eps = 1e-9
        return -np.mean(np.sum(Y * np.log(self.layers[-1].values + eps), axis=1))

    def __quadMean(self, Y):
        return (self.layers[-1].values - Y) ** 2
    
    def __quadMeanP(self, Y):
        return 2 * (self.layers[-1].values - Y)