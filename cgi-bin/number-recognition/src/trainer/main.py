import numpy as np
from network import *
from data_loader import loadData, loadMnist, storeWeights

def oneHotEncode(lastLayer, A):
    # A is the answer vector, size A = (batch size, 1)
    return np.eye(lastLayer)[A]

if __name__=="__main__":
    batchSize = 100
    firstLayer = 28 * 28
    lastLayer = 10
    print("Reading data...")
    X, y = loadData("mnist_train")
    network = Network(-1, 1, batchSize, [firstLayer, 32, lastLayer])
    for _ in range(6):
        a = 1
        for k in range(30):
            for i, (Xb, yb) in enumerate(loadMnist(X, y, batchSize)):
                result = network.forwardPropagate(Xb)
                answers = oneHotEncode(lastLayer, yb)
                network.backPropagate(answers)
                network.updateParams(a)
                
            print(k)
            
        print(network.cost())
        a /= 10

    storeWeights(network)