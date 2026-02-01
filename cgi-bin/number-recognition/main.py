#!/usr/bin/env python3

import sys
import json
import numpy as np
from src.trainer.network import *
from src.trainer.data_loader import loadData, loadWeights

# Read raw POST data
raw_data = sys.stdin.read()

# Set content type to JSON
print("Content-Type: application/json\n")

response = {}  # This will be returned as JSON

def predictNumber(X):
    batchSize = 10000
    firstLayer = 28 * 28
    lastLayer = 10
    savedNetwork = loadWeights("weights_and_biases_32")
    network = Network(-1, 1, batchSize, [firstLayer, 32, lastLayer], savedNetwork)
    result = network.forwardPropagate(np.array(X).reshape(1, -1))
    pred = result.argmax(axis=1)
    return int(pred)

if raw_data:
    try:
        data = json.loads(raw_data)
        # Example: respond with something based on input
        response["status"] = "success"
        response["received"] = data
        response["result"] = predictNumber(data.get("data"))  # just an example
    except json.JSONDecodeError:
        response["status"] = "error"
        response["message"] = "Invalid JSON"
else:
    response["status"] = "error"
    response["message"] = "No data received"

# Convert the response dictionary to JSON and print it
print(json.dumps(response))
