import math
import numpy as np

def getAction(state, actionId):
    actions = []
    for blobId in range(len(state["army"])):
        for otherBlob in state["enemy"]:
            actions.append({"type": "server/setDestination", "idBlob": blobId, "destination": {"x": otherBlob["x"], "y": otherBlob["y"]}})
            actions.append({"type": "server/triggerCard", "idBlob": blobId, "destination": {"x": otherBlob["x"], "y": otherBlob["y"]}, "idCard": 0})
            actions.append({"type": "server/triggerCard", "idBlob": blobId, "destination": {"x": otherBlob["x"], "y": otherBlob["y"]}, "idCard": 1})
    return actions[actionId]

def getFeatures(state):
    features = []
    features += [
        int(state["cards"][0]),
        int(state["cards"][1]),
        1 - int(state["cards"][0]),
        1 - int(state["cards"][1]),
    ]
    for blob in (state["army"] + state["enemy"]):
        features.append(1 - int(blob["alive"]))
    for blob in state["army"]:
        for otherBlob in state["enemy"]:
            distance = math.sqrt(((blob["x"] - otherBlob["x"]) ** 2 + (blob["y"] - otherBlob["y"]) ** 2) / 2)
            if (blob["destination"] == None):
                distanceToDest = distance
            else:
                distanceToDest = math.sqrt((blob["destination"]["x"] - otherBlob["x"]) ** 2 + (blob["destination"]["y"] - otherBlob["y"]) ** 2) / 2
            for d in [distance, (1 - distance) ** 2, (1 - distanceToDest) ** 2]:
                features += [
                    int(otherBlob["alive"]) * int(blob["alive"]) * d,
                    int(otherBlob["alive"]) * int(blob["alive"]) * d * int(blob["status"] == "normal"),
                    int(otherBlob["alive"]) * int(blob["alive"]) * d * int(blob["status"] == "hat"),
                    int(otherBlob["alive"]) * int(blob["alive"]) * d * int(blob["status"] == "ghost"),
                    int(otherBlob["alive"]) * int(blob["alive"]) * d * int(otherBlob["status"] == "normal"),
                    int(otherBlob["alive"]) * int(blob["alive"]) * d * int(otherBlob["status"] == "hat"),
                    int(otherBlob["alive"]) * int(blob["alive"]) * d * int(otherBlob["status"] == "ghost"),
                ]
    return features

def getXsa(state):
    features = getFeatures(state)
    Xsa = []
    zeros = np.zeros(len(features))
    for actionId in range(27):
        X = np.empty(0)
        for i in range(27):
            if (i == actionId):
                X = np.concatenate([X, features])
            else:
                X = np.concatenate([X, zeros])
        Xsa.append(X)
    return Xsa
