import math
import numpy as np

from constants import ARMY_SIZE

def orderArmyByDist(army, x, y):
    armyCopy = [2 * int(blob["alive"]) + math.hypot(blob["x"] - x, blob["y"] - y) for blob in army]
    return np.argsort(armyCopy)

def getPairFeatures(state):
    features = []
    for blob in state["army"]:
        blobFeatures = []
        for otherBlob in state["enemy"]:
            if (blob["alive"] == False) | (otherBlob["alive"] == False):
                pairFeatures = [0] * 18
            else:
                distance = math.sqrt(((blob["x"] - otherBlob["x"]) ** 2 + (blob["y"] - otherBlob["y"]) ** 2) / 2)
                statuses = [
                        int(blob["status"] == "normal"),
                        int(blob["status"] == "hat"),
                        int(blob["status"] == "ghost"),
                        int(otherBlob["status"] == "normal"),
                        int(otherBlob["status"] == "hat"),
                        int(otherBlob["status"] == "ghost"),
                ]
                pairFeatures = []
                for d in [distance, 1 - distance, (1 - distance) ** 2]:
                    pairFeatures += [d * s for s in statuses]
            blobFeatures.append(pairFeatures)
        features.append(blobFeatures)
    return features

def getFeatures(state, blob, pairFeatures):
    army = orderArmyByDist(state["army"], blob["x"], blob["y"])
    enemy = orderArmyByDist(state["enemy"], blob["x"], blob["y"])
    features = [
        1,
        int(state["cards"][0]),
        int(state["cards"][1]),
        1 - int(state["cards"][0]),
        1 - int(state["cards"][1]),
    ]
    for i in army:
        for j in enemy:
            features += pairFeatures[i][j]
    return features

def getXsaPerBlob(state):
    XsaPerBlob = []
    pairFeatures = getPairFeatures(state) # we want to calculate the features for a given pair of blobs only once
    for blob in state["army"]:
        features = getFeatures(state, blob, pairFeatures)
        Xsa = []
        zeros = np.zeros(len(features))
        for actionId in range(9):
            X = np.empty(0)
            for i in range(9):
                if (i == actionId):
                    X = np.concatenate([X, features])
                else:
                    X = np.concatenate([X, zeros])
            Xsa.append(X)
        XsaPerBlob.append(Xsa)
    return XsaPerBlob

def getBlobActions(blobId, state, bestActionId):
    actions = []
    blob = state["army"][blobId]
    ennemy = orderArmyByDist(state["enemy"], blob["x"], blob["y"])
    for i in ennemy:
        otherBlob = state["enemy"][i]
        actions += [
            [
                {"type": "server/setDestination", "idBlob": blobId, "destination": {"x": otherBlob["x"], "y": otherBlob["y"]}},
            ],
            [
                {"type": "server/setDestination", "idBlob": blobId, "destination": {"x": otherBlob["x"], "y": otherBlob["y"]}},
                {"type": "server/triggerCard", "idBlob": blobId, "destination": {"x": otherBlob["x"], "y": otherBlob["y"]}, "idCard": 0},
            ],
            [
                {"type": "server/setDestination", "idBlob": blobId, "destination": {"x": otherBlob["x"], "y": otherBlob["y"]}},
                {"type": "server/triggerCard", "idBlob": blobId, "destination": {"x": otherBlob["x"], "y": otherBlob["y"]}, "idCard": 1},
            ],
        ]
    return actions[bestActionId]
