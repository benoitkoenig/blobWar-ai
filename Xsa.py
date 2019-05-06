import math
import numpy as np

def calculateAngle(a, b, c):
    vector1 = (a["x"] - b["x"]) + (a["y"] - b["y"])* 1j
    vector2 = (c["x"] - b["x"]) + (c["y"] - b["y"])* 1j
    angle1 = np.angle(vector1)
    angle2 = np.angle(vector2)
    angle = (angle1 - angle2 + 2) % 2
    return angle

def getAction(state, actionId):
    actions = []
    for blobId in range(len(state["army"])):
        for otherBlob in state["enemy"]:
            actions.append({"type": "server/setDestination", "idBlob": blobId, "destination": {"x": otherBlob["x"], "y": otherBlob["y"]}})
            actions.append({"type": "server/triggerCard", "idBlob": blobId, "destination": {"x": otherBlob["x"], "y": otherBlob["y"]}, "idCard": 0})
            actions.append({"type": "server/triggerCard", "idBlob": blobId, "destination": {"x": otherBlob["x"], "y": otherBlob["y"]}, "idCard": 1})
    return actions[actionId]

def getFeatures(state):
    allBlobs = state["army"] + state["enemy"]
    features = [
        int(state["cards"][0]),
        int(state["cards"][1]),
        1 - int(state["cards"][0]),
        1 - int(state["cards"][1]),
    ]
    for blob in allBlobs:
        features += [
            1 - int(blob["alive"]),
            int(blob["status"] == "normal"),
            int(blob["status"] == "hat"),
            int(blob["status"] == "ghost"),
        ]
    for blob in state["army"]:
        for otherBlob in state["enemy"]:
            if (blob["alive"] == False) | (otherBlob["alive"] == False):
                features += [0] * 21
            else:
                distance = math.sqrt(((blob["x"] - otherBlob["x"]) ** 2 + (blob["y"] - otherBlob["y"]) ** 2) / 2)
                if (blob["destination"] == None):
                    distanceToDest = distance
                else:
                    distanceToDest = math.sqrt(((blob["destination"]["x"] - otherBlob["x"]) ** 2 + (blob["destination"]["y"] - otherBlob["y"]) ** 2) / 2)
                statuses = [
                        1,
                        int(blob["status"] == "normal"),
                        int(blob["status"] == "hat"),
                        int(blob["status"] == "ghost"),
                        int(otherBlob["status"] == "normal"),
                        int(otherBlob["status"] == "hat"),
                        int(otherBlob["status"] == "ghost"),
                ]
                for d in [distance, (1 - distance) ** 2, (1 - distanceToDest) ** 2]:
                    features += [d * s for s in statuses]
    for i in range(6):
        for index1 in range(6):
            for k in range(5 - index1):
                index2 = index1 + k + 1
                if (i != index1) & (i != index2):
                    if (allBlobs[index1]["alive"] == False) | (allBlobs[i]["alive"] == False) | (allBlobs[index2]["alive"] == False):
                        features.append(0)
                    else:
                        angle = calculateAngle(allBlobs[index1], allBlobs[i], allBlobs[index2])
                        features.append((1 - abs(angle)) ** 2)
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
