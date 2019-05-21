import math
import numpy as np

def getStateVector(state):
    stateVector = [
        1.,
        float(state["cards"][0]),
        float(state["cards"][1]),
        1. - int(state["cards"][0]),
        1. - int(state["cards"][1]),
    ]
    for blob in (state["army"] + state["enemy"]):
        stateVector += [
            float(blob["alive"]),
            float(blob["status"] == "normal"),
            float(blob["status"] == "hat"),
            float(blob["status"] == "ghost"),
        ]
    for blob in state["army"]:
        for otherBlob in state["enemy"]:
            distance = math.sqrt(((blob["x"] - otherBlob["x"]) ** 2 + (blob["y"] - otherBlob["y"]) ** 2) / 2)
            stateVector.append(distance)
    return stateVector

def getAction(state, bestActionId):
    actions = []
    for blobId in range(len(state["army"])):
        for otherBlob in state["enemy"]:
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
