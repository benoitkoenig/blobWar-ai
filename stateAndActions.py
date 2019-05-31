import math
import numpy as np

distanceToKill = 0.04 / math.sqrt(2)

def getStateVector(state):
    blob = state["army"][0]
    stateVector = [
        1.,
        float(blob["status"] == "normal"),
        float(blob["status"] == "ghost"),
        float(blob["status"] == "hat"),
    ]
    for otherBlob in state["enemy"]:
        distance = math.sqrt(((blob["x"] - otherBlob["x"]) ** 2 + (blob["y"] - otherBlob["y"]) ** 2) / 2)
        stateVector += [float(otherBlob["alive"]), distance, float(distance <= distanceToKill)]
    return stateVector

def getAction(state, bestActionId):
    blob = state["army"][0]
    blobId = 0
    actions = [
        [
            {"type": "server/setDestination", "idBlob": blobId, "destination": {"x": blob["x"], "y": blob["y"]}},
            {"type": "server/triggerCard", "idBlob": blobId, "destination": {"x": blob["x"], "y": blob["y"]}, "idCard": 0},
        ],
        [
            {"type": "server/setDestination", "idBlob": blobId, "destination": {"x": blob["x"], "y": blob["y"]}},
            {"type": "server/triggerCard", "idBlob": blobId, "destination": {"x": blob["x"], "y": blob["y"]}, "idCard": 1},
        ],
    ]
    for otherBlob in state["enemy"]:
        actions.append([{"type": "server/setDestination", "idBlob": blobId, "destination": {"x": otherBlob["x"], "y": otherBlob["y"]}}])
    return actions[bestActionId]
