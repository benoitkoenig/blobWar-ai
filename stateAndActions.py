import math
import numpy as np

range_RBotGhostBloc = 0.04 / math.sqrt(2)
range_RBotDashDash = .16 / math.sqrt(2)

def getFirstBlobIdAlive(state):
    if (state["army"][0]["alive"]):
        return 0
    if (state["army"][1]["alive"]):
        return 1
    return 2

def getStateVector(state, name):
    blobId = getFirstBlobIdAlive(state)
    blob = state["army"][blobId]
    stateVector = [
        1,
        int(state["cards"][0]),
        int(state["cards"][1]),
        int(blob["status"] == "normal"),
        int(blob["status"] == "ghost"),
        int(blob["status"] == "hat"),
    ]
    for otherBlob in state["enemy"]:
        distance = math.sqrt(((blob["x"] - otherBlob["x"]) ** 2 + (blob["y"] - otherBlob["y"]) ** 2) / 2)
        if name == "RBotGhostBloc":
            in_range = int(distance <= range_RBotGhostBloc)
        elif name == "RBotDashDash":
            in_range = int(distance <= range_RBotDashDash)
        else:
            in_range = 0
        stateVector += [int(otherBlob["alive"]), distance, in_range]
    return stateVector

def getAction(state, bestActionId):
    blobId = getFirstBlobIdAlive(state)
    actions = []
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
