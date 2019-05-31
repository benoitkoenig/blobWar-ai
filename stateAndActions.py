import math
import numpy as np

distanceToKill = 0.04 / math.sqrt(2)

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
        destinationVector = [0, 0]
        if (blob["destination"] != None):
            destinationVector = [blob["destination"]["x"] - blob["x"], blob["destination"]["y"] - blob["y"]]
        for otherBlob in state["enemy"]:
            distance = math.sqrt(((blob["x"] - otherBlob["x"]) ** 2 + (blob["y"] - otherBlob["y"]) ** 2) / 2)
            twoBlobsVector = [otherBlob["x"] - blob["x"], otherBlob["y"] - blob["y"]]
            lengthProduct = np.linalg.norm(destinationVector) * np.linalg.norm(twoBlobsVector)
            if lengthProduct == 0:
                scalar = 0
            else:
                scalar = np.dot(destinationVector, twoBlobsVector) / lengthProduct
            stateVector += [distance, float(distance <= distanceToKill), scalar ** 2]
    return stateVector

def getAction(state, bestActionId):
    actions = []
    for blobId in range(len(state["army"])):
        for otherBlob in state["enemy"]:
            ally1 = (blobId + 1) % 3
            ally2 = (blobId + 2) % 3
            actions += [
                [
                    {"type": "server/setDestination", "idBlob": blobId, "destination": {"x": otherBlob["x"], "y": otherBlob["y"]}},
                    {"type": "server/setDestination", "idBlob": ally1, "destination": {"x": state["army"][ally1]["x"], "y": state["army"][ally1]["y"]}},
                    {"type": "server/setDestination", "idBlob": ally2, "destination": {"x": state["army"][ally2]["x"], "y": state["army"][ally2]["y"]}},
                ],
                [
                    {"type": "server/setDestination", "idBlob": blobId, "destination": {"x": otherBlob["x"], "y": otherBlob["y"]}},
                    {"type": "server/triggerCard", "idBlob": blobId, "destination": {"x": otherBlob["x"], "y": otherBlob["y"]}, "idCard": 0},
                    {"type": "server/setDestination", "idBlob": ally1, "destination": {"x": state["army"][ally1]["x"], "y": state["army"][ally1]["y"]}},
                    {"type": "server/setDestination", "idBlob": ally2, "destination": {"x": state["army"][ally2]["x"], "y": state["army"][ally2]["y"]}},
                ],
                [
                    {"type": "server/setDestination", "idBlob": blobId, "destination": {"x": otherBlob["x"], "y": otherBlob["y"]}},
                    {"type": "server/triggerCard", "idBlob": blobId, "destination": {"x": otherBlob["x"], "y": otherBlob["y"]}, "idCard": 1},
                    {"type": "server/setDestination", "idBlob": ally1, "destination": {"x": state["army"][ally1]["x"], "y": state["army"][ally1]["y"]}},
                    {"type": "server/setDestination", "idBlob": ally2, "destination": {"x": state["army"][ally2]["x"], "y": state["army"][ally2]["y"]}},
                ],
            ]
    return actions[bestActionId]
