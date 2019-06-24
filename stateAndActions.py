import math
import numpy as np

range_RBotGhostBloc = 0.04 / math.sqrt(2)
range_RBotDashDash = .16 / math.sqrt(2)

def get_first_blob_id_alive(state):
    if (state["army"][0]["alive"]):
        return 0
    if (state["army"][1]["alive"]):
        return 1
    return 2

def get_state_vector(state, name):
    blobId = get_first_blob_id_alive(state)
    blob = state["army"][blobId]
    state_vector = [
        1,
        int(state["cards"][0]),
        int(state["cards"][1]),
        int(blob["status"] == "normal"),
        int(blob["status"] == "ghost"),
        int(blob["status"] == "hat"),
    ]
    for other_blob in state["enemy"]:
        distance = math.sqrt(((blob["x"] - other_blob["x"]) ** 2 + (blob["y"] - other_blob["y"]) ** 2) / 2)
        if name == "RBotGhostBloc":
            in_range = int(distance <= range_RBotGhostBloc)
        elif name == "RBotDashDash":
            in_range = int(distance <= range_RBotDashDash)
        else:
            in_range = 0
        state_vector += [int(other_blob["alive"]), distance, in_range]
    return state_vector

def get_action(state, bestActionId):
    blobId = get_first_blob_id_alive(state)
    actions = []
    for other_blob in state["enemy"]:
        actions += [
            [
                {"type": "server/setDestination", "idBlob": blobId, "destination": {"x": other_blob["x"], "y": other_blob["y"]}},
            ],
            [
                {"type": "server/setDestination", "idBlob": blobId, "destination": {"x": other_blob["x"], "y": other_blob["y"]}},
                {"type": "server/triggerCard", "idBlob": blobId, "destination": {"x": other_blob["x"], "y": other_blob["y"]}, "idCard": 0},
            ],
            [
                {"type": "server/setDestination", "idBlob": blobId, "destination": {"x": other_blob["x"], "y": other_blob["y"]}},
                {"type": "server/triggerCard", "idBlob": blobId, "destination": {"x": other_blob["x"], "y": other_blob["y"]}, "idCard": 1},
            ],
        ]
    return actions[bestActionId]
