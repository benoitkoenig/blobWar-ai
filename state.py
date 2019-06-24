import math
import numpy as np

from constants import ranges

def get_scalar(v1, v2):
    length_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if (length_product == 0):
        return 0
    return np.dot(v1, v2) / length_product

def get_state_vector(state, name):
    state_vector = [
        1,
        int(state["cards"][0]),
        int(state["cards"][1]),
    ]
    for blob in (state["army"] + state["enemy"]):
        state_vector += [
            int(blob["alive"]),
            int(blob["status"] == "normal"),
            int(blob["status"] == "ghost"),
            int(blob["status"] == "hat"),
        ]
    for blob in state["army"]:
        if (blob["destination"] == None):
            destination_vector = [0, 0]
        else:
            destination_vector = [blob["destination"]["x"] - blob["x"], blob["destination"]["y"] - blob["y"]]
        for other_blob in state["enemy"]:
            to_other_blob_vector = [other_blob["x"] - blob["x"], other_blob["y"] - blob["y"]]
            scalar = get_scalar(destination_vector, to_other_blob_vector)
            distance = math.sqrt(((blob["x"] - other_blob["x"]) ** 2 + (blob["y"] - other_blob["y"]) ** 2) / 2)
            if name in ranges:
                in_range = int(distance <= ranges[name])
            else:
                in_range = 0
            state_vector += [distance, in_range, scalar]
    return state_vector
