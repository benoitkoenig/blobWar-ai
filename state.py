import math
import numpy as np

from constants import ranges

def get_scalar(v1, v2):
    length_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if (length_product == 0):
        return 0
    return np.dot(v1, v2) / length_product

def get_distance(b1, b2):
    return math.sqrt(((b1["x"] - b2["x"]) ** 2 + (b1["y"] - b2["y"]) ** 2) / 2) # divided by 2 so the max distance is 1

def get_cards_data(state):
    state_vector = [
        int(state["cards"]["availability"][0]),
        int(state["cards"]["availability"][1]),
    ]
    for i in range(3):
        state_vector += [
            int(state["cards"]["currentBlob"][0] == i),
            int(state["cards"]["currentBlob"][1] == i),
        ]
    return state_vector

def get_all_blobs_features(state):
    state_vector = []
    all_blobs = (state["army"] + state["enemy"])
    for blob in all_blobs:
        state_vector += [
            int(blob["alive"]),
            int(blob["status"] == "normal"),
            int(blob["status"] == "ghost"),
            int(blob["status"] == "hat"),
            blob["x"],
            blob["y"],
        ]
    return state_vector

def get_pair_blob_features(state):
    state_vector = []
    all_blobs = (state["army"] + state["enemy"])
    for i in range(len(all_blobs)):
        b1 = all_blobs[i]
        for j in range(i):
            b2 = all_blobs[j]
            state_vector.append(get_distance(b1, b2))
    return state_vector

def get_in_range_data(state, name):
    state_vector = []
    for blob in state["army"]:
        for other_blob in state["enemy"]:
            if name in ranges:
                distance = get_distance(blob, other_blob)
                in_range = int(distance <= ranges[name])
            else:
                in_range = 0
            state_vector.append(in_range)
    return state_vector

def get_scalar_data(state):
    state_vector = []
    for blob in state["army"]:
        if (blob["destination"] == None):
            destination_vector = [0, 0]
        else:
            destination_vector = [blob["destination"]["x"] - blob["x"], blob["destination"]["y"] - blob["y"]]
        for other_blob in state["enemy"]:
            to_other_blob_vector = [other_blob["x"] - blob["x"], other_blob["y"] - blob["y"]]
            scalar = get_scalar(destination_vector, to_other_blob_vector)
            state_vector.append(scalar)
    return state_vector

def get_state_vector(state, name):
    state_vector = [1]
    state_vector += get_cards_data(state)
    state_vector += get_all_blobs_features(state)
    state_vector += get_pair_blob_features(state)
    state_vector += get_in_range_data(state, name)
    state_vector += get_scalar_data(state)
    return state_vector
