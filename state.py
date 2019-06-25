import math
import numpy as np

from constants import ranges

def get_scalar(v1, v2):
    length_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if (length_product == 0):
        return 0
    return np.dot(v1, v2) / length_product

def get_angle(b1, b2):
    v = [b2["x"] - b1["x"], b2["y"] - b1["y"]]
    norm = np.linalg.norm(v)
    if (norm == 0):
        return 0
    x = v[0] / norm
    angle = np.math.acos(x)
    if (v[1] < 0):
        angle = -angle
    return angle

def get_distance(b1, b2):
    return math.sqrt(((b1["x"] - b2["x"]) ** 2 + (b1["y"] - b2["y"]) ** 2) / 2) # divided by 2 so the max distance is 1

def get_state_vector(state, name):
    state_vector = [
        1,
        int(state["cards"]["availability"][0]),
        int(state["cards"]["availability"][1]),
    ]
    for i in range(3):
        state_vector += [
            int(state["cards"]["currentBlob"][0] == i),
            int(state["cards"]["currentBlob"][1] == i),
        ]
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
    for i in range(len(all_blobs)):
        for j in range(i):
            b1 = all_blobs[i]
            b2 = all_blobs[j]
            state_vector += [
                get_distance(b1, b2),
                get_angle(b1, b2),
            ]
            for k in range(j):
                combinations = [[i, j, k], [j, k, i], [k, i, j]]
                for c in combinations:
                    b1 = all_blobs[c[0]]
                    b2 = all_blobs[c[1]]
                    b3 = all_blobs[c[2]]
                    state_vector.append(get_angle(b1, b2) - get_angle(b1, b3))
    for blob in state["army"]:
        if (blob["destination"] == None):
            destination_vector = [0, 0]
        else:
            destination_vector = [blob["destination"]["x"] - blob["x"], blob["destination"]["y"] - blob["y"]]
        for other_blob in state["enemy"]:
            to_other_blob_vector = [other_blob["x"] - blob["x"], other_blob["y"] - blob["y"]]
            scalar = get_scalar(destination_vector, to_other_blob_vector)
            if name in ranges:
                distance = get_distance(blob, other_blob)
                in_range = int(distance <= ranges[name])
            else:
                in_range = 0
            state_vector += [in_range, scalar]
    return state_vector
