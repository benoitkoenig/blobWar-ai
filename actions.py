import numpy as np

from constants import ACTION_SIZE

def get_action_data(id):
    per_blob_action_size = int(ACTION_SIZE / 3)
    per_pair_action_size = int(per_blob_action_size / 3)

    id_blob = int(id / per_blob_action_size)
    id_other_blob = int((id % per_blob_action_size) / per_pair_action_size)

    id_action = id % per_pair_action_size
    if (id_action == 1):
        id_card = 0
    elif (id_action == 2):
        id_card = 1
    else:
        id_card = None

    return id_blob, id_other_blob, id_action, id_card

def get_action(state, best_action_id):
    id_blob, id_other_blob, id_action, _ = get_action_data(best_action_id)
    blob = state["army"][id_blob]
    other_blob = state["enemy"][id_other_blob]
    base_vector = np.array([other_blob["x"] - blob["x"], other_blob["y"] - blob["y"]])
    base_norm = np.linalg.norm(base_vector)
    if base_norm != 0:
        base_vector = base_vector / np.linalg.norm(base_vector) * 2
    destination = {"x": blob["x"] + base_vector[0], "y": blob["y"] + base_vector[1]}
    actions = [
        [
            {"type": "server/setDestination", "idBlob": id_blob, "destination": destination},
        ],
        [
            {"type": "server/setDestination", "idBlob": id_blob, "destination": destination},
            {"type": "server/triggerCard", "idBlob": id_blob, "destination": destination, "idCard": 0},
        ],
        [
            {"type": "server/setDestination", "idBlob": id_blob, "destination": destination},
            {"type": "server/triggerCard", "idBlob": id_blob, "destination": destination, "idCard": 1},
        ],
    ]
    return actions[id_action]
