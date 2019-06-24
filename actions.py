def get_action(state, best_action_id):
    id_blob = int(best_action_id / 9)
    other_blob = state["enemy"][int((best_action_id % 9) / 3)]
    action_index = best_action_id % 3
    actions = [
        [
            {"type": "server/setDestination", "idBlob": id_blob, "destination": {"x": other_blob["x"], "y": other_blob["y"]}},
        ],
        [
            {"type": "server/setDestination", "idBlob": id_blob, "destination": {"x": other_blob["x"], "y": other_blob["y"]}},
            {"type": "server/triggerCard", "idBlob": id_blob, "destination": {"x": other_blob["x"], "y": other_blob["y"]}, "idCard": 0},
        ],
        [
            {"type": "server/setDestination", "idBlob": id_blob, "destination": {"x": other_blob["x"], "y": other_blob["y"]}},
            {"type": "server/triggerCard", "idBlob": id_blob, "destination": {"x": other_blob["x"], "y": other_blob["y"]}, "idCard": 1},
        ],
    ]
    return actions[action_index]
