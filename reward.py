from actions import get_action_data

def calc_kills(state, new_state):
    new_alive_allies = sum(s["alive"] for s in new_state["army"])
    new_alive_enemies = sum(s["alive"] for s in new_state["enemy"])
    old_alive_allies = sum(s["alive"] for s in state["army"])
    old_alive_enemies = sum(s["alive"] for s in state["enemy"])

    return old_alive_allies - new_alive_allies, old_alive_enemies - new_alive_enemies

def calc_end_bonus(new_state):
    if (new_state["type"] != "endOfGame"):
        return 0
    if (new_state["value"] == "Victory !"):
        return 5
    if (new_state["value"] == "Defeat"):
        return 2
    if (new_state["value"] == "Timeout"):
        return -10
    return 4

def forbidden_move(state, action_id):
    id_blob, _, _, id_card = get_action_data(action_id)
    if id_blob == None:
        return 0
    if (state["army"][id_blob]["alive"] == False):
        return 1
    if (id_card != None):
        if state["cards"]["availability"][id_card] == False:
            return 1
    return 0

def determine_reward(state, new_state, action_id):
    if (state == None):
        return None

    alliesKilled, enemiesKilled = calc_kills(state, new_state)
    end_bonus = calc_end_bonus(new_state)
    forbidden = forbidden_move(state, action_id)

    return enemiesKilled * 20 - alliesKilled * 18 - 1 + end_bonus - 10 * forbidden
