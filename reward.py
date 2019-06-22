def calc_kills(state, new_state):
    newAliveAllies = sum(s["alive"] for s in new_state["army"])
    newAliveEnemies = sum(s["alive"] for s in new_state["enemy"])
    oldAliveAllies = sum(s["alive"] for s in state["army"])
    oldAliveEnemies = sum(s["alive"] for s in state["enemy"])

    return oldAliveAllies - newAliveAllies, oldAliveEnemies - newAliveEnemies

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

# Pb: counting this leads to a single action getting the top probability (probly bc advantage is then always > 0)
def forbidden_move(state, new_state, action_id):
    if (state["cards"][0] == False) & (action_id % 3 == 1):
        return 1
    if (state["cards"][1] == False) & (action_id % 3 == 2):
        return 1
    return 0

def determineReward(state, new_state, action_id):
    if (state == None):
        return None

    alliesKilled, enemiesKilled = calc_kills(state, new_state)
    end_bonus = calc_end_bonus(new_state)
    # forbidden = forbidden_move(state, new_state, action_id)

    return enemiesKilled * 20 - alliesKilled * 18 - 1 + end_bonus
