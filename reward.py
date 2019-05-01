def determineReward(state, newState):
    newAliveAllies = sum(s["alive"] for s in newState["army"])
    newAliveEnemies = sum(s["alive"] for s in newState["enemy"])
    oldAliveAllies = sum(s["alive"] for s in state["army"])
    oldAliveEnemies = sum(s["alive"] for s in state["enemy"])
    if (oldAliveAllies - newAliveAllies == 1 & oldAliveEnemies - newAliveEnemies == 1):
        return 50
    elif (oldAliveEnemies - newAliveEnemies == 1):
        return 100
    else:
        return -1

def determineEnfOfGameReward(value):
    if (value == "Victory !"):
        return 500
    if (value == "Defeat"):
        return 0
    return 100
