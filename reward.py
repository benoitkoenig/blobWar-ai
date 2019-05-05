def determineReward(state, newState):
    newAliveAllies = sum(s["alive"] for s in newState["army"])
    newAliveEnemies = sum(s["alive"] for s in newState["enemy"])
    oldAliveAllies = sum(s["alive"] for s in state["army"])
    oldAliveEnemies = sum(s["alive"] for s in state["enemy"])

    alliesKilled = oldAliveAllies - newAliveAllies
    enemiesKilled = oldAliveEnemies - newAliveEnemies

    return enemiesKilled * 50 - alliesKilled * 50 - 1

def determineEnfOfGameReward(value):
    if (value == "Victory !"):
        return 500
    if (value == "Defeat"):
        return -50
    return -20
