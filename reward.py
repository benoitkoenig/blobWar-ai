def determineReward(state, newState):
    newAliveAllies = sum(s["alive"] for s in newState["army"])
    newAliveEnemies = sum(s["alive"] for s in newState["enemy"])
    oldAliveAllies = sum(s["alive"] for s in state["army"])
    oldAliveEnemies = sum(s["alive"] for s in state["enemy"])

    alliesKilled = oldAliveAllies - newAliveAllies
    enemiesKilled = oldAliveEnemies - newAliveEnemies
    card0used = (state["cards"]["availability"][0] == True) & (newState["cards"]["availability"][0] == False)
    card1used = (state["cards"]["availability"][1] == True) & (newState["cards"]["availability"][1] == False)

    cardUnavailable = 0
    if ((state["action"] % 3 == 1) & (state["cards"]["availability"][0] == False)) | ((state["action"] % 3 == 2) & (state["cards"]["availability"][1] == False)):
        cardUnavailable = 1 # Penalty for using a spell that is not available. Hope this will prevent wrong returns

    return enemiesKilled * 5 - alliesKilled * 5 - 1 - 3 * (card0used + card1used) - 20 * cardUnavailable

def determineEnfOfGameReward(value):
    if (value == "Victory !"):
        return 40
    if (value == "Defeat"):
        return -15
    return 2
