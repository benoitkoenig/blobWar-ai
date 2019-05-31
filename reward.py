def determineReward(state, newState):
    newAliveAllies = sum(s["alive"] for s in newState["army"])
    newAliveEnemies = sum(s["alive"] for s in newState["enemy"])
    oldAliveAllies = sum(s["alive"] for s in state["army"])
    oldAliveEnemies = sum(s["alive"] for s in state["enemy"])

    alliesKilled = oldAliveAllies - newAliveAllies
    enemiesKilled = oldAliveEnemies - newAliveEnemies

    card0used = (state["cards"][0] == True) & (newState["cards"][0] == False)
    card1used = (state["cards"][1] == True) & (newState["cards"][1] == False)

    if (alliesKilled < enemiesKilled):
        print(">>>>>>>>>>>>>>>>>>>>>>>>> Proper kill")

    return enemiesKilled * 5 - alliesKilled * 5 - 1 - 3 * (card0used + card1used)

def determineEndOfGameReward(state, newState):
    value = newState["value"]
    base_reward = determineReward(state, newState)
    if (value == "Victory !"):
        return base_reward + 40
    if (value == "Defeat"):
        return base_reward - 15
    if (value == "Timeout"):
        return base_reward - 2
    return base_reward + 2
