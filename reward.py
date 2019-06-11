def determineReward(state, newState):
    if (state == None):
        return None

    if (newState["type"] != "endOfGame"):
        end_bonus = 0
    elif (newState["value"] == "Victory !"):
        end_bonus = 5
    elif (newState["value"] == "Defeat"):
        end_bonus = 2
    elif (newState["value"] == "Timeout"):
        end_bonus = 0
    else:
        end_bonus = 4

    newAliveAllies = sum(s["alive"] for s in newState["army"])
    newAliveEnemies = sum(s["alive"] for s in newState["enemy"])
    oldAliveAllies = sum(s["alive"] for s in state["army"])
    oldAliveEnemies = sum(s["alive"] for s in state["enemy"])

    if ((oldAliveAllies == newAliveAllies) & (oldAliveEnemies != newAliveEnemies)):
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Proper kill")

    alliesKilled = oldAliveAllies - newAliveAllies
    enemiesKilled = oldAliveEnemies - newAliveEnemies

    card0used = (state["cards"][0] == True) & (newState["cards"][0] == False)
    card1used = (state["cards"][1] == True) & (newState["cards"][1] == False)

    return enemiesKilled * 20 - alliesKilled * 18 - 1 - 3 * (card0used + card1used) + end_bonus
