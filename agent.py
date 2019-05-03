import numpy as np

from reward import determineReward, determineEnfOfGameReward
from Xsa import getXsa, getAction

W_SIZE = 5265

# w = np.zeros(W_SIZE)

Ws = {
    "BotGhostBloc": np.zeros(W_SIZE),
    "BotDashDash": np.zeros(W_SIZE),
}

np.set_printoptions(threshold=np.inf)

print("Loading Ws")
for key in Ws:
    print(key, Ws[key].min(), Ws[key].max())

alpha = 5. / W_SIZE
epsilon = 0.1
gamma = 0.9
traceDecay = 0.9

class Agent:
    def __init__(self, sio, id, name):
        sio.emit("learning_agent_created", {"id": id})
        self.sio = sio
        self.id = id
        self.name = name
        self.oldXsa = np.zeros(W_SIZE)
        self.oldState = None
        self.t = 0
        self.z = np.zeros(W_SIZE)

    def action(self, data):
        if (data["type"] == "update"):
            self.update(data)
        if (data["type"] == "endOfGame"):
            self.endOfGame(data["value"])

    def endOfGame(self, value):
        global Ws

        reward = determineEnfOfGameReward(value)
        self.z = traceDecay * gamma * self.z + self.oldXsa
        delta = reward - np.dot(Ws[self.name], self.oldXsa)
        Ws[self.name] = Ws[self.name] + alpha * delta * self.z

        print(self.name)
        print(Ws[self.name])

        self.sio.emit("action-" + str(self.id), {"type": None}) # Needs to play one last time for the game to properly end

    def update(self, state):
        global Ws
        if (self.oldState == None): # First update, initializing the state
            self.oldState = state
            return
        self.t += 1
        if (self.t % 15 != 0):
            self.sio.emit("action-" + str(self.id), {"type": None})
            return # give time for our action to have a consequence

        Xsa = getXsa(state) #Xsa is a list for each Xs for a given a
        Q = np.array([np.dot(Ws[self.name], X) for X in Xsa])
        if (np.random.uniform() < epsilon):
            bestActionId = np.random.choice(range(len(Q)))
        else:
            bestActionId = np.random.choice(np.flatnonzero(Q == Q.max()))

        reward = determineReward(self.oldState, state)
        self.z = traceDecay * gamma * self.z + self.oldXsa
        delta = reward + gamma * Q[bestActionId] - np.dot(Ws[self.name], self.oldXsa)
        Ws[self.name] = Ws[self.name] + alpha * delta * self.z

        self.oldState = state
        self.oldXsa = Xsa[bestActionId].tolist()
        bestAction = getAction(state, bestActionId)
        self.sio.emit("action-" + str(self.id), bestAction)
