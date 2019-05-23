import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.train import AdamOptimizer
import time

from constants import ACTION_SIZE, STATE_SIZE
from reward import determineReward, determineEnfOfGameReward
from stateAndActions import getStateVector, getAction

epsilon = 0.02
gamma = 0.95

model = Sequential()
model.add(Dense(24, input_dim=STATE_SIZE, activation="relu"))
# model.add(Dense(24, activation="relu"))
model.add(Dense(ACTION_SIZE, activation="linear"))
model.compile(loss="mse", optimizer=AdamOptimizer(learning_rate=.002))

model.load_weights("./weights/weights")

# Since the model is going to call fit and predict from socketio callbacks, it will be in a different thread
# Which means the current session and graph must be explicitely used
# Also, errors might occur if some operations happen in parrallel, which won't happen if only one bot trains at once
session = K.get_session()
graph = tf.get_default_graph()

np.set_printoptions(threshold=np.inf)

class Agent:
    def __init__(self, sio, id, name):
        sio.emit("learning_agent_created", {"id": id})
        self.sio = sio
        self.id = id
        self.oldState = None
        self.oldStateVector= []
        self.oldAction = None
        self.t = 0

    def action(self, data):
        if (data["type"] == "update"):
            self.update(data)
        if (data["type"] == "endOfGame"):
            self.endOfGame(data["value"])

    def endOfGame(self, value):
        global model, session, graph

        if (self.oldStateVector != []): # Due to exploratory starts, this could happen
            with session.as_default():
                with graph.as_default(): # Inside the socketio event, the thread is different, generating a new tensorflow session
                    reward = determineEnfOfGameReward(value)
                    target_f = model.predict(self.oldStateVector)
                    target_f[0][self.oldAction] = reward
                    model.fit(self.oldStateVector, target_f, epochs=1, verbose=0)
                    model.save_weights("./weights/weights")
        print("End of game after", self.t, "episodes")
        self.sio.emit("action-" + str(self.id), []) # Needs to play one last time for the game to properly end

    def update(self, state):
        global model, session, graph

        self.t += 1

        if(self.t == 1): # Due to exploratory starts, skip the first update, where instant arbitrary kills can occur and add bias
            self.sio.emit("action-" + str(self.id), [])
            return

        if (self.t % 8 != 2):
            self.sio.emit("action-" + str(self.id), [])
            return

        with session.as_default():
            with graph.as_default(): # Inside the socketio event, the thread is different, generating a new tensorflow session
                stateVector = np.array(getStateVector(state))
                stateVector = np.reshape(stateVector, [1, STATE_SIZE])
                actionValues = model.predict(stateVector)[0]
                maxActionValue = actionValues.max()

                if (self.oldStateVector != []):
                    reward = determineReward(self.oldState, state, self.oldAction)
                    target = reward + gamma * maxActionValue
                    target_f = model.predict(self.oldStateVector)
                    target_f[0][self.oldAction] = target
                    model.fit(self.oldStateVector, target_f, epochs=1, verbose=0)

        if (np.random.uniform() < epsilon):
            bestActionId = np.random.choice(ACTION_SIZE)
        else:
            bestActionId = np.random.choice(np.flatnonzero(actionValues == maxActionValue))

        self.oldStateVector = stateVector
        self.oldState = state
        self.oldAction = bestActionId

        action = getAction(state, bestActionId)
        self.sio.emit("action-" + str(self.id), action)
