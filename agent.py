import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.train import GradientDescentOptimizer
import time

from constants import ACTION_SIZE, STATE_SIZE
from reward import determineReward
from stateAndActions import getStateVector, getAction

tf.enable_eager_execution()

gamma = .9
epsilon = .005

class ActorModel(Model):
    def __init__(self):
        super(ActorModel, self).__init__()
        self.dense = Dense(24, activation='relu', use_bias=True)
        self.policy_logits = Dense(ACTION_SIZE)

    def call(self, inputs):
        x = self.dense(inputs)
        logits = self.policy_logits(x)
        return logits

class CriticModel(Model):
    def __init__(self):
        super(CriticModel, self).__init__()
        self.dense = Dense(24, activation='relu', use_bias=True)
        self.values = Dense(1)

    def call(self, inputs):
        v1 = self.dense(inputs)
        values = self.values(v1)
        return values

optActor = tf.train.GradientDescentOptimizer(1e-5)
optCritic = tf.train.GradientDescentOptimizer(1e-5)
actor = ActorModel()
critic = CriticModel()

actor(tf.convert_to_tensor(np.random.random((1, STATE_SIZE))))
critic(tf.convert_to_tensor(np.random.random((1, STATE_SIZE))))
actor.load_weights("weights/actor")
critic.load_weights("weights/critic")

class Agent:
    def __init__(self, sio, id, name):
        self.sio = sio
        self.id = id
        self.t = 0
        self.oldStateVector = None
        self.oldActionId = None
        self.oldState = None # Used only to calculate the reward
        sio.emit("learning_agent_created", {"id": id})

    def action(self, state):
        if (state["type"] == "gameStarted"):
            return # The bot does not need this signal
        if (state["type"] == "update"):
            self.t += 1

            # Due to exploratory starts, skip the first update, where instant arbitrary kills can occur and add bias
            if (self.t % 8 != 2):
                self.sio.emit("action-{}".format(self.id), [])
                return
        self.update(state)
        self.oldState = state

    def update(self, state):
        global actor, critic

        stateVector = np.array(getStateVector(state))
        stateVector = tf.convert_to_tensor(np.reshape(stateVector, [1, STATE_SIZE]))
        logits = actor(stateVector)
        probs = tf.nn.softmax(logits.numpy()[0])
        probs = np.array([(1 - epsilon) * p + epsilon / ACTION_SIZE for p in probs])
        bestActionId = np.random.choice(ACTION_SIZE, p=probs)

        reward = determineReward(self.oldState, state)
        if (reward != None):
            with tf.GradientTape(persistent=True) as tape:
                value = critic(self.oldStateVector)[0][0]
                newValue = critic(stateVector)[0][0]
                delta = tf.stop_gradient(reward + gamma * newValue - value)
                val = tf.multiply(-delta, value) # Using -delta instead of delta gets the right value, probly cause I use SGD but Sutton adds this value

                logits = actor(self.oldStateVector)[0]
                prob = tf.nn.softmax(logits)[self.oldActionId]
                prob = tf.multiply(1 - epsilon, prob) + epsilon / ACTION_SIZE
                const_prob = tf.stop_gradient(prob)
                pr = tf.multiply(-delta / const_prob, prob)

            newAliveAllies = sum(s["alive"] for s in state["army"])
            newAliveEnemies = sum(s["alive"] for s in state["enemy"])
            oldAliveAllies = sum(s["alive"] for s in self.oldState["army"])
            oldAliveEnemies = sum(s["alive"] for s in self.oldState["enemy"])
            alliesKilled = oldAliveAllies - newAliveAllies
            enemiesKilled = oldAliveEnemies - newAliveEnemies
            if ((alliesKilled == 0) & (enemiesKilled != 0)):
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Proper kill")

            gradCritic = tape.gradient(val, critic.trainable_weights)
            gradActor = tape.gradient(pr, actor.trainable_weights)

            optCritic.apply_gradients(zip(gradCritic, critic.trainable_weights))
            optActor.apply_gradients(zip(gradActor, actor.trainable_weights))

            if (state["type"] == "endOfGame"):
                actor.save_weights("weights/actor")
                critic.save_weights("weights/critic")
                print("End of game after {} episodes".format(self.t))

        self.oldStateVector = stateVector
        self.oldActionId = bestActionId

        action = getAction(state, bestActionId)
        self.sio.emit("action-{}".format(self.id), action)
