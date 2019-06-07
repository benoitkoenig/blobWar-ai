import numpy as np
import tensorflow as tf
from tensorflow.train import GradientDescentOptimizer

from constants import ACTION_SIZE, STATE_SIZE
from models import ActorModel, CriticModel
from reward import determineReward
from stateAndActions import getStateVector, getAction

tf.enable_eager_execution()

gamma = .9
epsilon = .1

optActor = tf.train.GradientDescentOptimizer(1e-3)
optCritic = tf.train.GradientDescentOptimizer(1e-3)
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
            if (self.t % 8 != 2): # 2 because arbitrary kills may happen at t=1 on exploratory starts
                self.sio.emit("action-{}".format(self.id), [])
                return
        self.update(state)

    def get_loss_computers(self, stateVector, reward):
        def get_value_loss():
            value = critic(self.oldStateVector)[0][0]
            newValue = critic(stateVector)[0][0]
            returnValue = reward + gamma * newValue
            self.advantage = returnValue - value # save it for policy_loss

            return self.advantage ** 2

        def get_policy_loss():
            # Warning: epsilon is not taken into account here. I dont know how it should be, though minimizing this loss should still work
            logits = actor(self.oldStateVector)
            policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=[self.oldActionId], logits=logits)[0] # The loss assuming we took the right action
            policy_loss = tf.multiply(tf.stop_gradient(self.advantage), policy_loss) # Multiply by advantage tells how much that action was right
            return policy_loss

        return get_value_loss, get_policy_loss

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
            get_value_loss, get_policy_loss = self.get_loss_computers(stateVector, reward) # calling them in the right order is important
            optCritic.minimize(get_value_loss, var_list=critic.trainable_weights)
            optActor.minimize(get_policy_loss, var_list=actor.trainable_weights)

            if (state["type"] == "endOfGame"):
                actor.save_weights("weights/actor")
                critic.save_weights("weights/critic")
                print("End of game after {} episodes".format(self.t))

        self.oldStateVector = stateVector
        self.oldActionId = bestActionId
        self.oldState = state

        action = getAction(state, bestActionId)
        self.sio.emit("action-{}".format(self.id), action)
