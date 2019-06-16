import numpy as np
import tensorflow as tf
from tensorflow.train import GradientDescentOptimizer

from constants import ACTION_SIZE, STATE_SIZE
from bots import Bots, CriticModel, ActorModel
from reward import determineReward
from stateAndActions import getStateVector, getAction

tf.enable_eager_execution()

gamma = .9
epsilon = .1
prob_flattener_factor = .01
update_interval = 10

bots = Bots()
bots.load()

class Agent:
    def __init__(self, sio, id, name):
        self.sio = sio
        self.id = id
        self.t = 0
        self.step = 0
        self.oldStateVector = None
        self.oldActionId = None
        self.oldState = None # Used only to calculate the reward

        self.bot = bots[name]

        self.local_critic = CriticModel()
        self.local_actor = ActorModel()
        self.local_critic(tf.convert_to_tensor(np.zeros((1, STATE_SIZE))))
        self.local_actor(tf.convert_to_tensor(np.zeros((1, STATE_SIZE))))
        self.local_critic.set_weights(self.bot.critic.get_weights())
        self.local_actor.set_weights(self.bot.actor.get_weights())

        self.grads_critic = []
        self.grads_actor = []

        sio.emit("learning_agent_created", {"id": id})

    def action(self, state):
        if (state["type"] == "gameStarted"):
            return # The bot does not need this signal
        if (state["type"] == "update"):
            self.t += 1
            if (self.t % 8 != 2): # 2 because arbitrary kills may happen at t=1 on exploratory starts
                self.sio.emit("action-{}".format(self.id), [])
                return
            self.step = int((self.t - 2) / 8)
        self.update(state)

    def get_loss_computers(self, stateVector, reward, isFinalState):
        def get_value_loss():
            value = self.local_critic(self.oldStateVector)[0][0]
            returnValue = reward
            if (isFinalState == False):
                newValue = self.local_critic(stateVector)[0][0]
                returnValue += gamma * newValue
            self.advantage = returnValue - value # save it for policy_loss
            return self.advantage ** 2

        def get_policy_loss():
            logits = self.local_actor(self.oldStateVector)
            s = tf.reduce_sum(tf.math.exp(logits))
            logits_with_epsilon = tf.map_fn(lambda l: tf.math.log((1 - epsilon) * tf.math.exp(l) + s * epsilon / ACTION_SIZE), logits)

            policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=[self.oldActionId], logits=logits_with_epsilon)[0] # The loss assuming we took the right action
            policy_loss = tf.multiply(tf.stop_gradient(self.advantage), policy_loss) # Multiply by advantage tells how much that action was right

            probs = tf.nn.softmax(logits[0])
            low_prob_advantage = tf.math.abs(tf.math.log(probs[self.oldActionId] * ACTION_SIZE))
            # The goal of low_prob_advantage is to give a very tiny advantage to actions with low probability,
            # so that actions with the same consequences tend to the same probability
            # I have seen entropy being used similar to this, probably has a similar output

            return policy_loss + prob_flattener_factor * low_prob_advantage

        return get_value_loss, get_policy_loss

    def update(self, state):
        stateVector = np.array(getStateVector(state))
        stateVector = tf.convert_to_tensor(np.reshape(stateVector, [1, STATE_SIZE]))
        logits = self.local_actor(stateVector)
        probs = tf.nn.softmax(logits[0]).numpy()
        probs = [(1 - epsilon) * p + epsilon / ACTION_SIZE for p in probs]
        bestActionId = np.random.choice(ACTION_SIZE, p=probs)

        reward = determineReward(self.oldState, state)
        if (reward != None):
            get_value_loss, get_policy_loss = self.get_loss_computers(stateVector, reward, (state["type"] == "endOfGame")) # calling them in the right order is important

            grads_critic = self.bot.optimizer_critic.compute_gradients(get_value_loss, self.local_critic.trainable_weights)
            grads_actor = self.bot.optimizer_actor.compute_gradients(get_policy_loss, self.local_actor.trainable_weights)

            for i in range(len(grads_critic)):
                grads_critic[i] = (grads_critic[i][0], self.bot.critic.trainable_weights[i])
            for i in range(len(grads_actor)):
                grads_actor[i] = (grads_actor[i][0], self.bot.actor.trainable_weights[i])

            self.grads_critic += grads_critic
            self.grads_actor += grads_actor

            if ((state["type"] == "endOfGame") | (self.step % update_interval == (update_interval - 1))):
                grads_critic = self.bot.optimizer_critic.apply_gradients(self.grads_critic)
                grads_actor = self.bot.optimizer_actor.apply_gradients(self.grads_actor)
                if (state["type"] == "endOfGame"):
                    self.bot.save()
                    print("End of game after {} steps".format(self.step))
                else:
                    self.grads_critic = []
                    self.grads_actor = []
                    self.local_critic.set_weights(self.bot.critic.get_weights())
                    self.local_actor.set_weights(self.bot.actor.get_weights())

        self.oldStateVector = stateVector
        self.oldActionId = bestActionId
        self.oldState = state

        action = getAction(state, bestActionId)
        self.sio.emit("action-{}".format(self.id), action)
