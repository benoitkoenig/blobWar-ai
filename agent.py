import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.train import GradientDescentOptimizer
import time

from constants import ACTION_SIZE, STATE_SIZE
from reward import determineReward, determineEndOfGameReward
from stateAndActions import getStateVector, getAction

tf.enable_eager_execution()

epsilon = .005
discounting_factor = .8

class ActorCriticModel(Model):
    def __init__(self):
        super(ActorCriticModel, self).__init__()
        self.dense1 = Dense(24, activation='relu')
        self.policy_logits = Dense(ACTION_SIZE)
        self.dense2 = Dense(24, activation='relu')
        self.values = Dense(1)

    def call(self, inputs):
        # Forward pass
        x = self.dense1(inputs)
        logits = self.policy_logits(x)
        v1 = self.dense2(inputs)
        values = self.values(v1)
        return logits, values

opt = tf.train.GradientDescentOptimizer(1e-4, use_locking=True)
global_model = ActorCriticModel()

global_model(tf.convert_to_tensor(np.random.random((1, STATE_SIZE))))
global_model.load_weights("weights/weights")

class Agent:
    def __init__(self, sio, id, name):
        self.sio = sio
        self.id = id
        self.t = 0
        # self.local_model = ActorCriticModel()
        # self.local_model.set_weights(global_model.get_weights())
        self.stateVectors = []
        self.actionIds = []
        self.rewards = []
        self.oldState = None # Used only to calculate the reward
        sio.emit("learning_agent_created", {"id": id})

    def action(self, data):
        if (data["type"] == "update"):
            self.update(data)
        if (data["type"] == "endOfGame"):
            self.endOfGame(data)

    def endOfGame(self, state):
        global global_model
        if (self.t < 2):
            # The game ended too soon. May happen due to exploratory starts
            self.sio.emit("action-" + str(self.id), []) # Needs to play one last time for the game to properly end
            return

        self.rewards.append(determineEndOfGameReward(self.oldState, state))

        with tf.GradientTape() as tape:
            total_loss = self.compute_loss()
            # computed_grads = opt.compute_gradients(self.compute_loss, self.local_model.trainable_weights)

        # grads = tape.gradient(total_loss, self.local_model.trainable_weights)
        grads = tape.gradient(total_loss, global_model.trainable_weights)

        opt.apply_gradients(zip(grads, global_model.trainable_weights))
        global_model.save_weights("weights/weights")

        print("End of game after", self.t, "episodes")
        self.sio.emit("action-" + str(self.id), []) # Needs to play one last time for the game to properly end

    def compute_loss(self):
        episodes_count = len(self.actionIds)
        discounted_rewards = np.zeros(episodes_count)
        reward_sum = 0
        for i in reversed(range(episodes_count)):
            reward_sum = self.rewards[i] + discounting_factor * reward_sum
            discounted_rewards[i] = reward_sum

        # logits, values = self.local_model(tf.convert_to_tensor(np.vstack(self.stateVectors)))
        logits, values = global_model(tf.convert_to_tensor(np.vstack(self.stateVectors)))

        advantage = tf.convert_to_tensor(np.array(discounted_rewards)[:, None]) - values
        value_loss = advantage ** 2

        policy = tf.nn.softmax(logits)
        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=policy, logits=logits)

        policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.actionIds, logits=logits)
        policy_loss *= tf.stop_gradient(advantage)
        policy_loss -= 0.01 * entropy
        total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))

        return total_loss

    def update(self, state):
        self.t += 1

        # Due to exploratory starts, skip the first update, where instant arbitrary kills can occur and add bias
        if (self.t % 20 != 2):
            self.sio.emit("action-" + str(self.id), [])
            return

        stateVector = np.array(getStateVector(state))
        # logits, _ = self.local_model(tf.convert_to_tensor(np.reshape(stateVector, [1, STATE_SIZE])))
        logits, values = global_model(tf.convert_to_tensor(np.reshape(stateVector, [1, STATE_SIZE])))
        print(values)
        probs = tf.nn.softmax(logits)
        if (np.random.random() < epsilon):
            bestActionId = np.random.choice(ACTION_SIZE)
        else:
            bestActionId = np.random.choice(ACTION_SIZE, p=probs.numpy()[0])

        if (self.oldState != None):
            self.rewards.append(determineReward(self.oldState, state))
        self.stateVectors.append(stateVector)
        self.actionIds.append(bestActionId)
        self.oldState = state

        action = getAction(state, bestActionId)
        self.sio.emit("action-" + str(self.id), action)
