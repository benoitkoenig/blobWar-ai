import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

from constants import STATE_SIZE, ACTION_SIZE, learning_rate_actor, learning_rate_critic, names

# Extending Model is overkill as a Sequential would be enough

class ActorModel(Model):
    def __init__(self):
        super(ActorModel, self).__init__()
        self.dense1 = Dense(1024, activation='relu')
        self.dense2 = Dense(1024, activation='relu')
        self.dense3 = Dense(1024, activation='relu')
        self.dense4 = Dense(1024, activation='relu')
        self.policy_logits = Dense(ACTION_SIZE, activation='linear')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        logits = self.policy_logits(x)
        return logits

class CriticModel(Model):
    def __init__(self):
        super(CriticModel, self).__init__()
        self.dense1 = Dense(1024, activation='relu')
        self.dense2 = Dense(1024, activation='relu')
        self.dense3 = Dense(1024, activation='relu')
        self.dense4 = Dense(1024, activation='relu')
        self.values = Dense(1, activation='linear')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        values = self.values(x)
        return values

class Bot():
    def __init__(self, name):
        self.optimizer_actor = tf.train.AdamOptimizer(learning_rate_actor, use_locking=True)
        self.optimizer_critic = tf.train.AdamOptimizer(learning_rate_critic, use_locking=True)
        self.actor = ActorModel()
        self.critic = CriticModel()

        self.actor(tf.convert_to_tensor(np.random.random((1, STATE_SIZE))))
        self.critic(tf.convert_to_tensor(np.random.random((1, STATE_SIZE))))
        self.name = name

    def save(self):
        self.actor.save_weights("weights/actor-{}".format(self.name))
        self.critic.save_weights("weights/critic-{}".format(self.name))

    def load(self):
        self.actor.load_weights("weights/actor-{}".format(self.name))
        self.critic.load_weights("weights/critic-{}".format(self.name))

class Bots(dict):
    def __init__(self):
        for name in names:
            self[name] = Bot(name)

    def load(self):
        for _, bot in self.items():
            bot.load()
