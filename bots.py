import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Concatenate

from constants import STATE_SHAPE, ACTION_SIZE, learning_rate_actor, learning_rate_critic, names

class ActorModel(Model):
    def __init__(self):
        super(ActorModel, self).__init__()
        self.enemy1 = Dense(32, activation="relu")
        self.army1 = Concatenate()
        self.army2 = Dense(128, activation="relu")
        self.total1 = Concatenate()
        self.total2 = Dense(256, activation="relu")
        self.policy_logits = Dense(ACTION_SIZE, activation="linear")

    def call(self, input):
        army = tf.gather(input, 0)
        enemy = tf.gather(input, 1)
        pair = tf.gather(input, 2)

        army0 = tf.gather(army, 0)
        army1 = tf.gather(army, 1)
        army2 = tf.gather(army, 2)

        enemy0 = tf.gather(enemy, 0)
        enemy1 = tf.gather(enemy, 1)
        enemy2 = tf.gather(enemy, 2)

        pair0 = tf.gather(pair, 0)
        pair1 = tf.gather(pair, 1)
        pair2 = tf.gather(pair, 2)

        enemy0 = self.enemy1(enemy0)
        enemy1 = self.enemy1(enemy1)
        enemy2 = self.enemy1(enemy2)

        army0 = self.army1([army0, pair0, enemy0, enemy1, enemy2])
        army1 = self.army1([army1, pair1, enemy0, enemy1, enemy2])
        army2 = self.army1([army2, pair2, enemy0, enemy1, enemy2])

        army0 = self.army2(army0)
        army1 = self.army2(army1)
        army2 = self.army2(army2)

        x = self.total1([army0, army1, army2])
        x = self.total2(x)

        logits = self.policy_logits(x)
        return logits

class CriticModel(Model):
    def __init__(self):
        super(CriticModel, self).__init__()
        self.enemy1 = Dense(32, activation="relu")
        self.army1 = Concatenate()
        self.army2 = Dense(128, activation="relu")
        self.total1 = Concatenate()
        self.total2 = Dense(256, activation="relu")
        self.values = Dense(1, activation="linear")

    def call(self, input):
        army = tf.gather(input, 0)
        enemy = tf.gather(input, 1)
        pair = tf.gather(input, 2)

        army0 = tf.gather(army, 0)
        army1 = tf.gather(army, 1)
        army2 = tf.gather(army, 2)

        enemy0 = tf.gather(enemy, 0)
        enemy1 = tf.gather(enemy, 1)
        enemy2 = tf.gather(enemy, 2)

        pair0 = tf.gather(pair, 0)
        pair1 = tf.gather(pair, 1)
        pair2 = tf.gather(pair, 2)

        enemy0 = self.enemy1(enemy0)
        enemy1 = self.enemy1(enemy1)
        enemy2 = self.enemy1(enemy2)

        army0 = self.army1([army0, pair0, enemy0, enemy1, enemy2])
        army1 = self.army1([army1, pair1, enemy0, enemy1, enemy2])
        army2 = self.army1([army2, pair2, enemy0, enemy1, enemy2])

        army0 = self.army2(army0)
        army1 = self.army2(army1)
        army2 = self.army2(army2)

        x = self.total1([army0, army1, army2, enemy0, enemy1, enemy2])
        x = self.total2(x)

        values = self.values(x)
        return values

class Bot():
    def __init__(self, name):
        self.optimizer_actor = tf.train.AdamOptimizer(learning_rate_actor)
        self.optimizer_critic = tf.train.AdamOptimizer(learning_rate_critic)
        self.actor = ActorModel()
        self.critic = CriticModel()

        self.actor(tf.convert_to_tensor(np.random.rand(3, 3, 1, 6), dtype=np.float32))
        self.critic(tf.convert_to_tensor(np.random.rand(3, 3, 1, 6), dtype=np.float32))
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
