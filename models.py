from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

from constants import ACTION_SIZE

class ActorModel(Model):
    def __init__(self):
        super(ActorModel, self).__init__()
        self.dense = Dense(24, activation='relu')
        self.policy_logits = Dense(ACTION_SIZE)

    def call(self, inputs):
        x = self.dense(inputs)
        logits = self.policy_logits(x)
        return logits

class CriticModel(Model):
    def __init__(self):
        super(CriticModel, self).__init__()
        self.dense = Dense(24, activation='relu')
        self.values = Dense(1)

    def call(self, inputs):
        v1 = self.dense(inputs)
        values = self.values(v1)
        return values
