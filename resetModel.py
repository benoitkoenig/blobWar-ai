import numpy as np
import tensorflow as tf

from constants import STATE_SIZE
from models import ActorModel, CriticModel

tf.enable_eager_execution()

actor = ActorModel()
critic = CriticModel()

actor(tf.convert_to_tensor(np.random.random((1, STATE_SIZE))))
critic(tf.convert_to_tensor(np.random.random((1, STATE_SIZE))))
actor.save_weights("weights/actor")
critic.save_weights("weights/critic")
