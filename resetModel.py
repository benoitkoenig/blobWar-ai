import numpy as np
import tensorflow as tf

from constants import STATE_SIZE
from bots import Bots

tf.enable_eager_execution()

bots = Bots()
for name in bots.names:
    bots[name].save()
