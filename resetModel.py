import numpy as np
import tensorflow as tf

from constants import STATE_SIZE, names
from bots import Bots

tf.enable_eager_execution()

bots = Bots()
for name in names:
    bots[name].save()
