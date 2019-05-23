from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from constants import STATE_SIZE, ACTION_SIZE

model = Sequential()
model.add(Dense(24, input_dim=STATE_SIZE, activation="relu"))
# model.add(Dense(24, activation="relu"))
model.add(Dense(ACTION_SIZE, activation="linear"))
model.compile(loss="mse", optimizer=Adam(lr=.002))

model.save_weights("./weights/weights")
