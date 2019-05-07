import numpy as np

from constants import W_SIZE

W_ghost_bloc = np.memmap("W_ghost_bloc", dtype='float32', mode='w+', shape=(W_SIZE))
W_dash_dash = np.memmap("W_dash_dash", dtype='float32', mode='w+', shape=(W_SIZE))
W_ghost_bloc[:] = np.zeros(W_SIZE)[:]
W_dash_dash[:] = np.zeros(W_SIZE)[:]
