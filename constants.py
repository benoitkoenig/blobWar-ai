import math

STATE_SIZE = 78
ACTION_SIZE = 27

step_size = 8
gamma = .9
epsilon = .05
prob_flattener_factor = .005
learning_rate_actor = 2e-5
learning_rate_critic = 2e-5

names = ["RBotGhostBloc", "RBotDashDash"]

ranges = {
    "RBotGhostBloc": 0.04 / math.sqrt(2),
    "RBotDashDash": .16 / math.sqrt(2),
}
