import math

STATE_SIZE = 54
ACTION_SIZE = 27

gamma = .9
epsilon = .1
prob_flattener_factor = .002
update_interval = 10
learning_rate_actor = 1e-3
learning_rate_critic = 1e-3

names = ["RBotGhostBloc", "RBotDashDash"]

ranges = {
    "RBotGhostBloc": 0.04 / math.sqrt(2),
    "RBotDashDash": .16 / math.sqrt(2),
}
