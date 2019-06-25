import math

STATE_SIZE = 168
ACTION_SIZE = 55

step_size = 4
gamma = .9
epsilon = .1
prob_flattener_factor = .01
update_interval = 10
learning_rate_actor = 1e-5
learning_rate_critic = 1e-5

names = ["RBotGhostBloc", "RBotDashDash"]

ranges = {
    "RBotGhostBloc": 0.04 / math.sqrt(2),
    "RBotDashDash": .16 / math.sqrt(2),
}
