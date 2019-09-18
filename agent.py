import numpy as np
import tensorflow as tf

from actions import get_action
from bots import Bots, CriticModel, ActorModel
from constants import ACTION_SIZE, STATE_SIZE, step_size, gamma, epsilon, prob_flattener_factor
from reward import determine_reward, calc_kills
from state import get_state_vector
from tracking import save_episode_data, save_step_data

tf.enable_eager_execution()

bots = Bots()
bots.load()

class Agent:
    def __init__(self, sio, id, name):
        self.sio = sio
        self.id = id
        self.t = 0
        self.name = name
        self.step = 0
        self.nb_proper_kills = 0
        self.nb_kamikaze = 0
        self.old_state_vector = None
        self.old_action_id = None
        self.old_state = None # Used only to calculate the reward
        self.old_probs = None # Used only for tracking

        self.bot = bots[name]

        self.grads_critic = []
        self.grads_actor = []

        sio.emit("learning_agent_created-{}".format(id))

    def action(self, state):
        if (state["type"] == "gameStarted"):
            return # The bot does not need this signal
        if (state["type"] == "update"):
            self.t += 1
            if (self.t == 1) | (self.t % step_size != 0): # self.t < 2 because arbitrary kills may happen at t=1 on exploratory starts
                self.sio.emit("action-{}".format(self.id), [])
                return
            self.step = int((self.t - 2) / step_size)
        self.update(state)

    def get_loss_computers(self, state_vector, reward, isFinalState):
        def get_value_loss():
            value = self.bot.critic(self.old_state_vector)[0][0]
            returnValue = reward
            if (isFinalState == False):
                newValue = self.bot.critic(state_vector)[0][0]
                returnValue += gamma * newValue
            self.advantage = returnValue - value # save it for policy_loss
            return self.advantage ** 2

        def get_policy_loss():
            logits = self.bot.actor(self.old_state_vector)
            s = tf.reduce_sum(tf.math.exp(logits))
            logits_with_epsilon = tf.math.log((1 - epsilon) * tf.math.exp(logits) + s * epsilon / ACTION_SIZE)

            policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=[self.old_action_id], logits=logits_with_epsilon)[0] # The loss assuming we took the right action
            policy_loss = tf.stop_gradient(self.advantage) * policy_loss # Multiply by advantage tells how much that action was right

            probs = tf.nn.softmax(logits[0])
            log_p = tf.math.abs(tf.math.log(probs * ACTION_SIZE))
            low_prob_advantage = tf.reduce_sum(log_p)
            # The goal of low_prob_advantage is to give a very tiny advantage to actions with low probability,
            # so that actions with the same consequences tend to the same probability
            # I have seen cross-entropy being used a similar way, but with my own solution I see better what happens

            print(policy_loss + prob_flattener_factor * low_prob_advantage)
            return policy_loss + prob_flattener_factor * low_prob_advantage

        return get_value_loss, get_policy_loss

    def update(self, state):
        state_vector = get_state_vector(state, self.name)
        state_vector = tf.convert_to_tensor([state_vector])
        logits = self.bot.actor(state_vector)
        probs = tf.nn.softmax(logits[0]).numpy()
        probs = (1 - epsilon) * probs + epsilon / ACTION_SIZE
        best_action_id = np.random.choice(ACTION_SIZE, p=probs)

        reward = determine_reward(self.old_state, state, self.old_action_id)
        if (reward != None):
            allies_killed, enemies_killed = calc_kills(self.old_state, state)
            if (allies_killed == 0) & (enemies_killed != 0):
                self.nb_proper_kills += 1
            elif (enemies_killed != 0):
                self.nb_kamikaze += 1

            get_value_loss, get_policy_loss = self.get_loss_computers(state_vector, reward, (state["type"] == "endOfGame")) # calling them in the right order is important

            self.bot.optimizer_critic.minimize(get_value_loss)
            self.bot.optimizer_actor.minimize(get_policy_loss)

            save_step_data(self.id, self.name, self.step - 1, self.old_probs.tolist(), self.old_action_id, reward, (allies_killed == 0) & (enemies_killed != 0), (allies_killed != 0) & (enemies_killed != 0))
            if (state["type"] == "endOfGame"):
                save_episode_data(self.id, self.name, self.step, self.nb_proper_kills, self.nb_kamikaze, state["value"])
                self.bot.save()

        self.old_state_vector = state_vector
        self.old_action_id = best_action_id
        self.old_state = state
        self.old_probs = probs

        action = get_action(state, best_action_id)
        self.sio.emit("action-{}".format(self.id), action)
