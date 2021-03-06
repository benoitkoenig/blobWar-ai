import ast
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from constants import ACTION_SIZE, names
from tracking import get_dataframes

pd.plotting.register_matplotlib_converters()

bot_name = None
for instruction in sys.argv:
    details = instruction.split("=")
    if (len(details) == 2):
        if (details[0] == "--bot"):
            if details[1] not in names:
                print("Invalid bot name. The name must be one of: {}".format(names))
                sys.exit()
            else:
                print("Bot selected: {}".format(details[1]))
                bot_name = details[1]

###############################
# Methods for data formatting #
###############################

def order_dates_by_category(df, key, categories):
    # Return an array containing, for each category, a Series of datetimes
    output = []
    for category in categories:
        output.append(df[df[key] == category]["datetime"].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')))
    return output

def get_color_density(df):
    # Return three arrays: action_id as x, reward as y, and the log of the occurences to be used by a colormap as c
    df2 = df[["action_id", "reward", "datetime"]].groupby(["action_id", "reward"]).count()
    df2_indexes = df2.index.to_list()
    x = [d[0] for d in df2_indexes]
    y = [d[1] for d in df2_indexes]
    c = [np.log(d) for d in df2["datetime"].to_list()]
    return x, y, c

def get_highest_prob_per_kill_data(df):
    # Returns highest probs for steps classified into Proper Kill, Kamikaze, or No Kill
    values = [
        df[(df["kamikaze"] == False) & (df["proper_kill"] == False)]["highest_prob"].values,
        df[df["kamikaze"] == True]["highest_prob"].values,
        df[df["proper_kill"] == True]["highest_prob"].values,
    ]
    for i in range(3):
        if len(values[i]) == 0:
            values[i] = [0.] # An empty array throws an error. If the highest prob appears to be null, we know for sure this is what happened
    return values

#########################################
# Initializing dataframes and variables #
#########################################

df_per_step, df_per_episode = get_dataframes()
df_per_step["highest_prob"] = df_per_step["probs"].map(lambda x: np.max(ast.literal_eval(x)))

first_date = datetime.strptime(df_per_episode["datetime"].loc[0], '%Y-%m-%d %H:%M:%S.%f')
last_date = datetime.strptime(df_per_episode["datetime"][df_per_episode.shape[0]-1], '%Y-%m-%d %H:%M:%S.%f')

if bot_name != None:
    df_per_step = df_per_step[df_per_step["name"] == bot_name]
    df_per_episode = df_per_episode[df_per_episode["name"] == bot_name]

df_per_step_tail = df_per_step.tail(100000)
df_per_step_pre_tail = df_per_step.tail(200000).head(100000)

############
# Plotting #
############

plt.figure(figsize=(18, 12))

# Number of steps per game
plt.subplot(3, 3, 1)
nb_steps = df_per_episode["nb_steps"].values
plt.plot(nb_steps, color="orange")
plt.title("Nb Steps Per Game")

# Outcome of the game per time
plt.subplot(3, 3, 2)
labels = ["Timeout", "Defeat", "It's a draw", "Victory !"]
ordered_results = order_dates_by_category(df_per_episode, "result", labels)
plt.hist(ordered_results, bins=24, stacked=True, color=["yellow", "red", "blue", "green"], label=labels)
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
plt.xlim(first_date, last_date)
plt.title("Result Per Time")

# Number of proper kills per time
plt.subplot(3, 3, 3)
ordered_proper_kills = order_dates_by_category(df_per_episode, "nb_proper_kills", [3, 2, 1])
plt.hist(ordered_proper_kills, bins=24, stacked=True, color=["red", "blue", "green"], label=[3, 2, 1])
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
plt.xlim(first_date, last_date)
plt.title("Nb Proper Kills Per Time")

# Scatter plot of rewards obtaining by action id (pretail)
plt.subplot(3, 3, 4)
x, y, c = get_color_density(df_per_step_pre_tail)
plt.scatter(x, y, c=c, cmap="viridis")
plt.ylim(-22, 45)
plt.xticks(range(ACTION_SIZE))
plt.title("Log Count Reward Per Action Id - Pre-Tail")

# Scatter plot of rewards obtaining by action id (tailed)
plt.subplot(3, 3, 5)
x, y, c = get_color_density(df_per_step_tail)
plt.scatter(x, y, c=c, cmap="viridis")
plt.ylim(-22, 45)
plt.xticks(range(ACTION_SIZE))
plt.title("Log Count Reward Per Action Id - Tail")

# Number of kamikaze per time
plt.subplot(3, 3, 6)
ordered_kamikaze = order_dates_by_category(df_per_episode, "nb_kamikaze", [3, 2, 1])
plt.hist(ordered_kamikaze, bins=24, stacked=True, color=["red", "blue", "green"], label=[3, 2, 1])
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
plt.xlim(first_date, last_date)
plt.legend()
plt.title("Nb Kamikaze Per Time")

# Density of highest probability per step ordered by kill status (pretail)
plt.subplot(3, 3, 7)
highest_prob_per_kill_data = get_highest_prob_per_kill_data(df_per_step_pre_tail)
plt.violinplot(highest_prob_per_kill_data)
plt.xticks([1, 2, 3], ["No Kill", "Kamikaze", "Proper Kill"])
plt.ylim(0., 1.)
plt.title("Highest Prob Per Kill - Pre-Tail")

# Density of highest probability per step ordered by kill status (tail)
plt.subplot(3, 3, 8)
highest_prob_per_kill_data = get_highest_prob_per_kill_data(df_per_step_tail)
plt.violinplot(highest_prob_per_kill_data)
plt.xticks([1, 2, 3], ["No Kill", "Kamikaze", "Proper Kill"])
plt.ylim(0., 1.)
plt.title("Highest Prob Per Kill - Tail")

# Highest probability per step
plt.subplot(3, 3, 9)
highest_prob = df_per_step["highest_prob"].values
plt.plot(highest_prob)
plt.ylim(0., 1.)
plt.title("Highest Prob Per Step")

plt.show()
