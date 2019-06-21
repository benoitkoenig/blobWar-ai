import ast
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tracking import get_dataframes

def order_dates_by_category(df, key, categories):
    output = []
    for _ in range(len(categories)):
        output.append([])
    categories = np.array(categories)

    def order(x):
        indexes = np.where(categories == x[key])[0]
        if len(indexes) != 0:
            dt = datetime.strptime(x["datetime"], '%Y-%m-%d %H:%M:%S.%f')
            output[indexes[0]].append(dt)

    df.apply(order, axis=1)
    return output

df_per_step, df_per_episode = get_dataframes()
df_per_step["highest_prob"] = df_per_step["probs"].map(lambda x: np.max(ast.literal_eval(x)))

first_date = datetime.strptime(df_per_episode["datetime"][0], '%Y-%m-%d %H:%M:%S.%f')
last_date = datetime.strptime(df_per_episode["datetime"][df_per_episode.shape[0]-1], '%Y-%m-%d %H:%M:%S.%f')

plt.figure(figsize=(16, 8))

nb_steps = df_per_episode["nb_steps"].values
plt.subplot(2, 4, 1)
plt.plot(nb_steps, color="orange")
plt.title("Nb Steps Per Game")

ordered_results = order_dates_by_category(df_per_episode, "result", ["Timeout", "Defeat", "It's a draw", "Victory !"])
plt.subplot(2, 4, 2)
plt.hist(
    ordered_results,
    bins=24,
    stacked=True,
    color=["purple", "red", "blue", "green"],
    label=["Timeout", "Defeat", "It's a draw", "Victory !"],
)
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
plt.xlim(first_date, last_date)
plt.title("Result Per Time Unit")
plt.subplot(2, 4, 3)

count = {}
total = 10000. # float(df_per_step.shape[0])
def set_color_density(x):
    if (x["action_id"] not in count):
        count[x["action_id"]] = {}
    if (x["reward"] not in count[x["action_id"]]):
        count[x["action_id"]][x["reward"]] = 1 - df_per_step[df_per_step["action_id"] == x["action_id"]][df_per_step["reward"] == x["reward"]].shape[0] / total
    return count[x["action_id"]][x["reward"]]

colors = df_per_step.tail(10000).apply(set_color_density, axis=1)
plt.scatter(df_per_step.tail(10000)["action_id"], df_per_step.tail(10000)["reward"], c=colors, cmap="viridis")
plt.title("Reward Per Action Id - tail(10000)")

ordered_proper_kills = order_dates_by_category(df_per_episode, "nb_proper_kills", [3, 2, 1])
plt.subplot(2, 4, 4)
plt.hist(
    ordered_proper_kills,
    bins=24,
    stacked=True,
    color=["red", "blue", "green"],
    label=[3, 2, 1],
)
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
plt.xlim(first_date, last_date)
plt.title("Nb Proper Kills Per Time Step")

highest_prob = df_per_step["highest_prob"].values
plt.subplot(2, 4, 5)
plt.plot(highest_prob)
plt.ylim(0., 1.)
plt.title("Highest Prob Per Step")

highest_prob_per_kill_data = [[], [], []]
def fill_highest_prob_per_kill_data(x):
    if x["proper_kill"]:
        highest_prob_per_kill_data[2].append(x["highest_prob"])
    elif x["kamikaze"]:
        highest_prob_per_kill_data[1].append(x["highest_prob"])
    highest_prob_per_kill_data[0].append(x["highest_prob"])
df_per_step.tail(10000).apply(fill_highest_prob_per_kill_data, axis=1)
plt.subplot(2, 4, 6)
plt.violinplot(highest_prob_per_kill_data)
plt.xticks([1, 2, 3], ["No Kill", "Kamikaze", "Proper Kill"])
plt.ylim(0., 1.)
plt.title("Highest Prob Per Kill - tail(10000)")

highest_prob_per_kill_data = [[], [], []]
df_per_step.apply(fill_highest_prob_per_kill_data, axis=1)
plt.subplot(2, 4, 7)
plt.violinplot(highest_prob_per_kill_data)
plt.xticks([1, 2, 3], ["No Kill", "Kamikaze", "Proper Kill"])
plt.ylim(0., 1.)
plt.title("Highest Prob Per Kill overall")

ordered_kamikaze = order_dates_by_category(df_per_episode, "nb_kamikaze", [3, 2, 1])
plt.subplot(2, 4, 8)
plt.hist(
    ordered_kamikaze,
    bins=24,
    stacked=True,
    color=["red", "blue", "green"],
    label=[3, 2, 1],
)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
plt.xlim(first_date, last_date)
plt.legend()
plt.title("Nb Kamikaze Per Time Step")

plt.show()
