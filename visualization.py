import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tracking import get_dataframes

# per_episode_columns = ["datetime", "agent_id", "nb_steps", "nb_proper_kills", "nb_kamikaze", "result"]
# per_step_columns = ["datetime", "agent_id", "step", "probs", "action_id", "reward", "proper_kill", "kamikaze"]

df_per_step, df_per_episode = get_dataframes()
df_per_step["highest_prob"] = df_per_step["probs"].map(lambda x: np.max(ast.literal_eval(x)))

plt.figure(figsize=(16, 8))

# Per episode data

nb_steps = df_per_episode["nb_steps"].values
plt.subplot(2, 4, 1)
plt.plot(nb_steps)
plt.title("Nb Steps Per Game")

nb_proper_kills = df_per_episode["nb_proper_kills"].values
plt.subplot(2, 4, 2)
plt.plot(nb_proper_kills)
plt.title("Nb Proper Kills Per Game")

nb_kamikaze = df_per_episode["nb_kamikaze"].values
plt.subplot(2, 4, 3)
plt.plot(nb_kamikaze)
plt.title("Nb Kamikaze Per Game")

ordered_results = np.array(["Timeout", "Defeat", "Draw", "Victory !"])
result_value = df_per_episode["result"].map(lambda x: np.where(x == ordered_results)[0][0])
plt.subplot(2, 4, 4)
plt.plot(result_value)
plt.title("Result Per Game")

highest_prob = df_per_step["highest_prob"].values
plt.subplot(2, 4, 5)
plt.plot(highest_prob)
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
plt.title("Highest Prob Per Kill for 10000 last steps")

highest_prob_per_kill_data = [[], [], []]
df_per_step.apply(fill_highest_prob_per_kill_data, axis=1)
plt.subplot(2, 4, 7)
plt.violinplot(highest_prob_per_kill_data)
plt.xticks([1, 2, 3], ["No Kill", "Kamikaze", "Proper Kill"])
plt.title("Highest Prob Per Kill overall")

highest_prob = df_per_step["reward"].values
plt.subplot(2, 4, 8)
plt.plot(highest_prob)
plt.title("Reward Per Step")

plt.show()
