import datetime
import pandas as pd

per_episode_columns = ["datetime", "agent_id", "name", "nb_steps", "nb_proper_kills", "nb_kamikaze", "result"]
per_episode_file_path = "data/per_episode.csv"
per_step_columns = ["datetime", "agent_id", "name", "step", "probs", "action_id", "reward", "proper_kill", "kamikaze"]
per_step_file_path = "data/per_step.csv"

def reset_data():
    df = pd.DataFrame({}, columns=per_episode_columns)
    df.to_csv(per_episode_file_path, header=True, index=False)
    df = pd.DataFrame({}, columns=per_step_columns)
    df.to_csv(per_step_file_path, header=True, index=False)

def save_episode_data(agent_id, name, nb_steps, nb_proper_kills, nb_kamikaze, result):
    df = pd.DataFrame({
        "datetime": [datetime.datetime.now()],
        "agent_id": [agent_id],
        "name": [name],
        "nb_steps": [nb_steps],
        "nb_proper_kills": [nb_proper_kills],
        "nb_kamikaze": [nb_kamikaze],
        "result": [result],
    }, columns=per_episode_columns)
    df.to_csv(per_episode_file_path, mode="a", header=False, index=False)

def save_step_data(agent_id, name, step, probs, action_id, reward, proper_kill, kamikaze):
    df = pd.DataFrame({
        "datetime": [datetime.datetime.now()],
        "agent_id": [agent_id],
        "name": [name],
        "step": [step],
        "probs": [probs],
        "action_id": [action_id],
        "reward": [reward],
        "proper_kill": [proper_kill],
        "kamikaze": [kamikaze],
    }, columns=per_step_columns)
    df.to_csv(per_step_file_path, mode="a", header=False, index=False)

def get_dataframes():
    return pd.read_csv(per_step_file_path), pd.read_csv(per_episode_file_path)
