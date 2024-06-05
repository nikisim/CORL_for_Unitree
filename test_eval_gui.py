import jax
import optax
import flax
from algorithms.offline.rebrac_Fetch_UR5 import DetActor, ActorTrainState, ReplayBuffer, Config
import gym
import gym_UR5_FetchReach
from algorithms.offline.rebrac_Fetch_UR5 import wrap_env, evaluate, compute_mean_std
import json
import argparse
import numpy as np
import tkinter as tk
from threading import Thread

# GUI class
class RobotGUI3:
    def __init__(self, master):
        self.master = master
        master.title("Robot Manipulator Status")

        # Define a larger, prettier font
        self.large_font = ('Helvetica', 14, 'bold')

        # Define colors
        self.header_bg = '#FFACAC'  # Gold for headers
        self.cell_bg = '#ADD8E6'  # Light blue for cells
        self.highlight_bg = '#32CD32'  # Lime green for highlighted error cells

        # Headers
        headers = ["ee_pos", "target_pos", "action", "error"]
        for i, header in enumerate(headers):
            tk.Label(master, text=header, font=self.large_font, bg=self.header_bg, borderwidth=2, relief="groove").grid(row=0, column=i, sticky="nsew", padx=1, pady=1)

        # Initialize rows list to keep track of data rows
        self.rows = []
        self.max_rows = 5  # Maximum number of rows to display

    def add_row(self, ee_pos, target_pos, error, action):
        if len(self.rows) >= self.max_rows:
            return  # Do not add more than max_rows

        row_index = len(self.rows) + 1  # +1 to account for the header row
        row_data = {
            "ee_pos_var": tk.StringVar(value=str(ee_pos)),
            "target_pos_var": tk.StringVar(value=str(target_pos)),
            "error_var": tk.StringVar(value=f"{error:.4f}"),
            "action_var": tk.StringVar(value=str(action)),
            "error_label": None  # To update background color later
        }

        # Create labels for the new row with specified background and font
        for i, key in enumerate(["ee_pos_var", "target_pos_var", "action_var", "error_var"]):
            label = tk.Label(self.master, textvariable=row_data[key], font=self.large_font, bg=self.cell_bg, borderwidth=2, relief="groove")
            label.grid(row=row_index, column=i, sticky="nsew", padx=1, pady=1)
            if key == "error_var":
                row_data["error_label"] = label

        self.rows.append(row_data)

        # Update the background color if needed
        self.update_error_background(row_data, error)

    def update_error_background(self, row_data, error):
        if error < 0.05:
            row_data["error_label"].config(background=self.highlight_bg)
        else:
            row_data["error_label"].config(background=self.cell_bg)

    def update_values(self, ee_pos, target_pos, error, action):
        # If the last row's error is green or there are no rows, add a new row
        if not self.rows or self.rows[-1]["error_label"].cget("background") == self.highlight_bg:
            self.add_row(ee_pos, target_pos, error, action)
        else:
            # Update the last row
            last_row = self.rows[-1]
            last_row["ee_pos_var"].set(str(ee_pos))
            last_row["target_pos_var"].set(str(target_pos))
            last_row["error_var"].set(f"{error:.4f}")
            last_row["action_var"].set(str(action))
            self.update_error_background(last_row, error)

        self.master.update()

# Load the TrainState from a file
def load_train_state(save_path, state_structure):
    with open(save_path, 'rb') as f:
        state_dict = flax.serialization.from_bytes(state_structure, f.read())
    return state_dict

def create_train_state(actor_module, actor_key, init_state, actor_learning_rate):
    return ActorTrainState.create(
        apply_fn=actor_module.apply,
        params=actor_module.init(actor_key, init_state),
        target_params=actor_module.init(actor_key, init_state),
        tx=optax.adam(learning_rate=actor_learning_rate),
    )

def get_action(obs):

    config_path = '/home/nikisim/Mag_diplom/CORL/data/saved_models/sim_as_real/FetchReach_UR5_ReBRAC_ac1300_bc_1100_ReBRAC_1300.0_1100.0_Reach--d8e5f605/config.json'
    model_path = '/home/nikisim/Mag_diplom/CORL/data/saved_models/sim_as_real/FetchReach_UR5_ReBRAC_ac1300_bc_1100_ReBRAC_1300.0_1100.0_Reach--d8e5f605/actor_state4200.pkl'

    with open(config_path) as json_file:
        config_dict = json.load(json_file)

    config = Config.from_dict(Config, config_dict)

    dataset_name = '/home/nikisim/Mag_diplom/UR5_FetchReach_real/datasets/UR5_FetchReach_real_small.npy'

    buffer = ReplayBuffer()
    buffer.create_from_d4rl(
        dataset_name, False, False
    )
    @jax.jit
    def actor_action_fn(params: jax.Array, obs: jax.Array):
        return actor.apply_fn(params, obs)

    init_state = buffer.data["states"][0][None, ...]
    init_action = buffer.data["actions"][0][None, ...]

    key = jax.random.PRNGKey(seed=51)
    key, actor_key, _ = jax.random.split(key, 3)

    actor_module = DetActor(
            action_dim=init_action.shape[-1],
            hidden_dim=config.hidden_dim,
            layernorm=config.actor_ln,
            n_hiddens=config.actor_n_hiddens,
        )

    train_state_struc = create_train_state(actor_module, actor_key, init_state, config.actor_learning_rate)

    actor = load_train_state(model_path, train_state_struc)

    action = np.asarray(jax.device_get(actor_action_fn(actor.params, obs)))

    return action

def main(gui):

    env = gym.make('gym_UR5_FetchReach/UR5_FetchReachEnv-v0', render=True)
    # env = gym.make('FetchPickAndPlaceDense-v2', render_mode='human')

    dataset_name = '/home/nikisim/Mag_diplom/UR5_FetchReach_real/datasets/UR5_FetchReach_real_small.npy'

    buffer = ReplayBuffer()
    buffer.create_from_d4rl(
        dataset_name, False, False
    )

    env.action_space.seed(42)
    env.observation_space.seed(42)
    env = wrap_env(env, buffer.mean, buffer.std)

    for _ in range(5):
        obs, _ = env.reset()
        done = False
        # print("Observation:", obs)
        # print("Observation_type:", type(obs))
        total_reward = 0.0
        while not done:
            action = get_action(obs)
            obs, reward, termination, truncation, info = env.step(action)
            print("EE_POS", obs[:3])
            print("TARGET_POS", obs[7:10])
            error = -reward
            print("Error", -reward)

            gui.update_values(np.round(obs[:3],3), np.round(obs[7:10],3), error, np.round(action,3))
            done = termination or truncation

            total_reward += reward
    env.close()

def run_robot_control(gui):
    thread = Thread(target=main, args=(gui,))
    thread.start()

if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Evaluate a CORL pre-trained model.")
    parser.add_argument("--env_name", type=str, default='FetchReach',help="Name of the environment to run.")
    parser.add_argument("--config_path", type=str, default='/home/nikisim/Mag_diplom/CORL/data/saved_models/sim_as_real/FetchReach_UR5_ReBRAC_ac1300_bc_1100_ReBRAC_1300.0_1100.0_Reach--d8e5f605/config.json', help="Path to the configuration JSON file.")
    parser.add_argument("--model_path", type=str, default='/home/nikisim/Mag_diplom/CORL/data/saved_models/sim_as_real/FetchReach_UR5_ReBRAC_ac1300_bc_1100_ReBRAC_1300.0_1100.0_Reach--d8e5f605/actor_state4200.pkl', help="Path to the saved model.")
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to run.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility.")

    # Parse the command-line arguments
    args = parser.parse_args()
    # Call the main function with the parsed arguments

    root = tk.Tk()
    gui = RobotGUI3(root)
    run_robot_control(gui)
    root.mainloop()