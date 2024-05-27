# import rtde_control 
# import rtde_receive
import random
import time
import csv
import numpy as np
import asyncio
import jax
import optax
import flax
from algorithms.offline.rebrac_Fetch_UR5 import DetActor, ActorTrainState, ReplayBuffer, Config
import json
import argparse

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

# Configuration
ROBOT_IP = "192.168.86.7"  # Change to your UR robot's IP address
CSV_FILE_PATH = "ur5_data_async.csv"
CONTROL_FREQUENCY = 250  # Control frequency in Hz
DT = 1 / CONTROL_FREQUENCY  # Sleep duration to match control frequency

# Connect to the robot
# rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
# rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP, CONTROL_FREQUENCY)

# Function to generate a random target within a safe workspace
def generate_random_target():
    # These limits are very conservative to ensure safety. Adjust them according to your setup.
    x = random.uniform(-0.1, 0.3)  # X-axis limit
    y = random.uniform(-0.3, -0.55)  # Y-axis limit
    z = random.uniform(0.35, 0.58)   # Z-axis limit
    rx, ry, rz = 0, -3.14, 0  # Fixed orientation for simplicity
    return [x, y, z, rx, ry, rz]

def get_action(obs):

    config_path = '/home/nikisim/Mag_research/UR5_FetchReach_Real/CORL_for_Unitree/data/saved_models/sim_as_real/FetchReach_UR5_ReBRAC_ac1000_bc_1000/config.json'
    model_path = '/home/nikisim/Mag_research/UR5_FetchReach_Real/CORL_for_Unitree/data/saved_models/sim_as_real/FetchReach_UR5_ReBRAC_ac1000_bc_1000/actor_state4200.pkl'

    with open(config_path) as json_file:
        config_dict = json.load(json_file)

    config = Config.from_dict(Config, config_dict)

    dataset_name = '/home/nikisim/Mag_research/UR5_FetchReach_Real/CORL_for_Unitree/data/UR5_FetchReach_real_small.npy'

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

def main():

    n = 5
    start_time = time.time()
    env = gym.make('gym_UR5_FetchReach/UR5_FetchReachEnv-v0', render=False)
    # env = gym.make('FetchPickAndPlaceDense-v2', render_mode='human')

    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    env = wrap_env(env, buffer.mean, buffer.std)
    for i in range(n):  # Collect data for 10 movements
        print(f"Iteration {i+1} out of {n}")
        target = generate_random_target()
        # target = [0.106, -0.357, 0.550, -0.0001, -3.14, 0.04] # home position
        ee_pose = rtde_r.getActualTCPPose()
        while np.linalg.norm(target - ee_pose) > 0.05:
            ee_pose = rtde_r.getActualTCPPose()
            ee_vel = rtde_r.getActualTCPSpeed()
            obs = np.concatenate((ee_pose, ee_vel, target, ee_pose))

            action = get_action(obs)
            # Check if any component of action is >= 0.15
            if np.any(action[:3] >= 0.15):
                print("Action component >= 0.15, stopping program.")
                return  # Exit the program

            # Check if the last component of ee_pose + action < 0.3
            if (ee_pose[-1] + action[-1]) < 0.3:
                print("Danger, stopping program.")
                return  # Exit the program

            to_point = ee_pose + action
            print(f"Moving to point: {to_point}")
            input("Press Enter to continue...")  # Wait for user to press Enter

            rtde_c.moveL(to_point, 0.5, 0.3, asynchronous=True)  # Start moving to the target asynchronously
            # collect_data(target,start_time)
            # print(f"Data collected for target: {target}") 


print("Data collection complete.")


# Main loop to move the robot and collect data
if __name__ == '__main__':
    main()