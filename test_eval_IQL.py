import jax
import optax
import flax
from algorithms.offline.rebrac_Fetch_UR5 import DetActor, ActorTrainState, ReplayBuffer, Config
import gym
import gym_UR5_FetchReach
import torch
from pathlib import Path
from algorithms.offline.rebrac_Fetch_UR5 import wrap_env, evaluate
import yaml
import numpy as np
import argparse
from algorithms.offline.iql_Fetch_UR5 import ImplicitQLearning, eval_actor, compute_mean_std



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


def main(env_name, num_episodes, config_path, model_path, seed):

    trainer = ImplicitQLearning(**kwargs)

    
    policy_file = Path(model_path)
    trainer.load_state_dict(torch.load(policy_file))
    actor = trainer.actor
    

    with open(config_path) as yaml_file:
        config_dict = yaml.load(yaml_file)

    config = Config.from_dict(Config, config_dict)

    dataset_name = '/home/nikisim/Mag_diplom/CORL/data/UR5_FetchReach_new_action2.npy'

    dataset = np.load(dataset_name, allow_pickle=True).item()
    replay_buffer = ReplayBuffer()
    replay_buffer.create_from_d4rl(
        dataset_name, False, False
    )

    env = gym.make('gym_UR5_FetchReach/UR5_FetchReachEnv-v0', render=True)

    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    env = wrap_env(env, replay_buffer.mean, replay_buffer.std)

    evaluations = []
    eval_scores, eval_success = eval_actor(
        env,
        actor,
        device=config.device,
        n_episodes=num_episodes,
        seed=config.seed,
    )
    env.close()

if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Evaluate a CORL pre-trained model.")
    parser.add_argument("--env_name", type=str, default='FetchReach',help="Name of the environment to run.")
    parser.add_argument("--config_path", type=str, default='/home/nikisim/Mag_diplom/CORL/data/saved_models/IQL_UR5_FetchReach_new_action/IQL-FetchReach_UR5-221b042a/config.yaml', help="Path to the configuration YAML file.")
    parser.add_argument("--model_path", type=str, default='/home/nikisim/Mag_diplom/CORL/data/saved_models/IQL_UR5_FetchReach_new_action/IQL-FetchReach_UR5-221b042a/checkpoint_429999.pt', help="Path to the saved model.")
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to run.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility.")

    # Parse the command-line arguments
    args = parser.parse_args()
    # Call the main function with the parsed arguments
    main(args.env_name, args.num_episodes, args.config_path, 
         args.model_path, args.seed)