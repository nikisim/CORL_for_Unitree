import jax
import optax
import flax
from algorithms.offline.rebrac_Fetch import DetActor, ActorTrainState, ReplayBuffer, Config
import gym
from algorithms.offline.rebrac_Fetch import wrap_env
import json
import metagym.quadrupedal
from typing import Any, Callable
import numpy as np
import argparse
from tqdm.auto import trange


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

def evaluate(
    env: gym.Env,
    params: jax.Array,
    action_fn: Callable,
    num_episodes: int,
    seed: int,
) -> np.ndarray:
    # env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    returns = []
    success = []
    for _ in trange(num_episodes, desc="Eval", leave=False):
        obs, _ = env.reset()
        done = False
        # print("Observation:", obs)
        # print("Observation_type:", type(obs))
        total_reward = 0.0
        while not done:
            action = np.asarray(jax.device_get(action_fn(params, obs)))
            obs, reward, done, info = env.step(action)
            # print("Observation:", obs)
            # print("Action:", action)
            # print("New Obs:", obs)
            # print("Reward:", reward)
            # done = termination or truncation
            total_reward += reward
        # success.append(info['is_success'])
        returns.append(total_reward)
    # print("---"*10)
    # print(f"{int(sum(success))} Suceess Episodes out of {len(success)}")
    # print("---"*10)
    # success_rate = sum(success)/len(success)

    return np.array(returns)#, success_rate



def main(env_name, num_episodes, config_path, model_path, seed):

    with open(config_path) as json_file:
        config_dict = json.load(json_file)

    config = Config.from_dict(Config, config_dict)

    dataset_name = f'data/dataset_unitree_ground2.npy'

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

    env = gym.make('quadrupedal-v0',render=1,task="ground")
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    # env = wrap_env(env, buffer.mean, buffer.std)

    eval_returns, eval_success = evaluate(
        env,
        actor.params,
        actor_action_fn,
        num_episodes,
        seed=seed,
    )
    env.close()

if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Evaluate a CORL pre-trained model.")
    parser.add_argument("--env_name", type=str, default='FetchReach',help="Name of the environment to run.")
    parser.add_argument("--config_path", type=str, default='data/saved_models/FetchPush/config.json', help="Path to the configuration JSON file.")
    parser.add_argument("--model_path", type=str, default='data/saved_models/FetchPush/actor_state90.pkl', help="Path to the saved model.")
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to run.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility.")

    # Parse the command-line arguments
    args = parser.parse_args()
    # Call the main function with the parsed arguments
    main(args.env_name, args.num_episodes, args.config_path, 
         args.model_path, args.seed)