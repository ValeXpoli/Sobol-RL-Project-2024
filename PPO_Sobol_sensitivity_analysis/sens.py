import gym
import math
from gym.envs.registration import register, registry
import argparse
import copy
from typing import Dict, List, Tuple, Any
import numpy as np
from SALib.sample import saltelli, latin
from SALib.analyze import sobol
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from joblib import Parallel, delayed
import logging
from tqdm import tqdm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from env import custom_hopper
import time
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Register the custom environment globally
def register_custom_envs():
    register(
        id='CustomHopper-source-v0',
        entry_point='env.custom_hopper:CustomHopper',
        kwargs={'domain': 'source'},
        max_episode_steps=500,
    )

register_custom_envs()  # Register once at the top

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-timesteps", default=None, type=int, help="Number of training episodes")
    parser.add_argument("--outName", default=None, type=str, help="Output filename")
    parser.add_argument("--model", default=None, type=str, help="File to load")
    parser.add_argument("--test-episodes", default=1, type=int, help="Number of episodes to use for testing")
    parser.add_argument("--algorithm", default="PPO", type=str, help="[SAC, PPO]")
    parser.add_argument("--DR", default="no_DR", type=str, help="Type of domain randomization [UDR], default None")
    parser.add_argument("--render", default=False, action="store_true", help="Render the simulation")
    parser.add_argument("--device", default="cpu", type=str, help="[cpu, cuda]")
    parser.add_argument("--range", default=5, type=float, help="Range for the UDR")
    parser.add_argument("--parallel", default=8, type=int, help="Number of parallel processes")
    parser.add_argument("--n", default=1000, type=int, help="Number of samples")
    parser.add_argument("--mode", default="sobol", type=str, help="[sobol, linear]")
    parser.add_argument("--mass", default=None, type=int, help="[1,2,3]")
    return parser.parse_args()

args = parse_args()

def load_model_in_process(alg: str, model_name: str, device: str) -> Any:
    try:
        model = SAC.load(model_name, device=device) if alg == "SAC" else PPO.load(model_name, device=device)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    return model

def test_model(sample_batch: np.ndarray, baseline: float, env_name: str, model_name: str, alg: str, device: str, test_episodes: int, batch_index: int) -> np.ndarray:
    register(
            id='CustomHopper-source-v0',
            entry_point='env.custom_hopper:CustomHopper',
            kwargs={'domain': 'source'},
            max_episode_steps=500,
        )
    env = gym.make(env_name)
    env = Monitor(env)
    
    # Load the model
    model = load_model_in_process(alg, model_name, device)
    
    # Store rewards for each sample in the batch
    rewards = []
    
    for i, sample in enumerate(sample_batch):
        # Assign the sample to the environment's body_mass
        env.sim.model.body_mass[2:] = sample
        
        # Evaluate the policy and store the mean reward
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=test_episodes, deterministic=True)
        rewards.append(mean_reward)
        
        # Log after processing each sample
        logger.info(f"Processed sample {i+1}/{len(sample_batch)} in batch {batch_index+1}")
    
    return np.array(rewards)

def parallel_evaluate(samples: np.ndarray, baseline: float, env_name: str, model_name: str, alg: str, device: str, test_episodes: int, n_parallel: int) -> np.ndarray:
    # Divide samples into batches
    batch_size = len(samples) // n_parallel
    batches = [samples[i:i + batch_size] for i in range(0, len(samples), batch_size)]
    
    results = Parallel(n_jobs=n_parallel, backend='loky')(delayed(test_model)(
        batch, baseline, env_name, model_name, alg, device, test_episodes, batch_index
    ) for batch_index, batch in enumerate(tqdm(batches, desc="Processing batches in parallel")))
    
    # Combine all results into a single array
    return np.concatenate(results)

def compute_hopper_mass_bounds() -> Tuple[List[float], List[float]]:
    train_env = gym.make("CustomHopper-source-v0")
    masses = copy.deepcopy(train_env.get_parameters())
    minimum = [max(mass - 3, 0.01) for mass in masses[1:]]
    maximum = [mass + 3 for mass in masses[1:]]
    return minimum, maximum

def compute_names(alg: str, train_type: str, dr_type: str) -> str:
    ending_part_names = f"{train_type}_{dr_type}_ts_{args.train_timesteps}"
    if dr_type == "UDR":
        ending_part_names += f"_range_{args.range}"
        if args.mass in [1, 2, 3]:
            ending_part_names += f"_m{args.mass}"
    model_name = f"{alg}model_{ending_part_names}" if args.outName is None else f"{args.outName}_{ending_part_names}"
    if args.model == "final":
        return f"./bestmodel/final_model/{model_name}"
    else:
        return f"./bestmodel/models/{model_name}"

def loop_sobol(n: int, n_parallel: int) -> Dict[str, Any]:
    if n < 0 or n_parallel < 0:
        raise ValueError("The number of samples and parallel processes must be positive.")

    model_name = compute_names(args.algorithm, "source", args.DR)
    env_name = "CustomHopper-source-v0"
    minimum, maximum = compute_hopper_mass_bounds()
    problem = {
        "num_vars": 3,
        "names": ["mass_1", "mass_2", "mass_3"],
        "bounds": [[minimum[0], maximum[0]], [minimum[1], maximum[1]], [minimum[2], maximum[2]]],
    }
    X = saltelli.sample(problem, N=n, calc_second_order=True)
    Y = parallel_evaluate(X, baseline=0, env_name=env_name, model_name=model_name, alg=args.algorithm, device=args.device, test_episodes=args.test_episodes, n_parallel=n_parallel)
    sobol_result = sobol.analyze(problem, Y, print_to_console=True, calc_second_order=True)
    return sobol_result

def linear(n: int, n_parallel: int):
    model_name = compute_names(args.algorithm, "source", args.DR)
    env_name = "CustomHopper-source-v0"
    minimum, maximum = compute_hopper_mass_bounds()
    problem = {
        "num_vars": 3,
        "names": ["mass_1", "mass_2", "mass_3"],
        "bounds": [[minimum[0], maximum[0]], [minimum[1], maximum[1]], [minimum[2], maximum[2]]],
    }
    
    # Latin Hypercube Sampling for linear regression
    X = latin.sample(problem, n)
    
    # Parallel evaluation
    Y = parallel_evaluate(X, baseline=0, env_name=env_name, model_name=model_name, alg=args.algorithm, device=args.device, test_episodes=args.test_episodes, n_parallel=n_parallel)
    
    delta = sm.add_constant(np.abs(X - np.array(gym.make(env_name).get_parameters()[1:])))
    
    plt.scatter(X[:, 0], Y, s=0.8)
    plt.ylabel("Reward")
    plt.xlabel("Mass 1")
    
    XY_sorted = list(zip(X[:, 0], Y))
    XY_sorted.sort(key=lambda x: x[0])
    XY_sorted = np.array(XY_sorted)
    centroids = np.zeros((5, 2))
    index = 0
    for i in range(5):
        step = math.ceil(n / 3)
        if index + step < n:
            X_centroid = np.mean(XY_sorted[index : index + step, 0])
            Y_centroid = np.mean(XY_sorted[index : index + step, 1])
            index = index + step
        else:
            X_centroid = np.mean(XY_sorted[index:, 0])
            Y_centroid = np.mean(XY_sorted[index:, 1])
        centroids[i, 0] = X_centroid
        centroids[i, 1] = Y_centroid
    
    print(centroids)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", c="red", label="Centroids of the left, middle and upper portion")
    plt.axvline(gym.make(env_name).get_parameters()[1], alpha=0.5, ls="--", c="green", linewidth=1.5, label="Original value")
    plt.legend()
    plt.title("Reward with respect to the first mass")
    plt.savefig("./plots/m1_vs_reward.png")
    plt.show()

    model = sm.OLS(Y, delta)
    res = model.fit_regularized()
    results = model.fit(params=res.params)

    print(results.params)

    plt.scatter(delta[:, 1], Y, s=0.8)
    plt.axline((0, results.params[0]), slope=results.params[1], linewidth=1.5, ls="--", c="red", label="Regression line")
    plt.legend()
    plt.ylabel("Reward")
    plt.xlabel("Distance from original value (about 3.93)")
    plt.title("Rewards with respect to the distance from original value of m1")
    plt.savefig("./plots/delta_m1_vs_reward.png")
    plt.show()

    df = pd.DataFrame(data=X, columns=["mass1", "mass2", "mass3"])
    df.insert(value=Y, column="Y", loc=3)
    print(df)
    df.to_csv("./logsens/linear.csv", index=False)
    print(results.summary())

def main() -> Any:
    start_time = time.time()
    try:
        if args.mode == "sobol":
            result = loop_sobol(args.n, args.parallel)
        elif args.mode == "linear":
            linear(args.n, args.parallel)
        else:
            raise ValueError(f"Invalid mode: {args.mode}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise
    finally:
        logger.info(f"Execution time: {time.time() - start_time:.2f} seconds")
    return result

if __name__ == "__main__":
    main()


