import gym
import torch
import argparse
import os
import math
import optuna
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, EveryNTimesteps
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from gym.envs.registration import register

from UDR import UDRCallback

register(
    id='CustomHopper-source-v0',
    entry_point='env.custom_hopper:CustomHopper',
    kwargs={'domain': 'source'},
    max_episode_steps=1000,
)

register(
    id='CustomHopper-target-v0',
    entry_point='env.custom_hopper:CustomHopper',
    kwargs={'domain': 'target'},
    max_episode_steps=1000,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-timesteps", default=1000000, type=int, help="Number of training episodes")
    parser.add_argument("--outName", default=None, type=str, help="output filename")
    parser.add_argument("--model", default=None, type=str, help="file to load")
    parser.add_argument("--test-episodes", default=50, type=int, help="number of episodes to use for testing")
    parser.add_argument("--algorithm", default="PPO", type=str, help="[SAC, PPO]")
    parser.add_argument("--render", default=False, action="store_true", help="Render the simulation")
    parser.add_argument("--device", default="cpu", type=str, help="[cpu, cuda]")
    parser.add_argument("--DR", default=None, type=str, help="Type of domain randomization [UDR], default None")
    parser.add_argument("--range", default=5, type=float, help="Range for the UDR")
    parser.add_argument("--mass", default=None, type=int, help="Mass for the UDR")
    parser.add_argument("--optimize", default=False, action="store_true", help="Use Optuna for hyperparameter optimization")
    parser.add_argument('--n-trials', default=12, type=int, help='Number of Optuna trials')
    return parser.parse_args()

args = parse_args()
list_DR = ["UDR"]

def compute_names(alg, train_type, dr_type):
    ending_part_names = f"{train_type}_{dr_type}_ts_{args.train_timesteps}"
    if dr_type == "UDR":
        ending_part_names += f"_range_{args.range}"
        if args.mass in [1, 2, 3]:
            ending_part_names += f"_m{args.mass}"
    model_name = f"{alg}model_{ending_part_names}" if args.outName is None else f"{args.outName}_{ending_part_names}"
    evaluation_name = f"eval_{model_name}"
    monitor_name = f"monitor_{model_name}"
    return model_name, evaluation_name, monitor_name, ending_part_names

def load_model(alg, source_env, dr_type):
    model_name, _, _, ending_part = compute_names(alg, source_env, dr_type)
    if args.model not in [None, "best", "final"]:
        source_model_name = f"{args.model}_{ending_part}"
    elif args.model == "final":
        source_model_name = f"./bestmodel/final_model/{model_name}"
    else:
        source_model_name = f"./bestmodel/models/{model_name}"
    print(source_model_name)

    if alg == "SAC":
        source_model = SAC.load(source_model_name, device=args.device)
    else:
        source_model = PPO.load(source_model_name, device=args.device)

    return source_model

def save_rename_files(model, model_name, evaluation_name, monitor_name, log_dir):
    os.rename("./bestmodel/models/best_model.zip", f"./bestmodel/models/{model_name}.zip")
    os.rename("./bestmodel/logs/evaluations.npz", f"./bestmodel/logs/{evaluation_name}.npz")
    os.rename(f"{log_dir}monitor.csv", f"{log_dir}{monitor_name}.csv")
    model.save(f"./bestmodel/final_model/{model_name}")

def initialize_variables():
    alg = "SAC" if args.algorithm == "SAC" else "PPO"
    log_dir = f"./total_logs/{alg}Logs/"
    os.makedirs(log_dir, exist_ok=True)
    dr_type = args.DR if args.DR in list_DR else "no_DR"
    train_types = ["source"] if dr_type in list_DR else ["source", "target"]

    if dr_type in list_DR:
        source_target_tuples = [("source", "source"), ("source", "target")]
        titles = [
            f"\n----------- SOURCE TO SOURCE ({dr_type}) -----------",
            f"\n----------- SOURCE TO TARGET ({dr_type}) -----------",
        ]
    else:
        source_target_tuples = [("source", "source"), ("source", "target"), ("target", "target")]
        titles = [
            "\n----------- SOURCE TO SOURCE -----------",
            "\n----------- SOURCE TO TARGET (lower bound) -----------",
            "\n----------- TARGET TO TARGET (upper bound) -----------",
        ]

    return alg, log_dir, dr_type, train_types, source_target_tuples, titles

def compute_callback(dr_type, train_env):
    eval_callback = EvalCallback(
        train_env,
        log_path="./bestmodel/logs/",
        best_model_save_path="./bestmodel/models/",
        eval_freq=math.ceil(args.train_timesteps / 1000),
        deterministic=True,
        render=False,
    )
    if dr_type == "UDR":
        UDR = UDRCallback(env=train_env, range_value=args.range, verbose=1, mass=args.mass)
        callback_list = CallbackList([eval_callback, UDR])
        callback = callback_list
    else:
        callback = eval_callback
    return callback

def test(source_model, target_env, deterministic=False):
    total_reward = torch.zeros((args.test_episodes,))

    for episode in range(args.test_episodes):
        done = False
        test_reward = 0
        state = target_env.reset()

        while not done:
            action, _ = source_model.predict(state, deterministic=deterministic)
            state, reward, done, info = target_env.step(action)

            if args.render:
                target_env.render()

            test_reward += reward
        total_reward[episode] = torch.tensor(test_reward, dtype=torch.float32)
    std, mean = torch.std_mean(total_reward)
    print("Average reward: ", mean.item())
    print("Std :", std.item())
    return mean.item(), std.item()

def optimize_hyperparameters(trial):
    alg, log_dir, dr_type, train_types, source_target_tuples, titles = initialize_variables()

    # Define the search space for hyperparameters
    if args.algorithm == "PPO":
        hyperparameters = {
            "gamma": trial.suggest_float("gamma", 0.9, 0.9999, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True),
            #"ent_coef": trial.suggest_float("ent_coef", 0.0, 0.1),
            "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
        }
    elif args.algorithm == "SAC":
        hyperparameters = {
            "gamma": trial.suggest_float("gamma", 0.9, 0.9999, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.1),
            "tau": trial.suggest_float("tau", 0.005, 0.02),
        }
    
    # Make environment
    def make_env():
        train_env = gym.make(f"CustomHopper-source-v0")
        train_env = Monitor(train_env, log_dir)
        return train_env
    # Create vectorized environment
    train_env = DummyVecEnv([make_env])
    # Tensorboard logging
    trial_log_dir = f"./total_logs/{args.algorithm}Logs/trial_{trial.number}"
    new_logger = configure(trial_log_dir, ["tensorboard"])
    # Create the model
    if args.algorithm == "PPO":
        model = PPO("MlpPolicy", train_env, device=args.device, **hyperparameters)
    elif args.algorithm == "SAC":
        model = SAC("MlpPolicy", train_env, device=args.device, **hyperparameters)

    model.set_logger(new_logger)

    # Use EvalCallback to evaluate the model and select the best one
    callback = compute_callback(dr_type, train_env)
    model.learn(total_timesteps=args.train_timesteps, callback=callback)

    mean_reward, _ = test(model, train_env)
    return mean_reward

def main():
    if args.optimize:
        study_name = "PPO_tuning13"
        storage_dir = "optuna_trials"
        os.makedirs(storage_dir, exist_ok=True)
        storage_name = f"sqlite:///{storage_dir}/optuna_trials_opt.db"
        study = optuna.create_study(study_name=study_name, storage=storage_name, direction='maximize')
        study.optimize(optimize_hyperparameters,n_trials=args.n_trials)
        print("Best trial:")
        trial = study.best_trial
        print(trial.values)
        print("Best hyperparameters:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
    else:
        alg, log_dir, dr_type, train_types, source_target_tuples, titles = initialize_variables()

        if args.model is None:
            for train_type in train_types:
                print(f"\n----------- {train_type.capitalize()} environment -----------")

                train_env = gym.make(f"CustomHopper-{train_type}-v0")
                train_env = Monitor(train_env, log_dir)

                print("State space:", train_env.observation_space)
                print("Action space:", train_env.action_space)
                print("Dynamics parameters:", train_env.get_parameters())

                # Tensorboard Logging
                log_dir = f"./total_logs/{args.algorithm}Logs/"
                new_logger = configure(log_dir, ["tensorboard"])

                if alg == "SAC":
                    model = SAC("MlpPolicy", train_env, device=args.device)
                else:
                    model = PPO("MlpPolicy", train_env, device=args.device)

                model.set_logger(new_logger)

                callback = compute_callback(dr_type, train_env)
                model.learn(total_timesteps=args.train_timesteps, progress_bar=True, callback=callback)

                model_name, evaluation_name, monitor_name, _ = compute_names(alg, train_type, dr_type)
                save_rename_files(model, model_name, evaluation_name, monitor_name, log_dir)

        for title, (source_env, target_env) in zip(titles, source_target_tuples):
            source_model = load_model(alg, source_env, dr_type)
            print(title)
            target_env = gym.make(f"CustomHopper-{target_env}-v0")
            test(source_model, target_env, deterministic=False)

if __name__ == "__main__":
    main()
               

