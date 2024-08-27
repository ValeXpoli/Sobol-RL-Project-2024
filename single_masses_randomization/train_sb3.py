import gym
import torch
import argparse
import os
import math
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, EveryNTimesteps
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from env.custom_hopper import *
from UDR import UDRCallback

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-timesteps", default=None, type=int, help="Number of training episodes")
    parser.add_argument("--outName", default=None, type=str, help="output filename")
    parser.add_argument("--model", default=None, type=str, help="file to load")
    parser.add_argument("--test-episodes", default=2000, type=int, help="number of episodes to use for testing")
    parser.add_argument("--algorithm", default="PPO", type=str, help="[SAC, PPO]")
    parser.add_argument("--render", default=False, action="store_true", help="Render the simulation")
    parser.add_argument("--device", default="cpu", type=str, help="[cpu, cuda]")
    parser.add_argument("--DR", default=None, type=str, help="Type of domain randomization [UDR], default None")
    parser.add_argument("--range", default=5, type=float, help="Range for the UDR")
    parser.add_argument("--mass", default=None, type=int, help="Mass for the UDR")
    return parser.parse_args()

args = parse_args()
list_DR = ["UDR"]

def compute_names(alg, train_type, dr_type):
    # ending part of names
    ending_part_names = f"{train_type}_{dr_type}_ts_{args.train_timesteps}"
    if dr_type == "UDR":
        ending_part_names = ending_part_names + f"_range_{args.range}"
        if args.mass in [1, 2, 3,12,13,23]:
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

    # Choose the correct loader depending on the algorithm
    if alg == "SAC":
        source_model = SAC.load(source_model_name, device=args.device)
    else:
        source_model = PPO.load(source_model_name, device=args.device)

    return source_model

def save_rename_files(model, model_name, evaluation_name, monitor_name, log_dir):
    # Rename existing files to avoid overwriting
    os.rename("./bestmodel/models/best_model.zip", f"./bestmodel/models/{model_name}.zip")
    os.rename("./bestmodel/logs/evaluations.npz", f"./bestmodel/logs/{evaluation_name}.npz")
    os.rename(f"{log_dir}monitor.csv", f"{log_dir}{monitor_name}.csv")

    # Save final model as well
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
    # Callback used to save the best model at each step
    eval_callback = EvalCallback(
        train_env,
        log_path="./bestmodel/logs/",
        best_model_save_path="./bestmodel/models/",
        eval_freq=math.ceil(args.train_timesteps / 1000),
        deterministic=True,
        render=False,
    )
    # Handling Domain Randomization via custom callbacks
    if dr_type == "UDR":
        UDR = UDRCallback(env=train_env, range_value=args.range, verbose=1, mass=args.mass)
        callback_list = CallbackList([eval_callback, UDR])
        callback = callback_list
    else:
        callback = eval_callback
    return callback

def test(source_model, target_env, deterministic=False):
    # Creation of the target environment and initialization of the array with the rewards
    total_reward = torch.zeros((args.test_episodes,))

    # Testing loop over the episodes
    for episode in range(args.test_episodes):
        done = False
        test_reward = 0
        state = target_env.reset()

        # For each episode keep taking actions until it reaches terminal state
        while not done:
            action, _ = source_model.predict(state, deterministic=deterministic)  # compute action
            state, reward, done, info = target_env.step(action)  # take action and observe next state, rewards etc.

            # Allow user to choose to render (during the testing only)
            if args.render:
                target_env.render()

            test_reward += reward
        total_reward[episode] = test_reward
    std, mean = torch.std_mean(total_reward)
    print("Average reward: ", mean.item())
    print("Std :", std.item())
    return mean.item(), std.item()

def main():
    # --------- INITIALIZATION OF VARIABLES AND FOLDERS USED IN THE CODE ---------
    alg, log_dir, dr_type, train_types, source_target_tuples, titles = initialize_variables()

    # Configure TensorBoard Logger
    tb_log_dir = f"./total_logs/{alg}_tensorboard/"
    new_logger = configure(tb_log_dir, ["tensorboard"])

    #  --------- TRAINING OF THE MODEL (ONLY IF MODEL IS NOT GIVEN BY THE USER) ---------
    if args.model is None:
        for train_type in train_types:
            print(f"\n----------- {train_type.capitalize()} environment -----------")

            # Creation of the environment
            train_env = gym.make(f"CustomHopper-{train_type}-v0")
            train_env = Monitor(train_env, log_dir)  # track and save the logs of the training in log_dir

            # Display the parameters of the train environment
            print("State space:", train_env.observation_space)  # state-space
            print("Action space:", train_env.action_space)  # action-space
            print("Dynamics parameters:", train_env.get_parameters())  # masses of each link of the Hopper

            # Creation of the model using either SAC or PPO
            if alg == "SAC":
                model = SAC("MlpPolicy", train_env, device=args.device)
            else:
                model = PPO("MlpPolicy", train_env, device=args.device,gamma = 0.998 ,learning_rate = 0.0003, clip_range = 0.23)

            model.set_logger(new_logger)  # Set the TensorBoard logger for the model

            callback = compute_callback(dr_type, train_env)
            model.learn(total_timesteps=args.train_timesteps, progress_bar=True, callback=callback)

            # Handle correct saving by renaming already saved models
            model_name, evaluation_name, monitor_name, _ = compute_names(alg, train_type, dr_type)
            save_rename_files(model, model_name, evaluation_name, monitor_name, log_dir)

    # --------- TESTING THE MODEL ---------
    for title, (source_env, target_env) in zip(titles, source_target_tuples):
        # loading the model
        source_model = load_model(alg, source_env, dr_type)
        # **** Testing the model ****
        # Display title
        print(title)
        target_env = gym.make(f"CustomHopper-{target_env}-v0")

        test(source_model, target_env, deterministic=False)

if __name__ == "__main__":
    main()

