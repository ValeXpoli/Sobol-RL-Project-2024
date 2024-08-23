import argparse
import os
import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import optuna
from env.custom_hopper import *
from actor_critic_agent import Agent
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
os.makedirs("Home/Documents/github/MLDL/ActorCriticNew", exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", default=15000, type=int, help="Number of training episodes")
    parser.add_argument("--print-every", default=100, type=int, help="Print info every <> episodes")
    parser.add_argument("--device", default="cpu", type=str, help="network device [cpu, cuda]")
    parser.add_argument("--outName", default="model.mdl", type=str, help="output filename")
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
    parser.add_argument('--log-dir', default='runs', type=str, help='Directory to save TensorBoard logs')
    parser.add_argument('--n-trials', default=12, type=int, help='Number of Optuna trials')
    return parser.parse_args()

args = parse_args()

def optimize_agent(trial):
    try:
        env = gym.make("CustomHopper-source-v0")
    except gym.error.Error:
        print("Error: Failed to initialize the environment.")
        return -float('inf')

    # Parametri da ottimizzare
    gamma = trial.suggest_float('gamma', 0.8, 0.99)
    lr = trial.suggest_float('lr', 1e-3, 1e-2)

    action_space_dim = env.action_space.shape[-1]
    observation_space_dim = env.observation_space.shape[-1]

    agent = Agent(observation_space_dim, action_space_dim, gamma=gamma, device=args.device, lr=lr)

    rewards_results = np.zeros((args.n_episodes,))
    previous_best = 0
    total_reward = 0

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=args.log_dir)
    for episode in range(args.n_episodes):
        done = False
        train_reward = 0
        state = env.reset()  # Reset the environment and observe the initial state
        while not done:  # Loop until the episode is over
            action, action_log_prob = agent.get_action(state)
            previous_state = state
            state, reward, done, _ = env.step(action.detach().cpu().numpy())
            if args.render:
                env.render()

            agent.store_outcome(previous_state, state, action_log_prob, reward, done)
            train_reward += reward

        total_reward += train_reward

        if train_reward > previous_best:
            torch.save(agent.policy.state_dict(), args.outName)
            previous_best = train_reward

        rewards_results[episode] = train_reward
        agent.update_policy()   
        writer.add_scalar('Reward/Episode', train_reward, episode)

    writer.close()

    # Return the average reward
    avg_reward = np.mean(total_reward)

    # Definire la funzione obiettivo
    objective = avg_reward
    return objective

def main():
    study_name = "AC_tuning"
    storage_dir = "optuna_trials"
    os.makedirs(storage_dir, exist_ok=True)  # Crea la cartella se non esiste
    storage_name = f"sqlite:///{storage_dir}/optuna_trials.db"  # Aggiungi il prefisso 'sqlite:///' e un'estensione al file
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction='maximize')
    study.optimize(optimize_agent, n_trials=args.n_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main()

