import argparse
import torch
import gym
from env.custom_hopper import *
from actor_critic_agent import Agent,ActorCritic

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str, help='Path to the trained model')
    parser.add_argument('--device', default='cpu', type=str, help='Device to use for inference [cpu, cuda]')
    parser.add_argument('--render', action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=1000, type=int, help='Number of test episodes')
    return parser.parse_args()

def main():
    args = parse_args()

    # Initialize environment
    env = gym.make('CustomHopper-target-v0')
    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())

    # Define dimensions
    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    # Initialize policy (actor network) and load model weights
    policy = ActorCritic(observation_space_dim, action_space_dim)
    policy.load_state_dict(torch.load(args.model, map_location=args.device))

    
    agent = Agent(observation_space_dim, action_space_dim, device=args.device)
    agent.policy = policy

    rewards = []


    for episode in range(args.episodes):
        done = False
        test_reward = 0
        state = env.reset()

        while not done:
            action, _ = agent.get_action(state, evaluation=True)
            state, reward, done, _ = env.step(action.detach().cpu().numpy())


            if args.render:
                env.render()

            test_reward += reward

        print(f"Episode: {episode} | Return: {test_reward}")
        rewards.append(test_reward)

    rewards = torch.tensor(rewards)
    std_rewards, mean_rewards = torch.std_mean(rewards)


    print("\n==================")
    print("Mean Reward: ", mean_rewards.item())
    print("Std Reward: ", std_rewards.item())


if __name__ == '__main__':
    main()

