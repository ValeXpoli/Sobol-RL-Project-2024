import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import optuna

def discount_rewards(r, gamma):
    """Computes discounted rewards."""
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

class ActorCritic(torch.nn.Module):
    """Shared network for both actor and critic."""
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        # Shared network layers
        self.fc1_shared = torch.nn.Linear(state_space, self.hidden)
        self.fc2_shared = torch.nn.Linear(self.hidden, self.hidden)

        # Actor-specific layers
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space) + init_sigma)

        # Critic-specific layers
        self.fc3_critic_value = torch.nn.Linear(self.hidden, 1)

    def forward_shared(self, x):
        """Forward pass through the shared network."""
        x = self.tanh(self.fc1_shared(x))
        x = self.tanh(self.fc2_shared(x))
        return x

    def forward_actor(self, x):
        """Forward pass through the actor-specific network."""
        x = self.forward_shared(x)
        action_mean = self.fc3_actor_mean(x)
        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)
        return normal_dist

    def forward_critic(self, x):
        """Forward pass through the critic-specific network."""
        x = self.forward_shared(x)
        value = self.fc3_critic_value(x)
        return value

class Agent:
    def __init__(self, state_space, action_space,gamma=0.99, device="cpu", lr=1e-3):
        """Initializes the agent with shared actor and critic networks."""
        self.train_device = device

        # Shared actor and critic network
        self.policy = ActorCritic(state_space, action_space).to(self.train_device)

        self.optimizer = Adam(self.policy.parameters(), lr=lr)
        self.writer = SummaryWriter()
        self.gamma = gamma

        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []
        self.episode = 0

    def update_policy(self):
        """Updates the policy network using the stored experiences."""
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        values = self.policy.forward_critic(states).squeeze(-1)
        next_values = self.policy.forward_critic(next_states).squeeze(-1)

        # Compute bootstrapped discounted return estimate
        td_target = rewards + self.gamma * next_values * (1 - done)
        # Compute advantage term
        td_error = td_target - values
        advantages = td_error

        # Compute actor loss and critic loss
        actor_loss = (action_log_probs * advantages).mean()
        critic_loss = F.mse_loss(values, td_target)

        total_loss = -(actor_loss) + critic_loss 

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.writer.add_scalar('Loss/actor_loss', actor_loss.item(), self.episode)
        self.writer.add_scalar('Loss/critic_loss', critic_loss.item(), self.episode)
        self.writer.add_scalar('Loss/total_loss', total_loss.item(), self.episode)

        # Reset storage
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []
        self.episode += 1

    def get_action(self, state, evaluation=False):
        """Selects an action based on the current policy."""
        x = torch.from_numpy(state).float().to(self.train_device)
        normal_dist = self.policy.forward_actor(x)

        if evaluation:
            return normal_dist.mean, None
        else:
            action = normal_dist.sample()
            action_log_prob = normal_dist.log_prob(action).sum()
            return action, action_log_prob

    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        """Stores the outcome of an action for future policy updates."""
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.tensor([reward], dtype=torch.float32))
        self.done.append(done)

    def close(self):
        self.writer.close()


