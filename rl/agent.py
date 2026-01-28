import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        
        # Critic Network (Value Function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Actor Network (Policy)
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh() # Map to [-1, 1] range to match env action space
        )
        
        # Learnable Log Standard Deviation
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, state):
        return self.critic(state)

    def get_action(self, state):
        mean = self.actor_mean(state)
        std = self.actor_logstd.exp().expand_as(mean)
        dist = Normal(mean, std)
        
        action = dist.sample()
        # We clamp for numerical stability during log_prob, 
        # but the env expects [-1, 1]. The tanh on mean helps center it.
        action_clipped = torch.clamp(action, -1.0, 1.0) 
        
        log_prob = dist.log_prob(action).sum(axis=-1)
        
        return action_clipped, log_prob

    def evaluate(self, state, action):
        mean = self.actor_mean(state)
        std = self.actor_logstd.exp().expand_as(mean)
        dist = Normal(mean, std)
        
        action_logprobs = dist.log_prob(action).sum(axis=-1)
        dist_entropy = dist.entropy().sum(axis=-1)
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy
