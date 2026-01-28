import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from microgrid_env import MicrogridEnv
from agent import ActorCritic

# Hyperparameters
LR = 3e-4
GAMMA = 0.99
EPS_CLIP = 0.2
K_EPOCHS = 4
BATCH_SIZE = 64
TOTAL_TIMESTEPS = 10000

def train():
    env = MicrogridEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Initialize Agent
    policy = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    
    # Training Loop buffers
    states = []
    actions = []
    log_probs = []
    rewards = []
    dones = []
    
    print("Starting Training...")
    
    state, _ = env.reset()
    episode_reward = 0
    
    for t in range(TOTAL_TIMESTEPS):
        # 1. Collect Data
        state_tensor = torch.FloatTensor(state)
        action, log_prob = policy.get_action(state_tensor)
        
        next_state, reward, done, _, _ = env.step(action.detach().numpy())
        
        states.append(state_tensor)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        dones.append(done)
        
        state = next_state
        episode_reward += reward
        
        if done:
            state, _ = env.reset()
            # print(f"Episode finished. Reward: {episode_reward:.2f}")
            episode_reward = 0

        # 2. Update Policy (PPO) every batch
        if (t + 1) % BATCH_SIZE == 0:
            # Prepare batch
            old_states = torch.stack(states)
            old_actions = torch.stack(actions)
            old_log_probs = torch.stack(log_probs)
            
            # Compute Returns (Monte Carlo estimate)
            returns = []
            discounted_sum = 0
            for r, is_done in zip(reversed(rewards), reversed(dones)):
                if is_done:
                    discounted_sum = 0
                discounted_sum = r + (GAMMA * discounted_sum)
                returns.insert(0, discounted_sum)
            
            returns = torch.tensor(returns, dtype=torch.float32)
            # Normalize returns
            returns = (returns - returns.mean()) / (returns.std() + 1e-7)
            
            # PPO Update Steps
            for _ in range(K_EPOCHS):
                logprobs, state_values, dist_entropy = policy.evaluate(old_states, old_actions)
                
                state_values = torch.squeeze(state_values)
                
                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_log_probs.detach())
                
                # Finding Surrogate Loss
                advantages = returns - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-EPS_CLIP, 1+EPS_CLIP) * advantages
                
                # Final loss
                loss = -torch.min(surr1, surr2) + 0.5*nn.MSELoss()(state_values, returns) - 0.01*dist_entropy
                
                # Take Gradient Step
                optimizer.zero_grad()
                loss.mean().backward()
                optimizer.step()
                
            # Clear buffers
            states = []
            actions = []
            log_probs = []
            rewards = []
            dones = []
            
            print(f"Step {t+1}/{TOTAL_TIMESTEPS} | Loss: {loss.mean().item():.4f}")

    # Save Model
    print("Training Complete. Saving model to agent_logic.pt")
    torch.save(policy.state_dict(), "agent_logic.pt")

if __name__ == "__main__":
    train()
