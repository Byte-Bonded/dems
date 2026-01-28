import torch
import numpy as np
from microgrid_env import MicrogridEnv
from agent import ActorCritic

def execute_agent():
    # 1. Setup Environment
    env = MicrogridEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 2. Load the Model
    model_path = "agent_logic.pt"
    try:
        policy = ActorCritic(state_dim, action_dim)
        policy.load_state_dict(torch.load(model_path))
        policy.eval() # Set to evaluation mode
        print(f"Successfully loaded {model_path}")
    except FileNotFoundError:
        print(f"Error: {model_path} not found. Please run train.py first.")
        return

    # 3. Run Inference Loop
    print("\n--- Starting DEMS Agent Execution ---")
    state, _ = env.reset()
    done = False
    total_cost = 0
    
    print(f"{ 'Step':<5} | {'SoC':<6} | {'Load':<6} | {'PV':<6} | {'Price':<6} | {'Action (Bat kW)':<15} | {'Cost':<6}")
    print("-" * 70)
    
    step_count = 0
    while not done:
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state)
        
        # Get Action (Deterministic for execution usually, or sample if stochastic)
        # Here we sample, but in strict deployment you might want mean directly.
        with torch.no_grad():
            action, _ = policy.get_action(state_tensor)
        
        # Execute in Environment
        # Action is scaled inside env.step, here it's [-1, 1]
        raw_action = action.numpy()
        next_state, reward, done, _, info = env.step(raw_action)
        
        # Visualization
        bat_power_kw = raw_action[0] * env.max_power
        cost = -reward
        total_cost += cost
        
        # State: [SoC, Load, PV, Price]
        print(f"{step_count:<5} | {state[0]:.2f}   | {state[1]:.2f}   | {state[2]:.2f}   | {state[3]:.2f}   | {bat_power_kw: .2f}            | {cost:.2f}")
        
        state = next_state
        step_count += 1
        
    print("-" * 70)
    print(f"Total Operational Cost for Day: ${total_cost:.2f}")

if __name__ == "__main__":
    execute_agent()
