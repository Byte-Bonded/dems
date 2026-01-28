import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MicrogridEnv(gym.Env):
    def __init__(self):
        super(MicrogridEnv, self).__init__()

        # Microgrid Parameters
        self.battery_capacity = 13.5  # kWh (e.g., Tesla Powerwall)
        self.max_power = 5.0          # kW
        self.dt = 1.0                 # Time step (1 hour)
        
        # Action Space: Battery Power (negative=discharge, positive=charge)
        # Normalized to [-1, 1] for stable RL training, scaled inside step()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation Space: [SoC (0-1), Load (kW), PV (kW), Grid Price ($/kWh)]
        # We assume reasonable bounds for normalization
        low = np.array([0.0, 0.0, 0.0, 0.0])
        high = np.array([1.0, 20.0, 20.0, 10.0]) # high upper bounds
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.state = None
        self.steps = 0
        self.max_steps = 24  # Simulate one day

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initial State
        self.soc = 0.5 # 50% charge
        self.steps = 0
        
        # Randomize initial load/pv/price slightly for variety
        load = np.random.uniform(1.0, 5.0)
        pv = np.random.uniform(0.0, 4.0)
        price = np.random.uniform(0.10, 0.30)
        
        self.state = np.array([self.soc, load, pv, price], dtype=np.float32)
        
        return self.state, {}

    def step(self, action):
        # Unpack action and scale it
        # action is [-1, 1], scale to [-max_power, max_power]
        power_kW = float(action[0]) * self.max_power
        
        soc, load, pv, price = self.state

        # Calculate Energy change
        energy_delta = power_kW * self.dt # kWh
        
        # Physical Constraints Check
        # If charging (power > 0) and full
        if power_kW > 0 and soc >= 1.0:
            power_kW = 0
            energy_delta = 0
        # If discharging (power < 0) and empty
        elif power_kW < 0 and soc <= 0.0:
            power_kW = 0
            energy_delta = 0
            
        # Update SoC
        new_soc = soc + (energy_delta / self.battery_capacity)
        new_soc = np.clip(new_soc, 0.0, 1.0)
        
        # Net Grid Interaction
        # Power Balance: Load = PV + Battery_Discharge + Grid_Import
        # Grid_Import = Load - PV + Battery_Charge
        # Note: power_kW is defined as Battery input (Charge). So Discharge is -power_kW.
        
        grid_exchange = load - pv + power_kW
        
        # Cost Calculation
        # If grid_exchange > 0, we buy electricity (cost positive)
        # If grid_exchange < 0, we sell electricity (cost negative / profit)
        cost = grid_exchange * price
        
        # Reward Function
        # We want to minimize cost, so reward is negative cost.
        # Add penalty for SoC limits to encourage safe operation if desired, 
        # but clipping handles physics.
        reward = -cost
        
        # Next State Generation (Simulated simple dynamics)
        self.steps += 1
        
        # Simple dummy profiles for next step
        next_load = np.random.uniform(1.0, 8.0)
        # PV is higher during midday (steps 8-16)
        if 8 <= self.steps <= 16:
            next_pv = np.random.uniform(2.0, 6.0)
        else:
            next_pv = 0.0
            
        next_price = 0.15 if self.steps < 16 else 0.30 # Peak pricing in evening
        
        self.state = np.array([new_soc, next_load, next_pv, next_price], dtype=np.float32)
        
        done = self.steps >= self.max_steps
        truncated = False
        
        info = {
            "cost": cost,
            "soc": new_soc,
            "grid_power": grid_exchange
        }
        
        return self.state, reward, done, truncated, info

    def render(self):
        print(f"Step: {self.steps}, State: {self.state}")
