"""
DEMS Environment for RL Training
Gymnasium-compatible environment for training RL agents with physics-based dynamics
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any


class DEMSEnvironment(gym.Env):
    """
    Custom Gym environment for DEMS with realistic energy dynamics
    Agents interact with this environment to learn optimal energy management
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, num_nodes: int = 10, max_steps: int = 1000):
        super().__init__()

        self.num_nodes = num_nodes
        self.max_steps = max_steps
        self.current_step = 0

        # Observation space: [storage_level, load, frequency, voltage] * nodes
        obs_size = 4 + (num_nodes * 4)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_size,), dtype=np.float32
        )

        # Action space: charge/discharge decisions for each node [0, 1]
        # 0 = maximum discharge, 0.5 = neutral, 1 = maximum charge
        self.action_space = spaces.Box(
            low=0, high=1, shape=(num_nodes,), dtype=np.float32
        )

        # Physical constraints and parameters
        self.max_storage_per_node = 100.0  # kWh
        self.charge_discharge_rate = 10.0  # kWh per step
        self.base_demand_per_node = 20.0   # kWh per step
        self.demand_variance = 0.2         # Stochastic demand
        self.max_frequency = 50.5          # Hz
        self.min_frequency = 49.5          # Hz
        self.nominal_voltage = 230.0       # V
        self.max_voltage_deviation = 10.0  # V

        # Internal state representation
        # state = [storage[0], load[0], freq[0], volt[0], storage[1], ..., global_freq, global_voltage]
        self.state = None
        self.episode_reward = 0
        self.node_storage = None  # Current storage level per node
        self.node_loads = None    # Current demand per node

    def _initialize_state(self) -> None:
        """
        Initialize state with reasonable defaults
        Storage at 50% capacity, loads at 50% of demand, frequency/voltage nominal
        """
        self.node_storage = np.full(self.num_nodes, self.max_storage_per_node * 0.5)
        self.node_loads = np.random.uniform(
            self.base_demand_per_node * 0.3,
            self.base_demand_per_node * 0.7,
            self.num_nodes
        )

        # Build state: [storage, load, freq, volt] per node + global metrics
        self.state = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        global_freq = 50.0
        global_volt = self.nominal_voltage

        for i in range(self.num_nodes):
            idx = i * 4
            # Normalize to [0, 1]
            self.state[idx] = self.node_storage[i] / self.max_storage_per_node
            self.state[idx + 1] = self.node_loads[i] / self.base_demand_per_node
            self.state[idx + 2] = (global_freq - self.min_frequency) / (self.max_frequency - self.min_frequency)
            self.state[idx + 3] = (global_volt - (self.nominal_voltage - self.max_voltage_deviation)) / (2 * self.max_voltage_deviation)

    def _get_obs(self) -> np.ndarray:
        """Get current observation"""
        if self.state is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        return np.clip(self.state, 0, 1).astype(np.float32)

    def _validate_action(self, action: np.ndarray) -> np.ndarray:
        """
        Validate and constrain action to allowed range
        
        Args:
            action: Raw action from agent
            
        Returns:
            Validated and clipped action
            
        Raises:
            ValueError: If action shape is incorrect
        """
        if action.shape != (self.num_nodes,):
            raise ValueError(
                f"Action shape {action.shape} does not match expected shape ({self.num_nodes},)"
            )
        
        # Clip actions to [0, 1] range
        return np.clip(action, 0.0, 1.0)

    def _update_dynamics(self, action: np.ndarray) -> Dict[str, Any]:
        """
        Update DEMS state based on action and physics constraints
        
        Args:
            action: Validated action array [0, 1] per node
                   0 = maximum discharge, 0.5 = neutral, 1 = maximum charge
                   
        Returns:
            Dictionary with state metrics and constraints info
        """
        metrics = {}

        # Stochastically vary demand for realism
        demand_variation = np.random.uniform(
            -self.demand_variance,
            self.demand_variance,
            self.num_nodes
        )
        self.node_loads = self.base_demand_per_node * (1.0 + demand_variation)
        self.node_loads = np.clip(self.node_loads, 0, self.base_demand_per_node * 1.5)

        # Convert action [0, 1] to charge/discharge rate [-rate, +rate]
        # 0 = -rate (discharge), 0.5 = 0 (neutral), 1 = +rate (charge)
        charge_discharge = (action - 0.5) * 2 * self.charge_discharge_rate

        # Update storage levels with bounds
        new_storage = self.node_storage + charge_discharge - self.node_loads
        self.node_storage = np.clip(new_storage, 0, self.max_storage_per_node)

        metrics["storage_levels"] = self.node_storage.copy()
        metrics["demands"] = self.node_loads.copy()
        metrics["charge_discharge"] = charge_discharge.copy()

        # Calculate grid frequency based on supply-demand balance
        total_generation = np.sum(charge_discharge) + np.sum(self.node_loads)
        total_demand = np.sum(self.node_loads)
        supply_demand_ratio = total_generation / (total_demand + 1e-6)
        
        # Frequency oscillates around 50 Hz based on balance
        freq_deviation = (supply_demand_ratio - 1.0) * 1.0  # ±1 Hz per unit imbalance
        global_freq = 50.0 + np.clip(freq_deviation, -0.5, 0.5)
        metrics["global_frequency"] = global_freq

        # Calculate grid voltage based on storage state and demand
        avg_storage = np.mean(self.node_storage) / self.max_storage_per_node
        voltage_deviation = (avg_storage - 0.5) * 2 * self.max_voltage_deviation
        global_volt = self.nominal_voltage + np.clip(voltage_deviation, -self.max_voltage_deviation, self.max_voltage_deviation)
        metrics["global_voltage"] = global_volt

        # Update state observation
        for i in range(self.num_nodes):
            idx = i * 4
            # Normalize to [0, 1]
            self.state[idx] = self.node_storage[i] / self.max_storage_per_node
            self.state[idx + 1] = self.node_loads[i] / self.base_demand_per_node
            self.state[idx + 2] = (global_freq - self.min_frequency) / (self.max_frequency - self.min_frequency)
            self.state[idx + 3] = (global_volt - (self.nominal_voltage - self.max_voltage_deviation)) / (2 * self.max_voltage_deviation)

        return metrics

    def _calculate_reward(self, metrics: Dict[str, Any], action: np.ndarray) -> float:
        """
        Calculate reward based on DEMS dynamics and constraints
        
        Rewards:
        - Efficient energy balance (supply ~= demand)
        - Grid frequency stability (near 50 Hz)
        - Reasonable storage levels (not empty or over-full)
        - Reasonable action magnitude (avoid extreme swings)
        
        Args:
            metrics: State metrics from dynamics update
            action: Applied action
            
        Returns:
            Reward value
        """
        reward = 0.0

        # Reward for frequency stability (target 50 Hz)
        freq = metrics["global_frequency"]
        freq_penalty = abs(freq - 50.0)
        reward += 1.0 - min(freq_penalty / 0.5, 1.0)  # 1.0 at 50 Hz, 0.0 if |deviation| > 0.5

        # Reward for storage balance (target ~50% capacity)
        avg_storage_ratio = np.mean(metrics["storage_levels"]) / self.max_storage_per_node
        storage_penalty = abs(avg_storage_ratio - 0.5)
        reward += 0.5 - min(storage_penalty / 0.5, 0.5)  # 0.5 at 50%, 0.0 if |deviation| > 50%

        # Penalty for extreme actions (encourage smooth control)
        action_magnitude = np.mean(np.abs(action - 0.5))
        reward += 0.3 * (1.0 - min(action_magnitude / 0.5, 1.0))

        # Penalty for over/under supply
        total_gen = np.sum(metrics["charge_discharge"]) + np.sum(metrics["demands"])
        total_demand = np.sum(metrics["demands"])
        supply_demand_error = abs(total_gen - total_demand) / (total_demand + 1e-6)
        reward -= 0.2 * min(supply_demand_error, 1.0)

        return float(reward)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the DEMS environment
        
        Args:
            action: Control action for charge/discharge per node [0, 1]
            
        Returns:
            observation: Current state observation
            reward: Step reward
            done: Episode termination flag
            info: Additional information (metrics, validated action, etc.)
            
        Raises:
            ValueError: If action shape is invalid
        """
        # Validate action at start of step
        try:
            action = self._validate_action(action)
        except ValueError as e:
            raise ValueError(f"Invalid action at step {self.current_step}: {str(e)}")

        self.current_step += 1

        # Update DEMS dynamics based on action and physics constraints
        metrics = self._update_dynamics(action)

        # Calculate reward based on resulting state
        reward = self._calculate_reward(metrics, action)
        self.episode_reward += reward

        # Check if episode is done
        done = self.current_step >= self.max_steps

        # Build info dictionary with applied action and resulting state metrics
        info = {
            "step": self.current_step,
            "episode_reward": self.episode_reward,
            "action_applied": action.copy(),
            "storage_levels": metrics["storage_levels"].copy(),
            "demands": metrics["demands"].copy(),
            "charge_discharge": metrics["charge_discharge"].copy(),
            "global_frequency": metrics["global_frequency"],
            "global_voltage": metrics["global_voltage"],
            "avg_storage_ratio": np.mean(metrics["storage_levels"]) / self.max_storage_per_node,
        }

        return self._get_obs(), reward, done, info

    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        self.episode_reward = 0
        self._initialize_state()  # Initialize storage, loads, and state
        return self._get_obs()

    def render(self, mode: str = "human") -> None:
        """Render environment state"""
        print(f"Step: {self.current_step}, Episode Reward: {self.episode_reward:.2f}")

    def close(self) -> None:
        """Close environment"""
        pass
