"""
DEMS Environment for RL Training
=================================
Gymnasium-compatible environment for training RL agents to manage a
microgrid with renewable generation, stochastic demand, and battery storage.

Physics:
    Each of *num_nodes* nodes has:
        - Renewable **generation** (solar-like, with time-of-day curve + noise)
        - Stochastic **demand** (base load + random variation)
        - **Battery** storage with finite capacity

    The agent controls per-node charge/discharge.  A positive action charges
    the battery (absorbing grid power), a negative action discharges (injecting
    power).  The grid frequency deviates from 50 Hz proportionally to the
    supply-demand mismatch:
        net_power = Σ generation − Σ demand − Σ charge_discharge
        freq = 50 + clip(net_power / Σ demand × 0.5, ±0.5)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional


class DEMSEnvironment(gym.Env):
    """
    Custom Gym environment for DEMS with realistic energy dynamics.

    Observation (44-dim, all normalised to [0, 1]):
        Per node (×10): [storage_ratio, demand_ratio, generation_ratio, local_imbalance]
        Global  (×4) : [freq_norm, voltage_norm, time_progress, avg_soc]

    Action (10-dim, continuous [0, 1]):
        0 → full discharge,  0.5 → neutral,  1 → full charge
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, num_nodes: int = 10, max_steps: int = 1000):
        super().__init__()

        self.num_nodes = num_nodes
        self.max_steps = max_steps
        self.current_step = 0

        # Observation space: per-node [storage, demand, generation, imbalance] + 4 global
        obs_size = 4 + (num_nodes * 4)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_size,), dtype=np.float32
        )

        # Action space: charge/discharge decisions for each node [0, 1]
        self.action_space = spaces.Box(
            low=0, high=1, shape=(num_nodes,), dtype=np.float32
        )

        # ── Physical constraints ─────────────────────────────────────
        self.max_storage_per_node = 100.0      # kWh
        self.charge_discharge_rate = 10.0      # kWh per step
        self.base_demand_per_node = 20.0       # kWh per step average
        self.demand_variance = 0.2             # ±20 % demand noise
        self.base_generation_per_node = 22.0   # kWh — slightly above avg demand
        self.generation_variance = 0.25        # ±25 % renewable noise
        self.max_frequency = 50.5              # Hz
        self.min_frequency = 49.5              # Hz
        self.nominal_voltage = 230.0           # V
        self.max_voltage_deviation = 10.0      # V

        # ── Internal state ───────────────────────────────────────────
        self.state = None
        self.episode_reward = 0.0
        self.node_storage = None
        self.node_loads = None
        self.node_generation = None
        self.prev_action = None  # for action-smoothness reward

    # ─────────────────────── helpers ──────────────────────────────────

    def _initialize_state(self) -> None:
        """Set storage to 50 %, sample initial loads / generation, build obs."""
        self.node_storage = np.full(self.num_nodes, self.max_storage_per_node * 0.5)
        self.node_loads = self.np_random.uniform(
            self.base_demand_per_node * 0.8,
            self.base_demand_per_node * 1.2,
            self.num_nodes,
        )
        self.node_generation = self.np_random.uniform(
            self.base_generation_per_node * 0.7,
            self.base_generation_per_node * 1.3,
            self.num_nodes,
        )
        self.prev_action = np.full(self.num_nodes, 0.5, dtype=np.float32)
        self.state = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        self._build_observation(50.0, self.nominal_voltage)

    def _build_observation(self, global_freq: float, global_volt: float) -> None:
        """Write normalised observation into *self.state*."""
        gen_max = self.base_generation_per_node * 2.0
        load_max = self.base_demand_per_node * 1.5
        imbalance_scale = self.base_demand_per_node * 2.0

        for i in range(self.num_nodes):
            idx = i * 4
            self.state[idx]     = self.node_storage[i] / self.max_storage_per_node
            self.state[idx + 1] = self.node_loads[i] / load_max
            self.state[idx + 2] = self.node_generation[i] / gen_max
            # Local generation-demand imbalance centred at 0.5
            local_imb = (self.node_generation[i] - self.node_loads[i]) / imbalance_scale
            self.state[idx + 3] = np.clip(0.5 + local_imb, 0.0, 1.0)

        g = self.num_nodes * 4
        self.state[g]     = (global_freq - self.min_frequency) / (self.max_frequency - self.min_frequency)
        self.state[g + 1] = (global_volt - (self.nominal_voltage - self.max_voltage_deviation)) / (2 * self.max_voltage_deviation)
        self.state[g + 2] = self.current_step / max(self.max_steps, 1)
        self.state[g + 3] = np.mean(self.node_storage) / self.max_storage_per_node

    def _get_obs(self) -> np.ndarray:
        if self.state is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        return np.clip(self.state, 0.0, 1.0).astype(np.float32)

    def _validate_action(self, action: np.ndarray) -> np.ndarray:
        if action.shape != (self.num_nodes,):
            raise ValueError(
                f"Action shape {action.shape} does not match expected ({self.num_nodes},)"
            )
        return np.clip(action, 0.0, 1.0)

    # ─────────────────────── dynamics ─────────────────────────────────

    def _update_dynamics(self, action: np.ndarray) -> Dict[str, Any]:
        """
        Advance the microgrid by one step.

        Power balance:
            net_power = Σ generation − Σ demand − Σ charge_discharge
            > 0  →  excess (frequency rises)
            < 0  →  deficit (frequency drops)
            ≈ 0  →  balanced (50 Hz)
        """
        metrics: Dict[str, Any] = {}

        # 1. Stochastic demand ─────────────────────────────────────────
        demand_noise = self.np_random.uniform(
            -self.demand_variance, self.demand_variance, self.num_nodes
        )
        self.node_loads = self.base_demand_per_node * (1.0 + demand_noise)
        self.node_loads = np.clip(self.node_loads, 0.0, self.base_demand_per_node * 1.5)

        # 2. Renewable generation (solar-like time-of-day curve + noise) ──
        gen_noise = self.np_random.uniform(
            -self.generation_variance, self.generation_variance, self.num_nodes
        )
        time_factor = 0.7 + 0.3 * np.sin(np.pi * self.current_step / max(self.max_steps, 1))
        self.node_generation = (
            self.base_generation_per_node * (1.0 + gen_noise) * time_factor
        )
        self.node_generation = np.clip(
            self.node_generation, 0.0, self.base_generation_per_node * 2.0
        )

        # 3. Agent action → charge / discharge ────────────────────────
        charge_discharge = (action - 0.5) * 2.0 * self.charge_discharge_rate

        # 4. Physical battery constraints ─────────────────────────────
        headroom = self.max_storage_per_node - self.node_storage   # max charge
        charge_discharge = np.clip(charge_discharge, -self.node_storage, headroom)

        # 5. Update storage (battery only changes via agent action) ───
        self.node_storage = np.clip(
            self.node_storage + charge_discharge, 0.0, self.max_storage_per_node
        )

        # 6. Grid power balance ───────────────────────────────────────
        #    net > 0 ⇒ surplus, net < 0 ⇒ deficit
        total_generation = np.sum(self.node_generation)
        total_demand = np.sum(self.node_loads)
        total_battery = np.sum(charge_discharge)  # +ve = charging (absorbs)
        net_power = total_generation - total_demand - total_battery
        supply_demand_error = abs(net_power) / (total_demand + 1e-6)

        # 7. Grid frequency ───────────────────────────────────────────
        freq_deviation = (net_power / (total_demand + 1e-6)) * 0.5
        global_freq = 50.0 + np.clip(freq_deviation, -0.5, 0.5)

        # 8. Grid voltage (from storage level) ────────────────────────
        avg_storage_ratio = np.mean(self.node_storage) / self.max_storage_per_node
        voltage_dev = (avg_storage_ratio - 0.5) * 2.0 * self.max_voltage_deviation
        global_volt = self.nominal_voltage + np.clip(
            voltage_dev, -self.max_voltage_deviation, self.max_voltage_deviation
        )

        # 9. Pack metrics ─────────────────────────────────────────────
        metrics["storage_levels"]       = self.node_storage.copy()
        metrics["demands"]              = self.node_loads.copy()
        metrics["generation"]           = self.node_generation.copy()
        metrics["charge_discharge"]     = charge_discharge.copy()
        metrics["global_frequency"]     = float(global_freq)
        metrics["global_voltage"]       = float(global_volt)
        metrics["supply_demand_ratio"]  = float(total_generation / (total_demand + 1e-6))
        metrics["supply_demand_error"]  = float(supply_demand_error)
        metrics["net_power"]            = float(net_power)

        # 10. Rebuild observation vector ───────────────────────────────
        self._build_observation(global_freq, global_volt)

        return metrics

    # ─────────────────────── reward ───────────────────────────────────

    def _calculate_reward(self, metrics: Dict[str, Any], action: np.ndarray) -> float:
        """
        Multi-objective reward with clear gradient signal.

        Components (max ≈ 3.8 per step when perfectly balanced):
            1. Supply-demand balance  (up to 2.0)  ← primary objective
            2. Frequency stability    (up to 1.0)  ← directly from S-D
            3. Battery SoC balance    (up to 0.5)  ← keep near 50 %
            4. Action smoothness      (up to 0.3)  ← avoid jerky control
            5. Critical-SoC penalty   (−0.1 per node < 10 % or > 90 %)
        """
        reward = 0.0

        # 1. Supply-demand balance (most important)
        sd_err = metrics["supply_demand_error"]
        reward += 2.0 * max(0.0, 1.0 - sd_err * 5.0)

        # 2. Frequency stability (50 Hz target)
        freq_dev = abs(metrics["global_frequency"] - 50.0)
        reward += 1.0 * max(0.0, 1.0 - freq_dev / 0.5)

        # 3. Average SoC near 50 %
        avg_soc = np.mean(metrics["storage_levels"]) / self.max_storage_per_node
        soc_dev = abs(avg_soc - 0.5)
        reward += 0.5 * max(0.0, 1.0 - soc_dev / 0.4)

        # 4. Action smoothness (penalise jerk between consecutive steps)
        if self.prev_action is not None:
            jerk = np.mean(np.abs(action - self.prev_action))
            reward += 0.3 * max(0.0, 1.0 - jerk / 0.5)
        else:
            reward += 0.3  # first step gets full smoothness bonus

        # 5. Critical-SoC penalty
        soc_ratios = metrics["storage_levels"] / self.max_storage_per_node
        n_critical = np.sum((soc_ratios < 0.1) | (soc_ratios > 0.9))
        reward -= 0.1 * n_critical

        return float(reward)

    # ─────────────────────── gym API ──────────────────────────────────

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Advance the environment one timestep.

        Returns:
            observation, reward, terminated, truncated, info
        """
        action = self._validate_action(action)
        self.current_step += 1

        metrics = self._update_dynamics(action)
        reward = self._calculate_reward(metrics, action)
        self.episode_reward += reward
        self.prev_action = action.copy()

        terminated = False
        truncated = self.current_step >= self.max_steps

        info = {
            "step":               self.current_step,
            "episode_reward":     self.episode_reward,
            "action_applied":     action.copy(),
            "storage_levels":     metrics["storage_levels"],
            "demands":            metrics["demands"],
            "generation":         metrics["generation"],
            "charge_discharge":   metrics["charge_discharge"],
            "global_frequency":   metrics["global_frequency"],
            "global_voltage":     metrics["global_voltage"],
            "avg_storage_ratio":  float(np.mean(metrics["storage_levels"]) / self.max_storage_per_node),
            "supply_demand_error": metrics["supply_demand_error"],
            "net_power":          metrics["net_power"],
        }

        return self._get_obs(), reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        self.current_step = 0
        self.episode_reward = 0.0
        self.prev_action = None

        self._initialize_state()

        info = {}
        if options is not None:
            info["options"] = options

        return self._get_obs(), info

    def render(self, mode: str = "human") -> None:
        print(f"Step: {self.current_step}, Episode Reward: {self.episode_reward:.2f}")

    def close(self) -> None:
        pass