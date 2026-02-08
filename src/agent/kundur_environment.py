"""
Kundur RL Environment for Fast PSS/Controller Testing
Simplified Gymnasium environment for the Kundur Two-Area System

This environment is optimized for:
- Power System Stabilizer (PSS) training and validation
- Wide-area damping controller development
- Inter-area oscillation control (0.6 Hz mode)
- Fast RL training (40 states vs 300+ in IEEE 39-bus)

Observation Space (20 dimensions):
- 4 generator states: rotor angle deviation, frequency deviation, P, Q
- 2 area states: net load, voltage
- 2 tie-line states: active power flow, reactive power flow
- Battery SOC for both areas (2)
- DER generation (solar, wind) for both areas (4)
- Voltage and frequency global metrics (2)

Action Space (8 dimensions):
- Generator power setpoints (4): G1, G2, G3, G4
- Battery dispatch (2): BESS_A1, BESS_A2
- Load shedding (2): Area 1, Area 2 (demand response)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
import logging

from ..simulation.kundur import KundurTwoAreaSystem, KundurConfig, AreaID
from ..simulation.power_flow import PowerFlowRunner, PowerFlowConfig

logger = logging.getLogger(__name__)


class KundurEnvironment(gym.Env):
    """
    Kundur Two-Area System RL Environment
    
    Simplified environment for training controllers on inter-area oscillations
    Much faster than full IEEE 39-bus system due to reduced state space
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 1}
    
    def __init__(
        self,
        config: Optional[KundurConfig] = None,
        max_steps: int = 500,
        enable_der: bool = True,
        oscillation_damping_weight: float = 10.0,
        voltage_weight: float = 5.0,
        frequency_weight: float = 5.0,
        reward_scale: float = 0.01,
    ):
        """
        Initialize Kundur RL Environment
        
        Args:
            config: Kundur system configuration
            max_steps: Maximum episode length
            enable_der: Include DER (solar, wind, battery)
            oscillation_damping_weight: Reward weight for damping inter-area swings
            voltage_weight: Penalty weight for voltage deviations
            frequency_weight: Penalty weight for frequency deviations
            reward_scale: Overall reward scaling factor
        """
        super().__init__()
        
        self.config = config or KundurConfig()
        self.max_steps = max_steps
        self.enable_der = enable_der
        self.current_step = 0
        
        # Reward weights
        self.w_damping = oscillation_damping_weight
        self.w_voltage = voltage_weight
        self.w_frequency = frequency_weight
        self.reward_scale = reward_scale
        
        # Create Kundur system
        self.kundur = KundurTwoAreaSystem(config=self.config, enable_der=enable_der)
        self.power_flow = PowerFlowRunner()
        
        # Observation space: 20 dimensions
        # [delta1, omega1, P1, Q1] * 4 generators = 16
        # + [area1_load, area1_voltage, area2_load, area2_voltage] = 4
        # + [tieline_P, tieline_Q] = 2 (if not using DER)
        # OR with DER: + [bess1_soc, bess2_soc, solar1, solar2, wind1, wind2] = 6
        obs_dim = 22 if enable_der else 20
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Action space: 8 dimensions (normalized to [-1, 1])
        # [gen1_delta, gen2_delta, gen3_delta, gen4_delta] = 4 (power adjustments)
        # [bess1_power, bess2_power] = 2 (battery dispatch)
        # [dr1, dr2] = 2 (demand response / load shedding)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(8,),
            dtype=np.float32
        )
        
        # State tracking
        self.state = None
        self.episode_reward = 0.0
        self.previous_tie_flow = 0.0
        self.previous_voltages = None
        
        # Nominal operating point
        self.nominal_gen_power = {
            "G1": 700.0, "G2": 700.0, "G3": 719.0, "G4": 700.0
        }
        self.max_gen_adjustment_mw = 50.0  # +/- 50 MW per step
        self.max_battery_power_mw = 30.0  # Max battery dispatch
        self.max_load_shed_mw = 100.0  # Max demand response
        
        logger.info(f"KundurEnvironment initialized (DER={'enabled' if enable_der else 'disabled'})")
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state
        
        Returns:
            observation: Initial state vector
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Recreate Kundur system for clean reset
        self.kundur = KundurTwoAreaSystem(config=self.config, enable_der=self.enable_der)
        
        # Apply random initial perturbation to create interesting dynamics
        if self.np_random.random() < 0.3:  # 30% chance of initial disturbance
            area = self.np_random.choice([AreaID.AREA_1, AreaID.AREA_2])
            delta_mw = self.np_random.uniform(-50, 50)
            self.kundur.apply_load_perturbation(area, delta_mw)
            
        # Run initial power flow
        result = self.power_flow.run(self.kundur.net, verbose=False)
        
        if not result.converged:
            logger.warning("Initial power flow did not converge, using flat start")
            
        # Get initial state
        self.current_step = 0
        self.episode_reward = 0.0
        self.state = self._get_observation()
        self.previous_tie_flow = self._get_tie_line_flow()
        self.previous_voltages = self._get_voltages()
        
        info = self._get_info()
        
        return self.state, info
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step
        
        Args:
            action: Control action vector (8 dimensions)
            
        Returns:
            observation: Next state
            reward: Reward signal
            terminated: Episode ended due to constraint violation
            truncated: Episode ended due to time limit
            info: Additional information
        """
        self.current_step += 1
        
        # Apply control actions
        self._apply_action(action)
        
        # Run power flow
        result = self.power_flow.run(self.kundur.net, verbose=False)
        
        # Check for divergence
        if not result.converged:
            logger.warning(f"Power flow diverged at step {self.current_step}")
            return self.state, -100.0, True, False, {"converged": False}
            
        # Get new observation
        self.state = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(result)
        self.episode_reward += reward
        
        # Check termination conditions
        terminated = self._check_termination(result)
        truncated = self.current_step >= self.max_steps
        
        # Update tracking
        self.previous_tie_flow = self._get_tie_line_flow()
        self.previous_voltages = self._get_voltages()
        
        info = self._get_info()
        
        return self.state, reward, terminated, truncated, info
        
    def _apply_action(self, action: np.ndarray) -> None:
        """Apply control actions to the system"""
        # Generator power adjustments (actions 0-3)
        gen_names = ["G1", "G2", "G3", "G4"]
        for i, gen_name in enumerate(gen_names):
            delta_mw = action[i] * self.max_gen_adjustment_mw
            new_power = self.nominal_gen_power[gen_name] + delta_mw
            self.kundur.set_generator_setpoint(gen_name, new_power)
            
        # Battery dispatch (actions 4-5)
        if self.enable_der and self.kundur.der_manager:
            bess_names = ["BESS_A1", "BESS_A2"]
            for i, bess_name in enumerate(bess_names):
                power_mw = action[4 + i] * self.max_battery_power_mw
                self.kundur.dispatch_battery(bess_name, power_mw, duration_hours=0.25)
                
        # Demand response / load shedding (actions 6-7)
        # Modify load at load buses
        areas = [AreaID.AREA_1, AreaID.AREA_2]
        for i, area in enumerate(areas):
            load_delta = action[6 + i] * self.max_load_shed_mw
            # Note: Negative action = increase load, positive = decrease (shed)
            self.kundur.apply_load_perturbation(area, -load_delta)
            
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector from current system state"""
        state = self.kundur.get_state()
        obs = []
        
        # Generator states (4 * 4 = 16 dimensions)
        for gen in state["generators"]:
            # Normalize: delta (assume small angle), omega (deviation), P, Q
            obs.extend([
                0.0,  # Rotor angle deviation (requires dynamics simulation)
                0.0,  # Frequency deviation (requires dynamics simulation)
                gen["p_mw"] / 1000.0,  # Normalize by 1000 MW
                gen["q_mvar"] / 500.0,  # Normalize by 500 MVAR
            ])
            
        # Area states (2 * 2 = 4 dimensions)
        for area_name in ["Area1", "Area2"]:
            area = state["areas"][area_name]
            obs.extend([
                area["load_mw"] / 2000.0,  # Normalize by 2000 MW
                area["avg_voltage_pu"],  # Already per-unit
            ])
            
        # Tie-line state (2 dimensions)
        tie_flow_mw = state["system"]["tie_line_flow_mw"]
        obs.extend([
            tie_flow_mw / 500.0,  # Normalize by 500 MW
            0.0,  # Reactive flow (placeholder)
        ])
        
        # DER states (6 dimensions if enabled)
        if self.enable_der and self.kundur.der_manager:
            # Battery SOC (2)
            bess_socs = [0.5, 0.5]  # Default
            for bess_name in ["BESS_A1", "BESS_A2"]:
                if bess_name in self.kundur.der_manager.battery_soc:
                    idx = 0 if "A1" in bess_name else 1
                    bess_socs[idx] = self.kundur.der_manager.battery_soc[bess_name]
            obs.extend(bess_socs)
            
            # Solar and wind generation (4)
            # Get from network sgen elements
            obs.extend([0.5, 0.5, 0.5, 0.5])  # Placeholder: normalized generation
            
        return np.array(obs, dtype=np.float32)
        
    def _calculate_reward(self, result) -> float:
        """
        Calculate reward based on multiple objectives
        
        Primary objective: Damp inter-area oscillations (tie-line flow stability)
        Secondary: Maintain voltage and frequency within limits
        """
        reward = 0.0
        
        # 1. Tie-line flow stability (damping oscillations)
        current_tie_flow = self._get_tie_line_flow()
        tie_flow_change = abs(current_tie_flow - self.previous_tie_flow)
        # Penalize large swings (oscillations)
        oscillation_penalty = -self.w_damping * tie_flow_change / 100.0
        reward += oscillation_penalty
        
        # Reward for being near nominal tie flow (~400 MW)
        nominal_tie_flow = 400.0
        tie_flow_error = abs(current_tie_flow - nominal_tie_flow)
        reward -= 0.1 * (tie_flow_error / 100.0)
        
        # 2. Voltage stability
        voltages = self._get_voltages()
        voltage_deviations = np.abs(voltages - 1.0)
        voltage_penalty = -self.w_voltage * np.mean(voltage_deviations)
        reward += voltage_penalty
        
        # 3. Voltage limit violations (hard penalty)
        voltage_violations = np.sum(
            (voltages < self.config.v_min_pu) | (voltages > self.config.v_max_pu)
        )
        reward -= 20.0 * voltage_violations
        
        # 4. Generation cost (minimize adjustments from nominal)
        gen_deviations = 0.0
        for gen in self.kundur.net.gen.itertuples():
            nominal = self.nominal_gen_power[gen.name]
            deviation = abs(gen.p_mw - nominal)
            gen_deviations += deviation
        reward -= 0.01 * gen_deviations
        
        # 5. Battery health (avoid excessive cycling)
        if self.enable_der and self.kundur.der_manager:
            for soc in self.kundur.der_manager.battery_soc.values():
                # Penalize extreme SOC
                if soc < 0.2 or soc > 0.8:
                    reward -= 1.0
                    
        # Scale reward
        reward *= self.reward_scale
        
        return float(reward)
        
    def _check_termination(self, result) -> bool:
        """Check if episode should terminate due to constraint violations"""
        # Voltage violations
        if result.num_voltage_violations > 2:
            logger.info("Episode terminated: voltage violations")
            return True
            
        # Line overloads
        if result.num_line_overloads > 0:
            logger.info("Episode terminated: line overloads")
            return True
            
        return False
        
    def _get_tie_line_flow(self) -> float:
        """Get total tie-line active power flow"""
        tie_flows = self.kundur.get_tie_line_flow()
        total_flow = sum(flow['p_from_mw'] for flow in tie_flows.values())
        return total_flow
        
    def _get_voltages(self) -> np.ndarray:
        """Get all bus voltages"""
        return self.kundur.net.res_bus['vm_pu'].values
        
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information for logging/debugging"""
        state = self.kundur.get_state()
        return {
            "step": self.current_step,
            "episode_reward": self.episode_reward,
            "total_generation_mw": state["system"]["total_generation_mw"],
            "total_load_mw": state["system"]["total_load_mw"],
            "tie_line_flow_mw": state["system"]["tie_line_flow_mw"],
            "converged": True,
        }
        
    def render(self) -> None:
        """Render environment state (console output)"""
        if self.state is None:
            return
            
        state = self.kundur.get_state()
        print(f"\n=== Kundur Environment (Step {self.current_step}) ===")
        print(f"Reward: {self.episode_reward:.2f}")
        print(f"Tie-line flow: {state['system']['tie_line_flow_mw']:.2f} MW")
        print(f"Area 1: Gen={state['areas']['Area1']['generation_mw']:.1f} MW, "
              f"Load={state['areas']['Area1']['load_mw']:.1f} MW")
        print(f"Area 2: Gen={state['areas']['Area2']['generation_mw']:.1f} MW, "
              f"Load={state['areas']['Area2']['load_mw']:.1f} MW")
        
        if self.enable_der:
            print(f"DER units: {len(self.kundur.der_manager.der_specs)}")
            
    def close(self) -> None:
        """Clean up resources"""
        pass
