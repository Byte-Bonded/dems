"""
Observation builder implementations for each agent hierarchy level.

Observation Design Principles:
- All values clipped to [-2, 2] to prevent exploding gradients
- Normalised to roughly unit scale
- Per-area builders see ONLY their own area (decentralised execution)
- Central builder sees inter-area state + aggregated summaries
- Sub-agent builders see narrow, task-specific features
"""

import numpy as np
from typing import Dict, List, Optional, Any

from src.simulation.supergrid import AreaID

# ─── Dimension constants ─────────────────────────────────────────────

# Microgrid-level observation (per area)
#   9 area power  + 3 area voltage + 4 DER + 3 dynamics + 3 env + 2 time = 24
MG_OBS_DIM = 24

# Central coordination observation
#   3×4 area summaries + 6 tie-line + 4 global + 2 time = 24
CENTRAL_OBS_DIM = 24

# Sub-agent observations
INVERTER_SUB_OBS_DIM = 10
RENEWABLE_SUB_OBS_DIM = 10
LOAD_SUB_OBS_DIM = 10

# Clipping range
_CLIP = 2.0


class MicrogridObsBuilder:
    """
    Build observation vector for a single microgrid (area) PPO agent.

    Layout (24-D):
        [0]  total_generation_mw / 5000
        [1]  total_load_mw / 5000
        [2]  net_interchange_mw / 500
        [3]  gen_headroom_mw / 1000       (Pmax - P_current, aggregated)
        [4]  gen_reserve_down_mw / 1000   (P_current - Pmin, aggregated)
        [5]  loss_mw / 200
        [6]  num_gens_online / 10
        [7]  avg_gen_loading_frac          (P/Pmax averaged)
        [8]  max_line_loading_pct / 100
        [9]  avg_voltage_pu
        [10] min_voltage_pu
        [11] max_voltage_pu
        [12] solar_output_mw / 100
        [13] wind_output_mw / 100
        [14] ev_load_mw / 50
        [15] dr_curtailment_frac
        [16] frequency_hz / 50
        [17] freq_deviation (clipped ±1)
        [18] avg_rotor_speed_pu
        [19] solar_irradiance / 1000
        [20] wind_speed / 25
        [21] load_scale_factor
        [22] hour_of_day / 24
        [23] step / episode_length
    """

    OBS_DIM = MG_OBS_DIM

    def __init__(self, area_id: AreaID, episode_length: int = 288):
        self.area_id = area_id
        self.episode_length = max(episode_length, 1)

    def build(
        self,
        area_state: Dict[str, Any],
        der_status: Dict[str, Any],
        gen_states: Dict[str, Any],
        frequency_hz: float,
        profiles: Any,  # StochasticProfileGenerator
        step: int,
        area_generators: Optional[List[int]] = None,
        net: Optional[Any] = None,  # pandapower net
    ) -> np.ndarray:
        obs = np.zeros(self.OBS_DIM, dtype=np.float32)

        # Area power metrics
        obs[0] = area_state.get("total_generation_mw", 0) / 5000.0
        obs[1] = area_state.get("total_load_mw", 0) / 5000.0
        obs[2] = area_state.get("net_interchange_mw", 0) / 500.0

        # Generator headroom/reserve (from pandapower net)
        headroom = 0.0
        reserve_down = 0.0
        n_gens = 0
        avg_loading = 0.0
        if net is not None and area_generators:
            for gi in area_generators:
                if gi in net.gen.index:
                    p = float(net.res_gen.at[gi, "p_mw"]) if gi in net.res_gen.index else 0.0
                    pmax = float(net.gen.at[gi, "max_p_mw"])
                    pmin = float(net.gen.at[gi, "min_p_mw"])
                    headroom += max(pmax - p, 0)
                    reserve_down += max(p - pmin, 0)
                    avg_loading += p / max(pmax, 1e-3)
                    n_gens += 1
        obs[3] = headroom / 1000.0
        obs[4] = reserve_down / 1000.0
        obs[5] = area_state.get("total_losses_mw", 0) / 200.0
        obs[6] = n_gens / 10.0
        obs[7] = avg_loading / max(n_gens, 1)

        # Line loading
        obs[8] = area_state.get("max_line_loading_pct", 0) / 100.0

        # Voltages
        obs[9] = area_state.get("avg_voltage_pu", 1.0)
        obs[10] = area_state.get("min_voltage_pu", 1.0)
        obs[11] = area_state.get("max_voltage_pu", 1.0)

        # DER status
        obs[12] = der_status.get("solar", {}).get("current_output_mw", 0) / 100.0
        obs[13] = der_status.get("wind", {}).get("current_output_mw", 0) / 100.0
        obs[14] = der_status.get("ev_charging", {}).get("current_load_mw", 0) / 50.0
        obs[15] = der_status.get("demand_response", {}).get("avg_curtailment_frac", 0)

        # Dynamics
        obs[16] = frequency_hz / 50.0
        obs[17] = np.clip(frequency_hz - 50.0, -1.0, 1.0)
        # Average rotor speed for this area's generators
        if gen_states:
            rotor_speeds = []
            for gid, gs in gen_states.items():
                if isinstance(gs, dict):
                    rotor_speeds.append(gs.get("omega_pu", 1.0))
            obs[18] = np.mean(rotor_speeds) if rotor_speeds else 1.0
        else:
            obs[18] = 1.0

        # Environment
        obs[19] = profiles.solar_irradiance(step) / 1000.0
        obs[20] = profiles.wind_speed(step) / 25.0
        obs[21] = profiles.load_scale_factor(step)

        # Time
        obs[22] = profiles.hour_of_day(step) / 24.0
        obs[23] = step / self.episode_length

        return np.clip(np.nan_to_num(obs, nan=0.0, posinf=_CLIP, neginf=-_CLIP), -_CLIP, _CLIP)


class CentralObsBuilder:
    """
    Build observation vector for the central coordination PPO agent.

    Layout (24-D):
        [0..3]   Area A summary: gen_mw/5000, load_mw/5000, net_interchange/500, freq_dev
        [4..7]   Area B summary  (same layout)
        [8..11]  Area C summary  (same layout)
        [12..17] Tie-line flows: 6 tie-lines × flow_mw/500
        [18]     system_frequency / 50
        [19]     total_gen / 15000
        [20]     total_load / 15000
        [21]     total_losses / 1000
        [22]     hour_of_day / 24
        [23]     step / episode_length
    """

    OBS_DIM = CENTRAL_OBS_DIM

    def __init__(self, episode_length: int = 288):
        self.episode_length = max(episode_length, 1)

    def build(
        self,
        global_state: Dict[str, Any],
        tie_line_flows: List[Dict[str, Any]],
        frequency_hz: float,
        profiles: Any,
        step: int,
    ) -> np.ndarray:
        obs = np.zeros(self.OBS_DIM, dtype=np.float32)
        areas = global_state.get("areas", {})

        # Per-area summaries
        for i, area_key in enumerate(["A", "B", "C"]):
            a = areas.get(area_key, {})
            base = i * 4
            obs[base + 0] = a.get("total_generation_mw", 0) / 5000.0
            obs[base + 1] = a.get("total_load_mw", 0) / 5000.0
            obs[base + 2] = a.get("net_interchange_mw", 0) / 500.0
            # Per-area frequency deviation proxy: use global for now
            obs[base + 3] = np.clip(frequency_hz - 50.0, -1.0, 1.0)

        # Tie-line flows (up to 6)
        for j, tl in enumerate(tie_line_flows[:6]):
            obs[12 + j] = tl.get("flow_mw", 0) / 500.0

        # Global metrics
        gm = global_state.get("global_metrics", {})
        obs[18] = frequency_hz / 50.0
        obs[19] = gm.get("total_generation_mw", 0) / 15000.0
        obs[20] = gm.get("total_load_mw", 0) / 15000.0
        obs[21] = gm.get("total_losses_mw", 0) / 1000.0

        # Time
        obs[22] = profiles.hour_of_day(step) / 24.0
        obs[23] = step / self.episode_length

        return np.clip(np.nan_to_num(obs, nan=0.0, posinf=_CLIP, neginf=-_CLIP), -_CLIP, _CLIP)


class SubAgentObsBuilder:
    """
    Build narrow observations for sub-agents (inverter, renewable, load).

    All sub-agents share the same builder; the observation content differs
    based on the sub-agent role.

    Each sub-obs is 10-D:
    - Inverter: [v_local, f_dev, p_gen, q_gen, p_max, loading_frac, v_min_area, v_max_area, hour, step]
    - Renewable: [irradiance_or_ws, p_output, p_rated, curtail_frac, v_local, f_dev, hour, step, 0, 0]
    - Load: [total_load, ev_load, dr_curtail, v_local, f_dev, load_scale, hour, step, 0, 0]
    """

    INVERTER_DIM = INVERTER_SUB_OBS_DIM
    RENEWABLE_DIM = RENEWABLE_SUB_OBS_DIM
    LOAD_DIM = LOAD_SUB_OBS_DIM

    def __init__(self, episode_length: int = 288):
        self.episode_length = max(episode_length, 1)

    def build_inverter(
        self,
        gen_idx: int,
        net: Any,
        frequency_hz: float,
        area_state: Dict,
        step: int,
        profiles: Any,
    ) -> np.ndarray:
        obs = np.zeros(INVERTER_SUB_OBS_DIM, dtype=np.float32)
        if gen_idx in net.res_gen.index:
            bus = int(net.gen.at[gen_idx, "bus"])
            obs[0] = float(net.res_bus.at[bus, "vm_pu"]) if bus in net.res_bus.index else 1.0
            obs[1] = np.clip(frequency_hz - 50.0, -1.0, 1.0)
            obs[2] = float(net.res_gen.at[gen_idx, "p_mw"]) / 500.0
            obs[3] = float(net.res_gen.at[gen_idx, "q_mvar"]) / 200.0
            obs[4] = float(net.gen.at[gen_idx, "max_p_mw"]) / 500.0
            pmax = float(net.gen.at[gen_idx, "max_p_mw"])
            p_cur = float(net.res_gen.at[gen_idx, "p_mw"])
            obs[5] = p_cur / max(pmax, 1e-3)
        obs[6] = area_state.get("min_voltage_pu", 1.0)
        obs[7] = area_state.get("max_voltage_pu", 1.0)
        obs[8] = profiles.hour_of_day(step) / 24.0
        obs[9] = step / self.episode_length
        return np.clip(np.nan_to_num(obs, nan=0.0, posinf=_CLIP, neginf=-_CLIP), -_CLIP, _CLIP)

    def build_renewable(
        self,
        resource_value: float,  # irradiance (W/m²) or wind_speed (m/s)
        resource_max: float,    # 1000 for solar, 25 for wind
        current_output_mw: float,
        rated_mw: float,
        curtail_frac: float,
        v_local_pu: float,
        frequency_hz: float,
        step: int,
        profiles: Any,
    ) -> np.ndarray:
        obs = np.zeros(RENEWABLE_SUB_OBS_DIM, dtype=np.float32)
        obs[0] = resource_value / max(resource_max, 1.0)
        obs[1] = current_output_mw / max(rated_mw, 1e-3)
        obs[2] = rated_mw / 100.0
        obs[3] = curtail_frac
        obs[4] = v_local_pu
        obs[5] = np.clip(frequency_hz - 50.0, -1.0, 1.0)
        obs[6] = profiles.hour_of_day(step) / 24.0
        obs[7] = step / self.episode_length
        return np.clip(np.nan_to_num(obs, nan=0.0, posinf=_CLIP, neginf=-_CLIP), -_CLIP, _CLIP)

    def build_load(
        self,
        total_load_mw: float,
        ev_load_mw: float,
        dr_curtail_frac: float,
        v_local_pu: float,
        frequency_hz: float,
        load_scale: float,
        step: int,
        profiles: Any,
    ) -> np.ndarray:
        obs = np.zeros(LOAD_SUB_OBS_DIM, dtype=np.float32)
        obs[0] = total_load_mw / 5000.0
        obs[1] = ev_load_mw / 50.0
        obs[2] = dr_curtail_frac
        obs[3] = v_local_pu
        obs[4] = np.clip(frequency_hz - 50.0, -1.0, 1.0)
        obs[5] = load_scale
        obs[6] = profiles.hour_of_day(step) / 24.0
        obs[7] = step / self.episode_length
        return np.clip(np.nan_to_num(obs, nan=0.0, posinf=_CLIP, neginf=-_CLIP), -_CLIP, _CLIP)
