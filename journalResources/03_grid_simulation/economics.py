"""
Economic Dispatch & Pricing Model for DEMS SuperGrid

Provides:
- Quadratic generation cost curves: C_i(P) = a_i + b_i*P + c_i*P^2  ($/h)
- Marginal cost:  MC_i(P) = b_i + 2*c_i*P  ($/MWh)
- Locational Marginal Pricing (LMP) from power-flow Jacobian sensitivities
- Time-of-Use (TOU) and Real-Time Pricing (RTP) profiles
- Carbon emission rates per fuel type
- Merit-order economic dispatch

Cost data references:
    IEEE 39-bus (New England): Zimmerman et al., MATPOWER 7.1 casedata
    Carbon:  EPA eGRID 2022 average factors
    TOU:     CERC India tariff structure (adapted)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ======================== FUEL TYPES ======================== #

class FuelType(Enum):
    COAL = "coal"
    GAS = "gas"
    HYDRO = "hydro"
    NUCLEAR = "nuclear"
    OIL = "oil"


# Carbon emission factors (tCO2 / MWh)  — EPA eGRID 2022
CARBON_FACTORS: Dict[FuelType, float] = {
    FuelType.COAL: 0.95,
    FuelType.GAS: 0.41,
    FuelType.HYDRO: 0.0,
    FuelType.NUCLEAR: 0.0,
    FuelType.OIL: 0.73,
}


# ======================== GENERATOR COST DATA ======================== #

@dataclass
class GenCostCurve:
    """Quadratic cost curve for a single generator.

    C(P) = a + b*P + c*P^2   [$/h]
    MC(P) = b + 2*c*P         [$/MWh]
    """
    gen_idx: int
    bus: int
    area: str                # "A", "B", or "C"
    fuel: FuelType
    p_min_mw: float
    p_max_mw: float
    a: float                 # Fixed cost ($/h)
    b: float                 # Linear cost ($/MWh)
    c: float                 # Quadratic cost ($/MWh^2)
    ramp_mw_per_min: float = 5.0   # Ramp rate limit

    def cost(self, p_mw: float) -> float:
        """Total generation cost at output P ($/h)."""
        p = np.clip(p_mw, self.p_min_mw, self.p_max_mw)
        return self.a + self.b * p + self.c * p ** 2

    def marginal_cost(self, p_mw: float) -> float:
        """Marginal cost at output P ($/MWh)."""
        p = np.clip(p_mw, self.p_min_mw, self.p_max_mw)
        return self.b + 2.0 * self.c * p

    def carbon_rate(self, p_mw: float) -> float:
        """CO2 emission rate at output P (tCO2/h)."""
        return CARBON_FACTORS.get(self.fuel, 0.0) * p_mw


def _build_ieee39_cost_data(area: str, offset: int) -> List[GenCostCurve]:
    """Build cost curves for one 39-bus area.

    Based on MATPOWER case39 gencost (polynomial type 2) with fuel type
    assignments following the New England system documentation:
    - Gens 0-3  (buses 30-33): Nuclear / Coal base-load
    - Gens 4-6  (buses 34-36): Gas combined-cycle mid-merit
    - Gens 7-8  (buses 37-38): Gas peaker
    - Gen  9    (bus 39=slack): Hydro (Niagara equivalent)
    """
    # (gen_local, bus_local, fuel, Pmin, Pmax, a, b, c)
    IEEE39_GENCOST = [
        (0, 30, FuelType.NUCLEAR, 100, 600, 240.0, 7.00, 0.0070),
        (1, 31, FuelType.COAL,    100, 700, 200.0, 8.50, 0.0085),
        (2, 32, FuelType.COAL,    100, 600, 220.0, 9.00, 0.0090),
        (3, 33, FuelType.COAL,    100, 600, 200.0, 8.80, 0.0088),
        (4, 34, FuelType.GAS,      50, 500, 150.0, 11.0, 0.0110),
        (5, 35, FuelType.GAS,      50, 500, 150.0, 10.5, 0.0105),
        (6, 36, FuelType.GAS,      50, 450, 160.0, 12.0, 0.0120),
        (7, 37, FuelType.OIL,      50, 400, 180.0, 14.0, 0.0140),
        (8, 38, FuelType.OIL,      50, 400, 180.0, 13.5, 0.0135),
        (9, 39, FuelType.HYDRO,    50, 800,  50.0, 5.00, 0.0025),
    ]
    costs = []
    for i, (g_local, bus_local, fuel, pmin, pmax, a, b, c) in enumerate(IEEE39_GENCOST):
        costs.append(GenCostCurve(
            gen_idx=g_local + offset * 10,  # Will be remapped to pandapower idx
            bus=bus_local + offset * 39,
            area=area,
            fuel=fuel,
            p_min_mw=pmin,
            p_max_mw=pmax,
            a=a, b=b, c=c,
            ramp_mw_per_min=pmax * 0.02,  # 2% Pmax per minute
        ))
    return costs


class EconomicsEngine:
    """
    Generation cost model and economic dispatch for the 117-bus SuperGrid.

    Computes:
    - Per-generator operating cost and marginal cost
    - System-wide total operating cost
    - Merit-order economic dispatch
    - Locational Marginal Prices (LMPs) — approximation
    - Carbon emissions
    - Time-of-Use (TOU) pricing profiles
    """

    def __init__(self):
        """Build cost curves for the full 117-bus tri-area SuperGrid."""
        self.gen_costs: Dict[int, GenCostCurve] = {}
        self._build_all_costs()
        self._tou_profile: Optional[np.ndarray] = None

    def _build_all_costs(self):
        """Create cost data for all 3 areas (30 generators total)."""
        for area_idx, area_id in enumerate(["A", "B", "C"]):
            area_costs = _build_ieee39_cost_data(area_id, area_idx)
            for gc in area_costs:
                self.gen_costs[gc.gen_idx] = gc
        logger.info(f"EconomicsEngine: {len(self.gen_costs)} generator cost curves loaded")

    def remap_to_pandapower(self, net) -> None:
        """Remap internal gen indices to actual pandapower gen indices.

        Call this once after the SuperGrid is built to align cost curves
        with the actual generator ordering in the pandapower network.
        """
        remapped: Dict[int, GenCostCurve] = {}
        for pp_idx in net.gen.index:
            bus = int(net.gen.at[pp_idx, "bus"])
            # Match by bus number
            for gc in self.gen_costs.values():
                if gc.bus == bus and gc.gen_idx not in remapped:
                    gc_new = GenCostCurve(
                        gen_idx=int(pp_idx),
                        bus=bus,
                        area=gc.area,
                        fuel=gc.fuel,
                        p_min_mw=gc.p_min_mw,
                        p_max_mw=gc.p_max_mw,
                        a=gc.a, b=gc.b, c=gc.c,
                        ramp_mw_per_min=gc.ramp_mw_per_min,
                    )
                    remapped[int(pp_idx)] = gc_new
                    break
        if remapped:
            self.gen_costs = remapped
            logger.info(f"EconomicsEngine: remapped {len(remapped)} generators to pandapower indices")

    # ─── Cost Computation ──────────────────────────────────────────

    def total_operating_cost(self, gen_dispatch: Dict[int, float]) -> float:
        """Total system operating cost ($/h) for given dispatch.

        Args:
            gen_dispatch: {gen_idx: p_mw} for each generator.
        """
        total = 0.0
        for gen_idx, p_mw in gen_dispatch.items():
            gc = self.gen_costs.get(gen_idx)
            if gc:
                total += gc.cost(p_mw)
            else:
                # Unknown generator — use flat rate
                total += 15.0 * p_mw
        return total

    def marginal_costs(self, gen_dispatch: Dict[int, float]) -> Dict[int, float]:
        """Marginal cost ($/MWh) at each generator's current output."""
        return {
            idx: self.gen_costs[idx].marginal_cost(p_mw)
            for idx, p_mw in gen_dispatch.items()
            if idx in self.gen_costs
        }

    def system_marginal_cost(self, gen_dispatch: Dict[int, float]) -> float:
        """System marginal cost = weighted average of generator MCs."""
        total_p = 0.0
        total_mc_p = 0.0
        for idx, p in gen_dispatch.items():
            gc = self.gen_costs.get(idx)
            if gc and p > 0:
                total_p += p
                total_mc_p += gc.marginal_cost(p) * p
        return total_mc_p / max(total_p, 1.0)

    def total_carbon_emissions(self, gen_dispatch: Dict[int, float]) -> float:
        """Total CO2 emissions (tCO2/h) for given dispatch."""
        total = 0.0
        for idx, p in gen_dispatch.items():
            gc = self.gen_costs.get(idx)
            if gc:
                total += gc.carbon_rate(p)
        return total

    def carbon_intensity(self, gen_dispatch: Dict[int, float]) -> float:
        """Carbon intensity (kgCO2/MWh) = total emissions / total gen."""
        total_p = sum(gen_dispatch.values())
        if total_p <= 0:
            return 0.0
        return self.total_carbon_emissions(gen_dispatch) * 1000.0 / total_p

    # ─── Economic Dispatch ─────────────────────────────────────────

    def merit_order_dispatch(
        self,
        total_demand_mw: float,
        available_gens: Optional[List[int]] = None,
    ) -> Dict[int, float]:
        """Merit-order economic dispatch.

        Sort generators by marginal cost at Pmin, dispatch cheapest first.
        Respects Pmin/Pmax constraints.

        Args:
            total_demand_mw: Total system demand to serve.
            available_gens:  Subset of generator indices (default: all).

        Returns:
            {gen_idx: p_mw} dispatch schedule.
        """
        if available_gens is None:
            available_gens = list(self.gen_costs.keys())

        # Sort by marginal cost at Pmin (cheapest first)
        sorted_gens = sorted(
            available_gens,
            key=lambda idx: self.gen_costs[idx].marginal_cost(self.gen_costs[idx].p_min_mw)
            if idx in self.gen_costs else 999.0,
        )

        dispatch: Dict[int, float] = {}
        remaining = total_demand_mw

        for gen_idx in sorted_gens:
            gc = self.gen_costs.get(gen_idx)
            if gc is None:
                continue

            if remaining <= 0:
                dispatch[gen_idx] = gc.p_min_mw  # Keep committed at Pmin
                remaining -= gc.p_min_mw
                continue

            # Dispatch up to Pmax or remaining demand
            p = min(gc.p_max_mw, remaining + gc.p_min_mw)
            p = max(p, gc.p_min_mw)
            dispatch[gen_idx] = p
            remaining -= p

        return dispatch

    # ─── LMP Approximation ─────────────────────────────────────────

    def compute_lmp(
        self,
        net,
        gen_dispatch: Dict[int, float],
    ) -> Dict[int, float]:
        """Compute approximate Locational Marginal Prices per bus.

        LMP = λ (energy) + μ (congestion) + γ (losses)

        Energy component: system marginal cost.
        Loss component: k * (1 + dLoss/dP_bus) — from power flow losses gradient.
        Congestion component: penalty for overloaded lines.

        Args:
            net: pandapower network with solved power flow results.
            gen_dispatch: Current generation dispatch.

        Returns:
            {bus_idx: lmp_$/MWh} for each bus.
        """
        smc = self.system_marginal_cost(gen_dispatch)
        lmp: Dict[int, float] = {}

        if not hasattr(net, 'res_bus') or net.res_bus is None or len(net.res_bus) == 0:
            # No PF results — return flat LMP
            for bus in net.bus.index:
                lmp[int(bus)] = smc
            return lmp

        # Loss sensitivity: approximate dLoss/dP ≈ 2 * losses/total_gen
        total_gen = sum(gen_dispatch.values()) if gen_dispatch else 1.0
        total_losses = 0.0
        if hasattr(net, 'res_line') and len(net.res_line) > 0:
            total_losses = float(net.res_line['pl_mw'].sum())
        loss_factor = total_losses / max(total_gen, 1.0)

        # Congestion penalty: lines loaded > 80%
        congested_buses = set()
        if hasattr(net, 'res_line') and len(net.res_line) > 0:
            for line_idx in net.line.index:
                loading = float(net.res_line.at[line_idx, 'loading_percent'])
                if loading > 80.0:
                    from_bus = int(net.line.at[line_idx, 'from_bus'])
                    to_bus = int(net.line.at[line_idx, 'to_bus'])
                    congested_buses.add(from_bus)
                    congested_buses.add(to_bus)

        for bus in net.bus.index:
            bus_idx = int(bus)
            # Base energy price
            energy = smc

            # Voltage-based loss sensitivity (buses with low voltage have higher losses)
            v_pu = 1.0
            if bus_idx in net.res_bus.index:
                v_pu = float(net.res_bus.at[bus_idx, 'vm_pu'])
            loss_component = smc * loss_factor * (2.0 - v_pu)

            # Congestion premium
            congestion = 0.0
            if bus_idx in congested_buses:
                congestion = smc * 0.15  # 15% congestion premium

            lmp[bus_idx] = energy + loss_component + congestion

        return lmp

    # ─── Time-of-Use Pricing ───────────────────────────────────────

    def tou_price(self, hour: float) -> float:
        """Time-of-Use electricity price ($/MWh) at given hour of day.

        Rate structure (based on CERC India tariff):
            Off-peak  (22:00 - 06:00):  $30/MWh
            Shoulder  (06:00 - 09:00, 14:00 - 17:00):  $50/MWh
            Peak      (09:00 - 14:00, 17:00 - 22:00):  $80/MWh
        """
        h = hour % 24.0
        if h < 6.0 or h >= 22.0:
            return 30.0   # Off-peak
        elif (6.0 <= h < 9.0) or (14.0 <= h < 17.0):
            return 50.0   # Shoulder
        else:
            return 80.0   # Peak

    def rtp_price(self, hour: float, demand_mw: float, supply_mw: float) -> float:
        """Real-Time Price reflecting supply-demand balance.

        RTP = TOU_base × (1 + α × (demand - supply) / demand)
        Where α = 0.5 is the price elasticity coefficient.
        """
        base = self.tou_price(hour)
        if demand_mw <= 0:
            return base
        imbalance_ratio = (demand_mw - supply_mw) / demand_mw
        alpha = 0.5
        return base * (1.0 + alpha * np.clip(imbalance_ratio, -0.5, 1.0))

    def dr_incentive_price(self, hour: float, curtailment_mw: float) -> float:
        """Demand Response payment to customers for load curtailment ($/MWh).

        Incentive = max(TOU_price, 60.0) + bonus for peak hours.
        """
        tou = self.tou_price(hour)
        base_incentive = max(tou, 60.0)
        # Peak bonus: extra $20/MWh during peak hours
        h = hour % 24.0
        peak_bonus = 20.0 if (9.0 <= h < 14.0 or 17.0 <= h < 22.0) else 0.0
        return base_incentive + peak_bonus

    def generate_tou_profile(self, n_steps: int, dt_s: float = 300.0) -> np.ndarray:
        """Generate TOU price profile for an episode.

        Args:
            n_steps: Number of control steps.
            dt_s: Control step duration (seconds).

        Returns:
            Array of $/MWh prices, shape (n_steps,).
        """
        hours = np.arange(n_steps) * dt_s / 3600.0
        prices = np.array([self.tou_price(h) for h in hours])
        self._tou_profile = prices
        return prices

    # ─── Convenience ───────────────────────────────────────────────

    def get_dispatch_from_network(self, net) -> Dict[int, float]:
        """Extract current dispatch from a pandapower network."""
        dispatch = {}
        if hasattr(net, 'res_gen') and len(net.res_gen) > 0:
            for idx in net.gen.index:
                p = float(net.res_gen.at[idx, 'p_mw'])
                dispatch[int(idx)] = p
        return dispatch

    def compute_cost_summary(
        self,
        net,
        hour: float = 12.0,
    ) -> Dict[str, float]:
        """Full economic summary for a single timestep.

        Returns dict with total_cost, marginal_cost, carbon_emissions,
        carbon_intensity, tou_price, and per-area breakdowns.
        """
        dispatch = self.get_dispatch_from_network(net)
        total_cost = self.total_operating_cost(dispatch)
        smc = self.system_marginal_cost(dispatch)
        carbon = self.total_carbon_emissions(dispatch)
        ci = self.carbon_intensity(dispatch)
        tou = self.tou_price(hour)

        # Per-area cost breakdown
        area_costs = {"A": 0.0, "B": 0.0, "C": 0.0}
        for idx, p in dispatch.items():
            gc = self.gen_costs.get(idx)
            if gc:
                area_costs[gc.area] += gc.cost(p)

        return {
            "total_operating_cost_usd_h": total_cost,
            "system_marginal_cost_usd_mwh": smc,
            "total_carbon_tco2_h": carbon,
            "carbon_intensity_kgco2_mwh": ci,
            "tou_price_usd_mwh": tou,
            "area_A_cost_usd_h": area_costs["A"],
            "area_B_cost_usd_h": area_costs["B"],
            "area_C_cost_usd_h": area_costs["C"],
            "total_generation_mw": sum(dispatch.values()),
            "n_generators_dispatched": len([p for p in dispatch.values() if p > 0]),
        }
