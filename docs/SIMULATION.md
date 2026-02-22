# Simulation Engine — Technical Reference

> Comprehensive documentation for the `src/simulation/` package of the
> Dynamic Energy Management System (DEMS).

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Module Dependency Graph](#2-module-dependency-graph)
3. [SuperGrid — Tri-Area 117-Bus Network](#3-supergrid--tri-area-117-bus-network)
4. [Power Flow Engine](#4-power-flow-engine)
5. [Electromechanical Dynamics](#5-electromechanical-dynamics)
6. [Distributed Energy Resources (DER)](#6-distributed-energy-resources-der)
7. [Grid Orchestrator — RL Interface](#7-grid-orchestrator--rl-interface)
8. [Microgrid Standalone Simulator](#8-microgrid-standalone-simulator)
9. [Tie-Line Management](#9-tie-line-management)
10. [IEEE Standard Compliance](#10-ieee-standard-compliance)
11. [Numerical Methods](#11-numerical-methods)
12. [Data Flow & Step-by-Step Pipeline](#12-data-flow--step-by-step-pipeline)
13. [Configuration Reference](#13-configuration-reference)
14. [API Quick Reference](#14-api-quick-reference)
15. [Testing](#15-testing)

---

## 1. Architecture Overview

The simulation engine models a **117-bus tri-area power grid** derived from
three merged IEEE 39-bus (New England) systems. It couples:

- **Steady-state AC power flow** (Newton-Raphson via pandapower)
- **Electromechanical dynamics** (swing equation, AVR, governor, PSS, AGC)
- **Distributed Energy Resources** (solar, wind, battery, EV, demand response)
- **Stochastic environment profiles** (Ornstein-Uhlenbeck wind/cloud, diurnal load)
- **Reinforcement Learning interface** (Gymnasium-compatible observation/action/reward)

All modules reside under `src/simulation/` and are designed for **50 Hz
Indian Grid Code (IEGC)** operation.

### Layer Stack

```
┌──────────────────────────────────────────────────────────┐
│                   RL Agent (PPO / SAC)                   │
│              stable-baselines3 / gymnasium               │
└────────────────────────┬─────────────────────────────────┘
                         │  action ∈ [0,1]^47
                         │  obs ∈ ℝ^42 , reward ∈ ℝ
┌────────────────────────▼─────────────────────────────────┐
│              GridOrchestrator (orchestrator.py)           │
│  • ActionMapper  • ObservationBuilder  • RewardCalculator│
│  • StochasticProfileGenerator                            │
└──┬──────────┬──────────┬──────────┬──────────┬───────────┘
   │          │          │          │          │
   ▼          ▼          ▼          ▼          ▼
SuperGrid  PowerFlow  Dynamics   DER       TieLines
(supergrid (power_   (dynamics  (der.py)  (tie_lines
  .py)      flow.py)   .py)               .py)
   │                     │
   └──── pandapower ─────┘
         (case39)
```

---

## 2. Module Dependency Graph

| Module | Lines | Depends On | Provides |
|--------|------:|------------|----------|
| `supergrid.py` | 1038 | pandapower, der, dynamics | 117-bus net, area management, state extraction |
| `dynamics.py` | 734 | numpy | Generator, exciter, governor, PSS, AGC, protection |
| `der.py` | 1077 | pandapower, numpy, threading | DER registration, dispatch, SOC, LVRT, anti-island |
| `orchestrator.py` | 950 | supergrid, power_flow, der, dynamics | RL interface: obs, action, reward, step, reset |
| `power_flow.py` | 439 | pandapower | NR/FDXB/DC solver, N-1 contingency, time-series |
| `microgrid.py` | 704 | numpy, (scipy.sparse) | Standalone NR solver, DER components, 24h simulator |
| `tie_lines.py` | 297 | pandapower, numpy | Tie-line config, thermal monitoring, flow analysis |

**Total: ~5,239 lines of simulation code.**

---

## 3. SuperGrid — Tri-Area 117-Bus Network

**File:** `src/simulation/supergrid.py` (1,038 lines)

### 3.1 Construction

The `SuperGrid` class merges three copies of IEEE 39-bus (New England)
into a single pandapower network:

```python
from src.simulation.supergrid import SuperGrid

sg = SuperGrid()            # builds 117-bus net automatically
sg.initialize_der()         # adds 18 DER units across 3 areas
sg.initialize_dynamics()    # creates 30 dynamic generator models
```

**Step-by-step build process:**

1. **Create 3 networks:** `pn.case39()` → `net_a`, `net_b`, `net_c`
2. **Tag zones:** `net_a.bus["zone"]=1`, etc.
3. **Merge sequentially:** `merge_nets(net_a, net_b)` then merge with `net_c`
4. **Update area mappings:** Bus ranges, generator/load index lists
5. **Add 8 tie-lines:** Inter-area HVAC connections (N-1 secure mesh)
6. **Configure slack buses:** 3 ext_grids at buses 30, 69, 108
7. **Improve convergence:** Reactive compensation, voltage correction

### 3.2 Area Configuration

| Area | Offset | Buses | Ext-Grid Bus | vm_pu | Character |
|------|-------:|-------|--------------|------:|-----------|
| A (North) | 0 | 0–38 | 30 | 1.030 | Solar-dominated |
| B (West) | 39 | 39–77 | 69 | 0.982 | Wind-dominated |
| C (East) | 78 | 78–116 | 108 | 1.050 | Hybrid |

Each area has **10 generators** (buses 29–38 within the area, offset by
`bus_offset`), for 30 generators total. One generator per area is at the
ext_grid bus (bus 30+offset), acting as the area slack.

### 3.3 Tie-Line Topology (8 Lines, N-1 Secure)

```
         AREA A (North)
        /      |      \
   TL1 /   TL2|   TL3 \
      /        |        \
 AREA B ──TL4,TL5,TL6── AREA C
      \                  /
   TL7 \              / TL8
        (A↔C direct)
```

| Tie-Line | From | To | Length | Rating |
|----------|------|----|-------:|-------:|
| TL_AB_1 | A:bus 1 | B:bus 1 | 100 km | 700 MVA |
| TL_AB_2 | A:bus 2 | B:bus 2 | 120 km | 600 MVA |
| TL_AB_3 | A:bus 38 | B:bus 9 | 150 km | 500 MVA |
| TL_BC_1 | B:bus 3 | C:bus 3 | 110 km | 650 MVA |
| TL_BC_2 | B:bus 14 | C:bus 14 | 130 km | 550 MVA |
| TL_BC_3 | B:bus 26 | C:bus 26 | 140 km | 500 MVA |
| TL_AC_1 | A:bus 9 | C:bus 9 | 200 km | 600 MVA |
| TL_AC_2 | A:bus 14 | C:bus 1 | 220 km | 500 MVA |

Parameters: `r=0.02 Ω/km`, `x=0.25 Ω/km`, `c=12 nF/km`, 345 kV base.

### 3.4 Convergence & Voltage Correction

`_improve_convergence()` runs iteratively (up to 5 rounds):

1. **Relax generator Q limits** — ensures generators can absorb reactive power
2. **Set generator vm_pu = 1.00** (ext_grid values preserved per BUG-12/NEW-BUG-03)
3. **Run Newton-Raphson** power flow
4. **Add shunt compensation** at buses with V < 0.97 or V > 1.03 pu
5. **Repeat** until voltage profile 0.95–1.05 pu or no more corrections

### 3.5 ULTC Tap Changers

```python
tap_changes = sg.step_ultc_tap_changers(
    deadband_pu=0.005,          # ±0.5% deadband
    v_setpoint_pu=1.0,          # target 1.0 pu
    max_tap_step=1              # max 1 step per call
)
```

Deadband-based automatic tap adjustment on all in-service transformers.
Tap range: -10 to +10 steps. Call once per 30–60s of simulation time.

### 3.6 Key Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `initialize_der()` | `DERManager` | Adds 18 default DER units |
| `initialize_dynamics()` | `DynamicsCoordinator` | Creates 30 gen models + 3 AGC |
| `step_dynamics(dt)` | `Dict` | Advances dynamics one timestep |
| `get_global_state()` | `Dict` | Full grid state for RL agent |
| `get_area_state(area_id)` | `Dict` | Per-area aggregated metrics |
| `get_tie_line_flows()` | `List[Dict]` | Tie-line power/loading/overload |
| `set_generator_setpoint(idx, p_mw)` | `None` | Set gen active power (clamped) |
| `scale_loads(area_id, factor)` | `None` | Scale area loads by multiplier |
| `reset_to_base_case()` | `None` | Rebuild grid from scratch |

---

## 4. Power Flow Engine

**File:** `src/simulation/power_flow.py` (439 lines)

### 4.1 Algorithms

| Algorithm | Enum | Speed | Accuracy | Use Case |
|-----------|------|-------|----------|----------|
| Newton-Raphson | `NEWTON_RAPHSON` | Fast | High | Default, production |
| Fast Decoupled | `FAST_DECOUPLED` | Faster | Good | Large systems |
| Gauss-Seidel | `GAUSS_SEIDEL` | Slow | High | Educational |
| DC Approximation | `DC` | Very fast | Low | Screening |

### 4.2 Configuration

```python
from src.simulation.power_flow import PowerFlowRunner, PowerFlowConfig

config = PowerFlowConfig(
    algorithm=PowerFlowAlgorithm.NEWTON_RAPHSON,
    max_iterations=30,
    tolerance_mva=1e-8,
    enforce_q_limits=True,
    v_min_pu=0.95,              # configurable violation thresholds
    v_max_pu=1.05,
)
runner = PowerFlowRunner(config)
result = runner.run(sg.net, verbose=True)
```

### 4.3 Result Object (`PowerFlowResult`)

| Field | Type | Description |
|-------|------|-------------|
| `converged` | `bool` | Whether NR converged |
| `iterations` | `int` | NR iteration count |
| `elapsed_time_ms` | `float` | Wall-clock time |
| `total_generation_mw` | `float` | Gen + ext_grid output |
| `total_load_mw` | `float` | Active load total |
| `total_losses_mw` | `float` | I²R line + trafo losses |
| `min_voltage_pu` | `float` | Lowest bus voltage |
| `max_voltage_pu` | `float` | Highest bus voltage |
| `num_voltage_violations` | `int` | Buses outside limits |
| `num_line_overloads` | `int` | Lines > 100% loading |
| `is_secure` | `bool` | Converged ∧ no violations |

**Loss computation:** Uses actual `res_line.pl_mw + res_trafo.pl_mw` (I²R),
not gen–load difference (which includes shunt losses, rounding errors).

### 4.4 N-1 Contingency Analysis

```python
contingencies = [("line", 5), ("line", 12), ("gen", 3)]
results = runner.run_with_contingency(sg.net, contingencies)
# results[0] = base case, results[1..N] = each contingency
```

### 4.5 Time-Series Power Flow

```python
from src.simulation.power_flow import run_time_series_power_flow

results = run_time_series_power_flow(
    net=sg.net,
    load_profiles=np.ones((288, n_loads)),   # 288 × 5min = 24h
    gen_profiles=gen_schedule,                # (288, n_gens)
    timesteps=288,
)
```

---

## 5. Electromechanical Dynamics

**File:** `src/simulation/dynamics.py` (734 lines)

This module implements the full electromechanical simulation chain for
power system transient stability analysis.

### 5.1 Synchronous Generator (`SynchronousGeneratorDynamic`)

**Swing Equation** — governs rotor angle (δ) and speed (ω):

$$\frac{d\omega}{dt} = \frac{1}{2H} \left( P_m - P_e - D(\omega - 1) \right)$$

$$\frac{d\delta}{dt} = \omega_0 (\omega - 1)$$

where $\omega_0 = 2\pi f_0 = 314.16$ rad/s (50 Hz), $H$ = inertia constant (s),
$D$ = damping coefficient.

**Integration:** Symplectic Euler — update ω first, then use new ω for δ.
This preserves the Hamiltonian structure of the swing equation better than
standard forward Euler.

**Field Flux Equation** (when Efd provided from exciter):

$$\frac{dE_q'}{dt} = \frac{E_{fd} - E_q'}{T_{d0}'}$$

Integrated via implicit trapezoidal rule, coupling AVR output back to the
generator's internal EMF.

**Initialization from Power Flow:**

Uses phasor calculation:
$\vec{E}' = \vec{V}_t + jX_d' \vec{I}$, where
$\vec{I} = (P - jQ) / \vec{V}_t^*$. This correctly determines
$E_q' = |E'|$ and the initial rotor angle.

**IEEE 39-Bus Generator Data:**

| Bus | H (s) | MVA | Type | D | PSS |
|-----|------:|----:|------|--:|:---:|
| 29 | 500.0 | 250 | Hydro | 0.1 | — |
| 30 | 30.3 | 520 | Slack | 2.0 | — |
| 31 | 35.8 | 650 | Steam | 2.0 | ✓ |
| 32 | 28.6 | 632 | Steam | 2.0 | ✓ |
| 33 | 26.0 | 508 | Steam | 2.0 | — |
| 34 | 34.8 | 650 | Steam | 2.0 | ✓ |
| 35 | 26.4 | 560 | Steam | 2.0 | — |
| 36 | 24.3 | 540 | Steam | 2.0 | — |
| 37 | 34.5 | 830 | Steam | 2.0 | ✓ |
| 38 | 42.0 | 1000 | Steam | 2.0 | ✓ |

PSS enabled on units ≥ 600 MVA (5 per area, 15 total).

### 5.2 Excitation System (`ExcitationSystem`)

**Model:** IEEE IEEET1 per IEEE Std 421.5-2016, Section 5.1, Figure 5-1.

```
                  ┌─────┐     ┌──────────┐     ┌──────────────┐
  Vref ──(+)──→──│ KA  │──→──│ VRMIN/MAX│──→──│ 1/(KE+sTE)  │──→ Efd
          │  (-) │ 1+sTA│     │  Limiter │     │  + SE(Efd)  │
          │      └─────┘     └──────────┘     └──────────────┘
          │                                           │
          │  ┌──────────┐  ┌─────────┐               │
          └──│ 1/(1+sTR)│←─│Vt (meas)│               │
          │  └──────────┘  └─────────┘               │
          │                                           │
          │  ┌──────────────────┐                     │
          └──│ KF·s / (1+sTF)  │←────────────────────┘
             │ Stabilizing loop │
             └──────────────────┘
```

**Parameters:**

| Symbol | Value | Description |
|--------|------:|-------------|
| KA | 200.0 | Regulator gain |
| TA | 0.02 s | Regulator time constant |
| TR | 0.02 s | Transducer time constant |
| KE | 1.0 | Exciter constant |
| TE | 0.5 s | Exciter time constant |
| KF | 0.03 | Stabilizing feedback gain |
| TF | 1.0 s | Stabilizing feedback time constant |
| VRMAX / VRMIN | +5.0 / −5.0 | Regulator output limits |
| EMAX / EMIN | 5.0 / 0.0 | Field voltage limits |
| SE_A, SE_B | 0.0039, 1.555 | Saturation: $SE = A \cdot e^{B|E_{fd}|}$ |

**Steady-State Initialization:**

```
Efd₀ = Eq'  (from generator phasor init)
SE₀  = SE_A · exp(SE_B · |Efd₀|)
Vr₀  = (KE + SE₀) · Efd₀
Vf₀  = (KF / TF) · Efd₀
Vref = Vt₀ + Vr₀ / KA
```

This ensures **zero regulator error at t=0**, preventing initial transients.

**Over-Excitation Limiter (OEL):**

Per IEEE 421.5-2016 §6. If |Efd| > `Efd_max_thermal` (3.0 pu) for longer
than `OEL_delay_s` (10 s), the OEL progressively reduces Vref to pull Efd
back within thermal limits. Gain = 10.0, max reduction = 0.5 pu.

### 5.3 Governor-Turbine (`GovernorTurbine`)

**Model:** IEEE/NERC TGOV1

```
                 ┌────────┐     ┌────────┐
  omega ──→ Δω/R ──(+)→──│1/(1+sTG)│──→──│1/(1+sTT)│──→ Pm
                   (+)    └────────┘     └────────┘
  Pref ────────────┘      (with rate     (turbine)
                           limits)        - Dt·Δω
```

| Symbol | Value | Description |
|--------|------:|-------------|
| R | 0.05 | Droop (5%) |
| TG | 0.2 s | Governor time constant |
| TT | 0.5 s | Turbine time constant |
| Pmax / Pmin | 1.2 / 0.0 pu | Power limits |
| valve_rate_up | 0.1 pu/s | Opening rate |
| valve_rate_down | −0.1 pu/s | Closing rate |
| Dt | 0.05 | Turbine damping |

Governor output includes turbine damping:
$P_m = P_{turbine} - D_t \cdot \Delta\omega$

### 5.4 Power System Stabilizer (`PowerSystemStabilizer`)

**Model:** PSS1A per IEEE Std 421.5-2016, Section 8.1.

Signal path: Speed deviation → Washout → K_PSS → Lead-Lag 1 → Lead-Lag 2 → Clamp

**Washout block** $(sT_w) / (1 + sT_w)$:
- Rejects DC offsets; passes only oscillatory speed deviations
- $T_w = 1.41$ s

**Lead-Lag blocks** $(1 + sT_{lead}) / (1 + sT_{lag})$:
- Stage 1: $T_{lead1}=0.154$, $T_{lag1}=0.033$
- Stage 2: $T_{lead2}=0.154$, $T_{lag2}=0.033$
- DC gain = 1.0 per stage (verified by unit test)

**Overall gain:** $K_{PSS} = 5.0$, output clamped to ±0.2 pu.

The PSS output ($V_{PSS}$) feeds into the exciter summing junction as an
additional signal to damp electromechanical oscillations (typically 0.2–2.0 Hz).

### 5.5 Automatic Generation Control (AGC)

**ACE (Area Control Error):**

$$ACE = \Delta P_{tie} + \beta \cdot \Delta f$$

where $\beta = 1000$ MW/Hz (frequency bias), $\Delta f$ = frequency deviation
from 50.0 Hz with **0.03 Hz deadband** (IEGC standard).

**PI Controller:**

$$AGC_{output} = -K_{agc} \cdot ACE - \frac{1}{T_{agc}} \int ACE \, dt$$

$K_{agc} = 0.5$, $T_{agc} = 4.0$ s, integral anti-windup clamped to ±1000.

**Dispatch:**

AGC output is distributed to participating generators via **normalized
participation factors** proportional to MVA rating. Adjustments are applied
incrementally to governor Pref: `gov.Pref += Δp_pu × dt`

### 5.6 Dynamic Load Model (`DynamicLoadModel`)

**ZIP model** with frequency dependence:

$$P = P_0 \left( Z_p V^2 + I_p V + P_p \right) (1 + K_{pf} \Delta f)$$
$$Q = Q_0 \left( Z_q V^2 + I_q V + P_q \right) (1 + K_{qf} \Delta f)$$

Default coefficients: $Z_p=0.4$, $I_p=0.3$, $P_p=0.3$ (balanced ZIP).
Frequency sensitivity: $K_{pf}=1.5$, $K_{qf}=-1.0$.

### 5.7 Protection Systems (`ProtectionRelay`)

**Indian Grid Code (IEGC) compliant** relays with:

| Protection | Threshold | Action | Delay |
|------------|-----------|--------|------:|
| Undervoltage trip | < 0.90 pu | Generator trip | 2.0 s |
| Overvoltage trip | > 1.10 pu | Generator trip | 2.0 s |
| Underfrequency trip | < 49.5 Hz | Generator trip | 0.2 s |
| Overfrequency trip | > 50.5 Hz | Generator trip | 0.2 s |
| UFLS Stage 1 | < 49.5 Hz | Shed 10% load | 0.2 s |
| UFLS Stage 2 | < 49.2 Hz | Shed 15% load | 0.2 s |
| UFLS Stage 3 | < 49.0 Hz | Shed 20% load | 0.2 s |
| OFGT | > 50.5 Hz | Generation trip | 0.5 s |

**UFLS** (Under-Frequency Load Shedding) is multi-stage per IEGC, with
cumulative load shedding up to 45%.

### 5.8 DynamicsCoordinator

Orchestrates all dynamic models in correct execution order:

1. **Governors** — update Pm from Δω
2. **PSS** — compute Vpss from speed deviation
3. **Exciters** — update Efd from Vt, Vpss
4. **Generators** — swing equation + field flux (Efd → Eq')
5. **System frequency** — center-of-inertia weighted average:
   $f_{sys} = \frac{\sum H_i M_i f_i}{\sum H_i M_i}$
6. **AGC** — compute ACE, distribute adjustments to governors
7. **Loads** — update P, Q from voltage and frequency
8. **Protection** — check all relays, accumulate trip timers

```python
result = dynamics.step(dt=0.02, bus_voltages=bus_v, gen_powers=gen_p)
# result keys:
#   time, system_frequency_hz, generator_frequencies,
#   generator_angles, agc_adjustments, load_updates, protection
```

---

## 6. Distributed Energy Resources (DER)

**File:** `src/simulation/der.py` (1,077 lines)

### 6.1 DER Types

| Type | Network Element | Control Variable |
|------|-----------------|-----------------|
| `SOLAR_PV` | `sgen` | Irradiance (W/m²) |
| `WIND` | `sgen` | Wind speed (m/s) |
| `BESS` | `storage` | Power (MW, ±) |
| `EV_CHARGING` | `load` | Utilization (0–1) |
| `DEMAND_RESPONSE` | `load` | Curtailment fraction (0–1) |

### 6.2 Default Configuration (18 Units)

| Name | Type | Area | Bus | Capacity |
|------|------|------|----:|---------|
| Solar_A1 | Solar | A | 3 | 50 MW |
| Solar_A2 | Solar | A | 7 | 40 MW |
| Solar_A3 | Solar | A | 15 | 60 MW |
| BESS_A1 | Battery | A | 20 | 30 MW / 120 MWh |
| EV_Station_A1 | EV | A | 4 | 50×50kW = 2.5 MW |
| EV_FastCharge_A1 | EV | A | 12 | 30×150kW = 4.5 MW |
| DR_Industrial_A1 | DR | A | 8 | 25 MW base, 7.5 MW curt. |
| Wind_B1 | Wind | B | 42 | 80 MW (20 turbines) |
| Wind_B2 | Wind | B | 47 | 100 MW (25 turbines) |
| BESS_B1 | Battery | B | 54 | 40 MW / 160 MWh |
| EV_Station_B1 | EV | B | 46 | 40×50kW = 2.0 MW |
| DR_Commercial_B1 | DR | B | 59 | 30 MW base, 7.5 MW curt. |
| DR_Industrial_B1 | DR | B | 57 | 20 MW base, 8.0 MW curt. |
| Solar_C1 | Solar | C | 82 | 55 MW |
| Wind_C1 | Wind | C | 90 | 70 MW (18 turbines) |
| BESS_C1 | Battery | C | 103 | 50 MW / 200 MWh |
| EV_Station_C1 | EV | C | 86 | 60×50kW = 3.0 MW |
| EV_FastCharge_C1 | EV | C | 94 | 20×150kW = 3.0 MW |
| DR_Residential_C1 | DR | C | 93 | 35 MW base, 7.0 MW curt. |

**Totals:** 205 MW solar, 250 MW wind, 120 MW / 480 MWh battery,
15 MW EV, 36.5 MW curtailable DR.

### 6.3 Solar PV Model

Output calculation:

$$P_{out} = \min\left( \frac{A_{panel} \times G \times \eta}{10^6}, \; P_{rated} \right)$$

where $A_{panel}$ = panel area (m²), $G$ = irradiance (W/m²), $\eta$ = efficiency.

If `panel_area_m2` is not provided, it is computed from capacity:
$A = P_{rated} \times 10^6 / (1000 \times \eta)$

**Ramp rate limiting** (IEEE 1547-2018):

```python
# If ramp_rate_mw_per_min is set on the DERSpec:
max_change = ramp_rate × (dt_s / 60.0)     # dt_s in seconds
output = clip(output, last - max_change, last + max_change)
```

### 6.4 Wind Turbine Model

**IEC 61400 cubic power curve:**

$$P = P_{rated} \times \frac{v^3 - v_{ci}^3}{v_r^3 - v_{ci}^3}, \quad v_{ci} \leq v < v_r$$

| Parameter | Value | Description |
|-----------|------:|-------------|
| $v_{ci}$ | 3.0 m/s | Cut-in speed |
| $v_r$ | 12.0 m/s | Rated speed |
| $v_{co}$ | 25.0 m/s | Cut-out speed |

Below cut-in or above cut-out: $P = 0$. Between rated and cut-out: $P = P_{rated}$.

### 6.5 Battery Energy Storage (BESS)

**SOC tracking** with enhanced `BatteryState`:

- **Symmetric efficiency:** $\eta_{one-way} = \sqrt{\eta_{rt}}$ where $\eta_{rt}=0.92$
  - Charging: $E_{stored} = E_{input} \times \sqrt{\eta}$
  - Discharging: $E_{delivered} = E_{stored} / \sqrt{\eta}$
- **Cycle counting:** Each Δ(SOC) contributes to `total_cycles` (divided by 2 for full cycle)
- **Degradation:** Linear model, 20% capacity loss at 5,000 cycles:
  $d = \max(0.8, \; 1 - cycles/5000 \times 0.2)$

**SOC limits:** Charge blocked at SOC ≥ 0.95, discharge blocked at SOC ≤ 0.05.

### 6.6 EV Charging

Modelled as controllable loads with hourly utilization profiles:

| Hour | 0–5 | 6–9 | 10–17 | 18–22 | 23 |
|------|-----|-----|-------|-------|----|
| Utilization | 10–20% | 10–15% | 30–40% | 50–60% | 30% |

**Smart charging:** When `smart_override=True`, load is halved (50% reduction).

### 6.7 Demand Response

Curtailment reduces load from baseline:

$$P_{actual} = P_{baseline} - P_{curtailable} \times f_{curtail}$$

where $f_{curtail} \in [0, 1]$ and $P_{curtailable} = P_{baseline} \times f_{available}$.

### 6.8 IEEE 1547-2018 Compliance

**LVRT/HVRT Ride-Through (Category III):**

| Voltage Range | Must Ride Through For |
|---------------|-----------------------|
| 0.0 – 0.50 pu | Up to 1.0 s |
| 0.50 – 0.70 pu | Up to 2.0 s |
| 0.70 – 0.88 pu | Up to 10.0 s |
| 1.10 – 1.20 pu | Up to 0.5 s |
| > 1.20 pu | Up to 0.16 s |
| 0.88 – 1.10 pu | Continuous (normal) |

**Anti-Islanding Detection (IEEE 1547-2018):**

Passive detection using three signals:
- Frequency deviation > 0.5 Hz from 50.0 Hz
- Voltage outside 0.88–1.10 pu
- ROCOF > 1.0 Hz/s

2-second confirmation timer before disconnection (prevents nuisance trips).

### 6.9 Thread Safety

All public methods in `DERManager` are protected by `threading.RLock()`.
Safe for concurrent access from API threads and simulation threads.

---

## 7. Grid Orchestrator — RL Interface

**File:** `src/simulation/orchestrator.py` (950 lines)

### 7.1 Overview

`GridOrchestrator` is the **single entry point** for the RL agent. The agent
never touches submodules directly.

```python
from src.simulation.orchestrator import GridOrchestrator

orch = GridOrchestrator(seed=42)
obs = orch.reset()                          # (42,) float32

for step in range(288):                     # 24h at 5-min steps
    action = agent.predict(obs)             # (47,) float32 in [0, 1]
    obs, reward, done, info = orch.step(action)
    if done:
        break

summary = orch.episode_summary()
```

### 7.2 Observation Space (42 dimensions)

| Index | Description | Normalization |
|------:|-------------|---------------|
| 0–8 | Area A: gen, load, interchange, V_avg/min/max, violations, solar, wind | /10k, /10k, /1k, raw, raw, raw, /2, /100, /100 |
| 9–17 | Area B (same layout) | same |
| 18–26 | Area C (same layout) | same |
| 27 | System frequency | /50 |
| 28 | Frequency deviation | clip(±1) |
| 29 | Total generation | /30000 |
| 30 | Total load | /30000 |
| 31 | Total losses | /1000 |
| 32 | Has voltage violations | 0/1 |
| 33 | Has thermal violations | 0/1 |
| 34 | Average battery SOC | /100 |
| 35 | Total DER generation | /1000 |
| 36 | Hour of day | /24 |
| 37 | Solar irradiance | /1000 |
| 38 | Wind speed | /25 |
| 39 | Load scale factor | raw |
| 40 | Step / episode length | raw |
| 41 | Last reward | /5, clip(±1) |

All values clipped to [−2, 2] to prevent exploding gradients.

### 7.3 Action Space (47 dimensions)

All actions are in [0, 1] and linearly rescaled:

| Indices | Count | Action | Rescaling |
|---------|------:|--------|-----------|
| 0–26 | 27 | Generator MW setpoints | Pmin + frac × (Pmax − Pmin) |
| 27–29 | 3 | Battery power command | (frac − 0.5) × 2 × max_MW |
| 30–32 | 3 | Solar curtailment | 0 = no curtail, 1 = 100% |
| 33–35 | 3 | Wind curtailment | 0 = no curtail, 1 = 100% |
| 36–40 | 5 | EV utilization | 0 = off, 1 = full |
| 41–46 | 6 | Demand response curtailment | 0 = none, 1 = max |

**Solar/Wind curtailment:** Power is computed from irradiance/wind first at
full capacity, then curtailed at the inverter output (not the physical input).

### 7.4 Reward Function (Multi-Objective)

$$R = \sum_i w_i \cdot r_i, \quad r_i \in [-1, 1]$$

| Component | Weight | Formula |
|-----------|-------:|---------|
| Frequency | 0.30 | $1 - \min((|\Delta f| / 0.5)^2, 1)$ |
| Voltage | 0.25 | (buses within limits) / 117 |
| Economics | 0.20 | $1 - \min(loss\% / 3, 1)$ |
| Losses | 0.10 | $1 - \min(loss_{MW} / 500, 1)$ |
| Smoothness | 0.10 | $1 - \min(|\Delta action| / 0.3, 1)$ |
| Protection | 0.05 | +1 if zero trips, −1 otherwise |

### 7.5 Stochastic Profile Generator

Each `reset()` generates fresh realizations:

**Solar** — Gaussian bell centered at noon (σ=3h) × (1 + OU cloud noise):
- OU parameters: θ=0.3, σ=0.15
- Night mask: zero before 6:00 and after 18:00
- Peak: configurable (default 1000 W/m²)

**Wind** — Ornstein-Uhlenbeck mean-reverting process:
- Mean: 8.0 m/s, std: 3.0 m/s
- θ=0.15 (mean-reversion rate)
- Clipped to [0, 35] m/s

**Load** — Double-hump diurnal curve:
- Morning peak at 9:00, evening peak at 19:00
- Base range: 0.7–1.3 × baseline
- Gaussian noise: σ = load_variation_pct / 300

### 7.6 Step Pipeline (Per Control Step)

```
1. Action → ActionMapper.apply()          Apply RL commands to grid
2. StochasticProfile → Load scaling        Scale loads (skip DER loads)
3. PowerFlowRunner.run()                   Newton-Raphson AC power flow
4. DynamicsCoordinator.step() × 250        250 × 20ms = 5s dynamics
5. DERManager.update_battery_soc()         SOC bookkeeping
6. ProtectionRelay.check()                 Count trips
7. RewardCalculator.compute()              Multi-objective reward
8. ObservationBuilder.build()              42-dim normalized vector
9. Restore loads to BASE values            Prevent float drift
10. Check termination                       Step ≥ 288 → done
```

**Key design decisions:**
- DER-managed loads (EV, DR) are **excluded from load scaling** so RL actions
  are not overwritten by the stochastic profile
- Observation is taken **before** load restoration so the agent sees the
  actual scaled loads it will optimize against
- Base loads are snapshots taken at reset(), preventing float drift from
  repeated multiply-divide cycles

### 7.7 Episode Summary

```python
summary = orch.episode_summary()
# {
#   "episode": 1,
#   "steps": 288,
#   "cumulative_reward": 245.3,
#   "avg_reward": 0.852,
#   "pf_convergence_rate": 1.0,
#   "avg_frequency_hz": 50.001,
#   "max_freq_deviation_hz": 0.012,
#   "total_voltage_violations": 0,
#   "total_protection_trips": 0,
# }
```

---

## 8. Microgrid Standalone Simulator

**File:** `src/simulation/microgrid.py` (704 lines)

A **self-contained** microgrid simulator that does not depend on pandapower.
Uses PyPower-compatible array structures and a from-scratch Newton-Raphson
solver.

### 8.1 Components

| Class | Description |
|-------|-------------|
| `MicrogridCase` | PyPower-compatible bus/gen/branch arrays |
| `PowerFlowSolver` | Full NR with Jacobian construction |
| `SolarPV` | Irradiance-based output model |
| `WindTurbine` | IEC 61400 cubic power curve |
| `BatteryESS` | SOC tracking with efficiency |
| `DieselGenerator` | Ramp-rate-limited backup gen |
| `MicrogridController` | P-f droop, Q-V droop, battery SOC management |

### 8.2 Newton-Raphson Solver

- Builds full Y-bus admittance matrix from branch data
- Constructs 4-block Jacobian [J11 J12; J21 J22]
- Uses `scipy.sparse.spsolve` when available (O(n) for sparse systems)
- Falls back to `numpy.linalg.solve` otherwise
- Convergence tolerance: 1e-6, max 20 iterations

### 8.3 Example: 5-Bus Microgrid

```python
from src.simulation.microgrid import create_example_microgrid, PowerFlowSolver

case, ders = create_example_microgrid()
# Bus 0: Grid (slack), Bus 1: Solar, Bus 2: Wind, Bus 3: Battery, Bus 4: Diesel

solver = PowerFlowSolver(case)
results = solver.run()
print(f"Converged: {results['converged']} in {results['iterations']} iterations")
```

### 8.4 Diesel Generator Ramp Rate

```python
diesel = DieselGenerator("Diesel_1", bus=4, rated_power=3.0, ramp_rate=0.5)
# ramp_rate is MW/min; dt is in hours
actual = diesel.set_output(P_target=2.0, dt=1.0/60.0)  # 1-minute step
# max_change = 0.5 MW/min × (1/60 × 60 min) = 0.5 MW per step
```

---

## 9. Tie-Line Management

**File:** `src/simulation/tie_lines.py` (297 lines)

Provides:
- `TieLineConfig` dataclass with electrical parameters
- `DEFAULT_TIE_LINES` — a default set of inter-area connections
- `TieLineMonitor` — real-time thermal monitoring, flow analysis
- `add_tie_lines_to_network()` — programmatic tie-line creation
- `TieLineType` enum — HVAC 345kV, HVAC 500kV, HVDC

This module is complementary to the tie-line handling built into `SuperGrid`
(which uses its own `TieLineSpec` dataclass for the 8 N-1-secure lines).

---

## 10. IEEE Standard Compliance

### 10.1 Standards Implemented

| Standard | Section | Implementation | File |
|----------|---------|----------------|------|
| IEEE Std 421.5-2016 | §5.1 | IEEET1 exciter with SE saturation, KF feedback | `dynamics.py` |
| IEEE Std 421.5-2016 | §6 | Over-Excitation Limiter (OEL) | `dynamics.py` |
| IEEE Std 421.5-2016 | §8.1 | PSS1A: washout + 2 lead-lag, DC gain=1 | `dynamics.py` |
| IEEE/NERC TGOV1 | — | Valve rate limits, Dt turbine damping | `dynamics.py` |
| IEEE 1547-2018 | §6.4.1 | LVRT/HVRT Category III ride-through | `der.py` |
| IEEE 1547-2018 | §8.1 | Anti-islanding passive detection, 2s trip | `der.py` |
| IEEE 1547-2018 | §4.6.2 | Active power ramp rates (MW/min) | `der.py` |
| IEC 61400-27 | — | Cubic wind power curve | `der.py`, `microgrid.py` |
| IEGC (Indian Grid) | — | 50 Hz, 49.5–50.5 Hz band | `dynamics.py` |
| IEGC | — | Multi-stage UFLS (49.5/49.2/49.0 Hz) | `dynamics.py` |
| IEGC | — | OFGT at 50.5 Hz | `dynamics.py` |
| IEGC | — | 0.03 Hz AGC deadband | `dynamics.py` |
| IEEE 39-bus | — | New England test case, 0-indexed buses | `supergrid.py` |

### 10.2 Verification

- **PSS washout:** Unit-tested for DC rejection, 1 Hz passband, 0.01 Hz attenuation
- **Exciter init:** Verified zero-error steady state (Vf, Vref computed from SS)
- **Field flux:** Efd → Eq' coupling via trapezoidal ensures AVR/PSS is non-decorative
- **UFLS:** Multi-stage with cumulative shedding and per-stage timers
- **Ramp rates:** dt_s parameter ensures correct physical units regardless of call frequency

---

## 11. Numerical Methods

### 11.1 Integration Schemes

| Model | Method | Stability | Order |
|-------|--------|-----------|------:|
| Swing equation (ω) | Symplectic Euler | Conditionally stable | 1 |
| Swing equation (δ) | Symplectic Euler (uses new ω) | Energy-preserving | 1 |
| Field flux (Eq') | Implicit Trapezoidal | Unconditionally stable | 2 |
| Exciter (all blocks) | Implicit Trapezoidal | Unconditionally stable | 2 |
| Governor (Pg, Pm) | Implicit Trapezoidal | Unconditionally stable | 2 |
| PSS (all states) | Implicit Trapezoidal | Unconditionally stable | 2 |
| ZIP loads | Algebraic | — | — |

### 11.2 Trapezoidal Integration

For first-order systems $\dot{x} = (u - x) / T$:

$$x_{new} = \frac{x_{old} (2T - \Delta t) + 2u \cdot \Delta t}{2T + \Delta t}$$

This is the **implicit trapezoidal rule** (Tustin's method), unconditionally
stable for any $\Delta t > 0$ and $T > 0$. When $T \leq 0$, returns $u$
directly (algebraic bypass).

### 11.3 Symplectic Euler

For the swing equation (Hamiltonian system):

```
ω_new = ω_old + (dω/dt) × dt      ← update momentum first
δ_new = δ_old + ω₀(ω_new - 1) × dt  ← use NEW ω for position
```

This preserves the symplectic structure better than forward Euler, preventing
artificial energy drift in long simulations.

### 11.4 Time Scales

| Physical Process | Time Constant | Simulation dt |
|-----------------|-------------:|-------------:|
| Swing dynamics | 0.5–10 s | 20 ms |
| Exciter | 20–500 ms | 20 ms |
| Governor | 200–500 ms | 20 ms |
| PSS | 33–1410 ms | 20 ms |
| AGC | 4 s | 20 ms × 250 = 5 s |
| Load profile | 5 min | 300 s (control step) |
| DER ramp rate | 1–5 min | 300 s |
| ULTC tap changer | 30–60 s | Per call |

Each control step (5 min) runs **250 dynamics substeps** at 20 ms each,
covering 5 seconds of electromechanical dynamics.

---

## 12. Data Flow & Step-by-Step Pipeline

### 12.1 Initialization Flow

```
GridOrchestrator.__init__()
  ├── SuperGrid()
  │     ├── 3 × pn.case39()
  │     ├── merge_nets() × 2
  │     ├── _add_tie_lines()
  │     ├── _configure_slack_bus()
  │     └── _improve_convergence()
  ├── sg.initialize_der()
  │     └── Adds 18 DER units (sgen, storage, load elements)
  ├── sg.initialize_dynamics()
  │     ├── For each of 30 generators:
  │     │     ├── SynchronousGeneratorDynamic (swing eq)
  │     │     ├── ExcitationSystem (AVR)
  │     │     ├── GovernorTurbine (primary freq response)
  │     │     └── PowerSystemStabilizer (if MVA ≥ 600)
  │     ├── Initialize from PF steady-state
  │     └── Create 3 AGC controllers (one per area)
  ├── PowerFlowRunner()
  ├── StochasticProfileGenerator()
  ├── ObservationBuilder(), RewardCalculator(), ActionMapper()
  └── _solve_power_flow()  ← initial NR solve
```

### 12.2 Per-Step Data Flow

```
RL Agent  ──action[47]──→  GridOrchestrator.step()
                              │
            ┌─────────────────┼──────────────────────┐
            ▼                 ▼                       ▼
     ActionMapper       StochasticProfile     PowerFlowRunner
     (gen, DER,          (load scaling,        (Newton-Raphson)
      EV, DR)           skip DER loads)              │
            │                                        ▼
            │                                  DynamicsCoord
            │                                  (×250 substeps)
            │                                  │ governors
            │                                  │ PSS → exciters
            │                                  │ gen swing eq
            │                                  │ AGC → gen Pref
            │                                  │ loads, protection
            │                                        │
            ▼                                        ▼
     DERManager                              sys_frequency_hz
     (battery SOC,                                   │
      ramp rates)                                    ▼
            │                                  RewardCalculator
            │                                  (6 components)
            ▼                                        │
     ObservationBuilder ←────────────────────────────┘
     (42-dim vector)
            │
            ▼
     RL Agent  ←──(obs, reward, done, info)
```

---

## 13. Configuration Reference

### 13.1 ScenarioConfig

| Parameter | Default | Description |
|-----------|--------:|-------------|
| `episode_length_steps` | 288 | 24h at 5-min resolution |
| `dt_dynamics_s` | 0.02 | Electromechanical integration step (20 ms) |
| `dynamics_substeps` | 250 | 250 × 0.02 = 5.0 s dynamics per control step |
| `dt_control_s` | 300.0 | Control step = 5 minutes |
| `solar_irradiance_peak_w_m2` | 1000.0 | Peak irradiance for profile generation |
| `wind_speed_mean_m_s` | 8.0 | Mean wind speed |
| `wind_speed_std_m_s` | 3.0 | Wind speed volatility |
| `load_variation_pct` | 15.0 | ± load variation from base |
| `w_frequency` | 0.30 | Reward weight: frequency |
| `w_voltage` | 0.25 | Reward weight: voltage |
| `w_economics` | 0.20 | Reward weight: economics |
| `w_losses` | 0.10 | Reward weight: losses |
| `w_action_smoothness` | 0.10 | Reward weight: smoothness |
| `w_protection` | 0.05 | Reward weight: protection |
| `f_nominal_hz` | 50.0 | Nominal frequency |
| `f_min_hz / f_max_hz` | 49.5 / 50.5 | IEGC operating band |
| `v_min_pu / v_max_pu` | 0.95 / 1.05 | Voltage limits |

### 13.2 SuperGridConfig

| Parameter | Default | Description |
|-----------|--------:|-------------|
| `nominal_frequency_hz` | 50.0 | System nominal frequency |
| `base_mva` | 100.0 | System base MVA |
| `tie_line_rating_mva` | 600.0 | Default tie-line thermal limit |
| `v_min_pu / v_max_pu` | 0.95 / 1.05 | Voltage operating limits |
| `f_min_hz / f_max_hz` | 49.5 / 50.5 | Frequency operating limits |

### 13.3 PowerFlowConfig

| Parameter | Default | Description |
|-----------|--------:|-------------|
| `algorithm` | `NEWTON_RAPHSON` | Solver algorithm |
| `max_iterations` | 30 | Maximum NR iterations |
| `tolerance_mva` | 1e-8 | Convergence tolerance |
| `enforce_q_limits` | True | Enforce generator Q limits |
| `v_min_pu / v_max_pu` | 0.95 / 1.05 | Violation thresholds |

---

## 14. API Quick Reference

### SuperGrid

```python
sg = SuperGrid(config=SuperGridConfig())
sg.initialize_der()                                 # → DERManager
sg.initialize_dynamics()                            # → DynamicsCoordinator
sg.step_dynamics(dt=0.01)                           # → Dict
sg.step_ultc_tap_changers()                         # → Dict[int, int]
sg.get_global_state()                               # → Dict
sg.get_area_state(AreaID.AREA_A)                    # → Dict
sg.get_tie_line_flows()                             # → List[Dict]
sg.set_generator_setpoint(gen_idx, p_mw)            # → None
sg.scale_loads(AreaID.AREA_A, scale_factor=1.1)     # → None
sg.reset_to_base_case()                             # → None
```

### PowerFlowRunner

```python
runner = PowerFlowRunner(config=PowerFlowConfig())
result = runner.run(net, algorithm=None, verbose=False)     # → PowerFlowResult
results = runner.run_with_contingency(net, contingencies)   # → List[PowerFlowResult]
```

### DERManager

```python
dm = DERManager(net)
dm.add_solar_pv(bus, capacity_mw, name, area_id)
dm.add_wind_turbine(bus, capacity_mw, name, area_id, num_turbines)
dm.add_battery(bus, power_mw, energy_mwh, name, area_id)
dm.add_ev_charging_station(bus, num_chargers, charger_power_kw, name, area_id)
dm.add_demand_response(bus, baseline_load_mw, curtailable_fraction, name, area_id)

dm.set_solar_output(name, irradiance_w_m2, dt_s=300.0)
dm.set_wind_output(name, wind_speed_m_s, dt_s=300.0)
dm.set_battery_power(name, power_mw)
dm.set_ev_charging_load(name, utilization, smart_override=False)
dm.set_demand_response_curtailment(name, curtailment_fraction)
dm.update_ev_charging_by_hour(hour)
dm.update_battery_soc(timestep_hours)

dm.get_der_state(name)                              # → DERState
dm.get_all_der_states()                             # → List[DERState]
dm.get_status()                                     # → Dict aggregated by type
dm.get_total_generation()                           # → float MW
dm.check_voltage_ride_through(name, v_pu, dur_s)    # → bool
dm.check_anti_islanding(name, f_hz, v_pu, rocof)    # → bool
```

### DynamicsCoordinator

```python
dc = DynamicsCoordinator()
dc.add_generator(gen_id, bus, params, with_avr, with_governor, with_pss)
dc.add_load(bus, P0_mw, Q0_mvar)
dc.add_agc(area_id, [(gen_id, pf), ...])

result = dc.step(dt, bus_voltages, gen_powers)      # → Dict
dc.get_system_frequency()                           # → float Hz
dc.get_generator_states()                           # → Dict
```

### GridOrchestrator

```python
orch = GridOrchestrator(scenario=ScenarioConfig(), seed=42)
obs = orch.reset(seed=None)                         # → np.ndarray (42,)
obs, reward, done, info = orch.step(action)         # → tuple
summary = orch.episode_summary()                    # → Dict
orch.get_grid_state()                               # → Dict
orch.get_der_status()                               # → Dict
orch.get_generator_states()                         # → Dict
orch.get_tie_line_flows()                           # → List[Dict]
```

---

## 15. Testing

157 tests across 6 test files:

```bash
# Run all tests
python -m pytest tests/ -v --tb=short

# By module
python -m pytest tests/test_simulation.py -v         # SuperGrid, dynamics, DER, orchestrator
python -m pytest tests/test_dems.py -v               # Integration tests
python -m pytest tests/test_energy_manager.py -v     # Energy manager
python -m pytest tests/test_grid_manager.py -v       # Grid manager
python -m pytest tests/test_rl_agent.py -v           # RL agent
python -m pytest tests/test_environment.py -v        # Gymnasium environment

# Specific test classes
python -m pytest tests/test_simulation.py -k "TestDynamics" -v
python -m pytest tests/test_simulation.py -k "TestPSSWashoutFilter" -v
python -m pytest tests/test_simulation.py -k "TestGridOrchestrator" -v
python -m pytest tests/test_simulation.py -k "TestMicrogrid" -v
```

### Key Test Categories

| Category | Tests | Covers |
|----------|------:|--------|
| Grid construction | ~15 | Bus count, gen count, tie-lines, merge |
| Power flow | ~10 | Convergence, losses, violations |
| Dynamics | ~20 | Swing eq, AVR, governor, PSS, AGC |
| DER | ~25 | Solar, wind, battery SOC, EV, DR, LVRT |
| Orchestrator | ~15 | Obs shape, action dim, step, reset, reward |
| Microgrid | ~20 | NR solver, DER components, controller |
| PSS Washout | 3 | DC rejection, 1 Hz pass, low-freq attenuation |
| Integration | ~15 | End-to-end RL loop, file existence |

---

*Generated from source code analysis of `src/simulation/` package.*
*All 157 tests passing. Python 3.9+, pandapower 3.2.1.*
