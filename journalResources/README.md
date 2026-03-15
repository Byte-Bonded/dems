# DEMS Journal Paper Resources

> **Dynamic Energy Management System: Hierarchical Multi-Agent Reinforcement Learning for IEEE-Compliant Power Grid Optimization**

This folder contains curated source code, architecture diagrams, and documentation organized for journal paper preparation.

---

## Folder Structure

```
journalResources/
├── README.md                          # This file
├── 01_multi_agent_rl/                 # Core RL agent implementations
│   ├── ppo_agents.py                  # PPO agent wrappers (Central/MG/Sub)
│   ├── coordinator.py                 # MultiAgentStepCoordinator (hierarchical step)
│   ├── ctde_trainer.py                # CTDE trainer with round-robin training
│   ├── obs_builder.py                 # Observation builders per hierarchy level
│   └── grid_constraints.py            # IEGC/IEEE constraint validation
├── 02_reward_functions/
│   └── reward_functions.py            # All 5 reward classes (Central/MG/3 Sub)
├── 03_grid_simulation/
│   ├── supergrid.py                   # 117-bus tri-area SuperGrid definition
│   ├── power_flow.py                  # NR solver with 6-strategy fallback
│   ├── orchestrator.py                # PhysicsEngine backbone
│   ├── der.py                         # DER Manager (Solar/Wind/BESS/EV/DR)
│   ├── economics.py                   # Economic dispatch, LMP, carbon
│   ├── tie_lines.py                   # Inter-area tie-line configuration
│   └── microgrid.py                   # Standalone microgrid simulator
├── 04_gymnasium_environments/
│   ├── central_env.py                 # CentralCoordEnv (24-D obs, 6-D action)
│   ├── microgrid_env.py               # MicrogridEnv (24-D obs, 8-D action)
│   ├── inverter_env.py                # InverterSubEnv (10-D obs, 4-D action)
│   ├── renewable_env.py               # RenewableSubEnv (10-D obs, 3-D action)
│   └── load_env.py                    # LoadSubEnv (10-D obs, 3-D action)
├── 05_dynamics_and_stability/
│   ├── dynamics.py                    # AVR/Governor/PSS/AGC/UFLS models
│   ├── small_signal.py                # Eigenvalue & participation factor analysis
│   └── transfer_functions.py          # IEEE Type 1 AVR/TGOV1/PSS1A TF models
├── 06_evaluation_and_baselines/
│   ├── run_eval.py                    # IEEE-compliant evaluation pipeline
│   ├── baselines.py                   # 4 baseline controllers (No-Control/Droop/MeritOrder/PI-AGC)
│   ├── ieee_plots.py                  # 18 IEEE publication-quality plot types
│   ├── metrics_collector.py           # Per-timestep metrics (freq/voltage/econ/reliability)
│   └── scenarios.py                   # 8 evaluation scenarios
├── 07_training_scripts/
│   ├── train.py                       # CLI training launcher
│   ├── train.sh                       # Automated training bash script
│   └── analyze_training.py            # Post-training analysis
├── 08_configuration/
│   ├── config.py                      # Centralized config (grid/RL/monitoring)
│   └── requirements.txt               # Python dependencies
├── 09_monitoring/
│   ├── metrics_exporter.py            # Prometheus metrics (40+ gauges/counters)
│   ├── prometheus.yml                 # Prometheus scrape configuration
│   └── alerts.yml                     # IEEE-compliant alert rules
└── 10_excalidraw_diagrams/
    ├── 01_system_architecture.excalidraw       # 4-layer system overview
    ├── 02_ctde_training_loop.excalidraw        # CTDE training flow
    ├── 03_supergrid_topology.excalidraw        # 117-bus tri-area topology
    ├── 04_reward_architecture.excalidraw       # Multi-level reward design
    └── 05_observation_action_spaces.excalidraw # Obs/action space details
```

---

## System Overview

| Aspect | Details |
|--------|---------|
| **Grid Model** | 117-bus tri-area SuperGrid (3x IEEE 39-bus New England) |
| **Generators** | 30 synchronous machines with AVR/Governor/PSS |
| **Transmission Lines** | 138+ lines, 8 tie-lines (mesh topology) |
| **DER Types** | Solar PV (205 MW), Wind (250 MW), BESS (120 MWh), EV (200 chargers), DR (36.5 MW) |
| **Frequency** | 50 Hz (Indian Grid Code / IEGC) |
| **RL Algorithm** | Proximal Policy Optimization (PPO) via Stable-Baselines3 |
| **Training Paradigm** | CTDE (Centralized Training, Decentralized Execution) |
| **Total Agents** | 13 (1 Central + 3 Microgrid + 9 Sub-agents) |
| **Test Cases** | 157 passing |

---

## Key Diagrams (Excalidraw)

Open these `.excalidraw` files in [Excalidraw](https://excalidraw.com/) for editing, then export as SVG/PNG for the paper.

| Diagram | Description | Suggested Paper Section |
|---------|-------------|------------------------|
| `01_system_architecture` | 4-layer architecture: Presentation → API → Multi-Agent RL → Physics Engine | Section II (System Architecture) |
| `02_ctde_training_loop` | 6-step CTDE training loop with reward computation sidebar | Section III (Methodology) |
| `03_supergrid_topology` | Tri-area 117-bus topology with tie-lines and DER | Section II-A (Grid Model) |
| `04_reward_architecture` | 3-level reward function decomposition with formulas | Section III-C (Reward Design) |
| `05_observation_action_spaces` | Complete obs/action space specification per agent level | Section III-B (Agent Design) |

---

## Paper Writing Plan

### Suggested Title
*"Hierarchical Multi-Agent Proximal Policy Optimization for Dynamic Energy Management in IEEE 39-Bus Based Tri-Area Power Systems with Distributed Energy Resources"*

### Proposed Outline

#### I. Introduction
- Motivation: Growing DER penetration challenges grid stability
- Gap: Traditional control methods lack adaptability to stochastic RE
- Contribution: Hierarchical MARL with CTDE for IEGC-compliant grid management

#### II. System Architecture
- **II-A. Grid Model**: 117-bus SuperGrid (3x IEEE 39-bus), tie-line mesh
  - *Key files*: `supergrid.py`, `tie_lines.py`
  - *Diagram*: `03_supergrid_topology.excalidraw`
- **II-B. Dynamic Models**: Swing equation, AVR (IEEE IEEET1), Governor (TGOV1), PSS (PSS1A), AGC, LFC
  - *Key files*: `dynamics.py`, `transfer_functions.py`
- **II-C. DER Integration**: IEEE 1547-2018 compliant Solar/Wind/BESS/EV/DR
  - *Key file*: `der.py`
- **II-D. Economic Model**: Quadratic cost curves, merit-order dispatch, LMP, TOU/RTP
  - *Key file*: `economics.py`

#### III. Proposed Methodology
- **III-A. Hierarchical Agent Design**: 3-level hierarchy (Central → Microgrid → Sub-agent)
  - *Key files*: `ppo_agents.py`, `coordinator.py`
  - *Diagram*: `01_system_architecture.excalidraw`
- **III-B. Observation & Action Spaces**: Detailed specification per level
  - *Key files*: `obs_builder.py`, all environment files in `04_gymnasium_environments/`
  - *Diagram*: `05_observation_action_spaces.excalidraw`
- **III-C. Reward Function Design**: Multi-objective, bounded [-1,1], weighted components
  - *Key file*: `reward_functions.py`
  - *Diagram*: `04_reward_architecture.excalidraw`
- **III-D. CTDE Training**: Round-robin training, Stable-Baselines3 PPO
  - *Key file*: `ctde_trainer.py`
  - *Diagram*: `02_ctde_training_loop.excalidraw`
- **III-E. Constraint Handling**: IEGC frequency limits, IEEE voltage standards, UFLS
  - *Key file*: `grid_constraints.py`

#### IV. Simulation Setup
- **IV-A. Scenarios**: 8 evaluation scenarios (base, high-RE, N-1, load ramp, islanding, etc.)
  - *Key file*: `scenarios.py`
- **IV-B. Baselines**: No-Control, Droop (R=5%), Merit-Order, PI-AGC
  - *Key file*: `baselines.py`
- **IV-C. Metrics**: IEEE 1547-2018, IEEE 1366, NERC BAL-001 compliant metrics
  - *Key file*: `metrics_collector.py`

#### V. Results & Discussion
- Training convergence curves
- Frequency regulation performance (IEGC compliance)
- Voltage profile quality (IEEE limits)
- Economic optimization (cost reduction vs baselines)
- DER utilization improvement
- Small-signal stability (eigenvalue analysis)
- **Plots**: Use `ieee_plots.py` for IEEE 2-column format (3.5"/7.16" width, 300 DPI, Times New Roman)

#### VI. Conclusion & Future Work

---

## IEEE Standards Referenced

| Standard | Application in DEMS |
|----------|-------------------|
| IEEE 39-bus | Test system topology (New England) |
| IEEE 421.5-2016 | Excitation system (IEEET1), PSS models |
| IEEE 1547-2018 | DER interconnection (LVRT/HVRT, anti-islanding) |
| IEEE 1366 | Reliability indices (SAIDI, SAIFI) |
| IEGC (Indian Grid Code) | Frequency limits (49.5-50.5 Hz), AGC deadband (0.03 Hz) |
| NERC BAL-001 | Area Control Error (ACE) requirements |

---

## PPO Hyperparameters (for paper's Table)

| Parameter | Value |
|-----------|-------|
| Learning Rate | 3 x 10^-4 |
| Discount Factor (gamma) | 0.99 |
| GAE Lambda | 0.95 |
| Clip Range | 0.2 |
| Entropy Coefficient | 0.01 |
| Value Function Coefficient | 0.5 |
| Network (Central/MG) | MLP [256, 256] |
| Network (Sub-agents) | MLP [128, 128] |
| Batch Size | 64 |
| Framework | Stable-Baselines3 + PyTorch |

---

## How to Generate Figures

1. Open any `.excalidraw` file at [excalidraw.com](https://excalidraw.com/)
2. Edit as needed (adjust labels, colors, layout)
3. Export as SVG or PNG (300 DPI recommended for IEEE)
4. For IEEE publication plots (Bode, eigenvalue, etc.), run:
   ```bash
   python evaluation/run_eval.py --compare-all
   ```
   This generates all 18 IEEE-format plot types via `ieee_plots.py`

---

## Quick Reference: File-to-Paper Mapping

| Paper Component | Source File(s) |
|----------------|---------------|
| Agent hierarchy table | `ppo_agents.py` (line ~1-30: agent configs) |
| Obs/action dimensions table | `obs_builder.py`, env files |
| Reward formulation equations | `reward_functions.py` |
| Grid parameters table | `supergrid.py`, `config.py` |
| DER specifications table | `der.py` |
| Dynamic model equations | `dynamics.py`, `transfer_functions.py` |
| Cost function equations | `economics.py` |
| Constraint thresholds table | `grid_constraints.py` |
| Evaluation scenario table | `scenarios.py` |
| Baseline descriptions | `baselines.py` |
| Training algorithm pseudocode | `ctde_trainer.py`, `coordinator.py` |
| Metric definitions | `metrics_collector.py` |
