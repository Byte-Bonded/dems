# Dynamic Energy Management System (DEMS)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-157%2F157-brightgreen)]()
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A realistic **117-bus tri-area power grid** simulation with integrated DER (Solar, Wind, Battery, EV, Demand Response), IEEE-compliant dynamic models, and a Reinforcement Learning orchestrator for intelligent grid optimization.

---

## Features

### Power System Simulation
- **117-Bus SuperGrid** — 3x IEEE 39-bus (New England) merged into a tri-area mesh topology with 8 N-1-secure tie-lines
- **AC Power Flow** — Newton-Raphson solver via pandapower, converges in ~4 iterations
- **Electromechanical Dynamics** — Swing equation (symplectic Euler), field flux coupling (Efd → Eq'), center-of-inertia frequency
- **IEEE 421.5 Excitation** — IEEET1 AVR with saturation, KF feedback, Over-Excitation Limiter (OEL)
- **IEEE/NERC TGOV1 Governor** — Valve rate limits, turbine damping Dt, PF-initialized Pref
- **IEEE 421.5 PSS1A** — Washout + 2 lead-lag stages, DC gain = 1, verified by unit tests
- **AGC** — PI-based secondary frequency control with IEGC 0.03 Hz deadband and normalized participation factors
- **Protection** — Multi-stage UFLS (49.5/49.2/49.0 Hz), OFGT (50.5 Hz), voltage trip with timing
- **ULTC Tap Changers** — Deadband-based automatic voltage regulation on transformers

### Distributed Energy Resources
- **Solar PV** — 205 MW across 3 areas, irradiance-based output with ramp rate limiting
- **Wind Farms** — 250 MW, IEC 61400 cubic power curve with cut-in/rated/cut-out
- **Battery Storage** — 120 MWh with symmetric `sqrt(eta)` efficiency, degradation tracking
- **EV Charging** — 200 chargers with smart charging profiles
- **Demand Response** — 36.5 MW curtailable load programs
- **IEEE 1547-2018** — LVRT/HVRT ride-through (Cat III), anti-islanding detection (2s), active power ramp rates

### RL Agent Interface
- **GridOrchestrator** — Single entry point: 42-dim observation, variable-dim action, multi-objective reward
- **Stochastic Profiles** — Ornstein-Uhlenbeck wind/cloud + diurnal load curves
- **PPO/SAC** — Stable-baselines3 integration (training on full 117-bus grid)

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all 157 tests
python -m pytest tests/ -v

# Use the SuperGrid directly
python -c "
from src.simulation.supergrid import SuperGrid
sg = SuperGrid()
sg.initialize_der()
sg.initialize_dynamics()
state = sg.get_global_state()
print(f'Buses: {len(sg.net.bus)}, Gens: {len(sg.net.gen)}')
print(f'Frequency: {state[\"global_metrics\"][\"system_frequency_hz\"]} Hz')
"
```

### Train RL Agent

```python
from src.simulation.orchestrator import GridOrchestrator
import numpy as np

orch = GridOrchestrator(seed=42)
obs = orch.reset()

for _ in range(288):  # 24h at 5-min steps
    action = np.random.uniform(0, 1, size=orch.action_dim)
    obs, reward, done, info = orch.step(action)
    if done:
        break

print(orch.episode_summary())
```

---

## Project Structure

```
dems/
├── src/
│   ├── simulation/           # Core power system simulation
│   │   ├── supergrid.py      # 117-bus tri-area grid (1036 lines)
│   │   ├── dynamics.py       # Swing eq, AVR, governor, PSS, AGC (733 lines)
│   │   ├── der.py            # DER manager (1071 lines)
│   │   ├── orchestrator.py   # RL orchestrator (945 lines)
│   │   ├── power_flow.py     # NR power flow engine (439 lines)
│   │   ├── microgrid.py      # Standalone microgrid simulator (704 lines)
│   │   └── tie_lines.py      # Tie-line configs
│   ├── orchestrator.py       # Monitoring orchestrator
│   ├── grid.py               # DEMSGrid wrapper (deprecated)
│   ├── agent/                # RL agent (PPO/SAC)
│   ├── api/                  # FastAPI backend
│   ├── core/                 # Energy/grid managers
│   └── monitoring/           # Prometheus metrics
├── tests/                    # 157 unit + integration tests
├── streamlit_app/            # Interactive dashboard
├── website/                  # React frontend
├── docs/                     # Architecture, API, DER guide
├── docker/                   # Dockerfiles
├── config/                   # Prometheus config
└── scripts/                  # Monitoring scripts
```

---

## IEEE Standard Compliance

| Standard | Implementation |
|----------|---------------|
| IEEE 39-bus (New England) | 3x merged, 0-indexed buses, correct gen/ext_grid mapping |
| IEEE Std 421.5-2016 sec5.1 | IEEET1 exciter: SE saturation, KF feedback, implicit trapezoidal |
| IEEE Std 421.5-2016 sec6 | OEL with thermal limit, timer delay, Vref reduction |
| IEEE Std 421.5-2016 sec8.1 | PSS1A: washout + 2 lead-lag, K=5, DC-rejection verified |
| IEEE/NERC TGOV1 | Valve rate limits, Dt=0.05 turbine damping |
| IEEE 1547-2018 | LVRT/HVRT Cat III, anti-islanding 2s, ramp rates |
| IEC 61400 | Cubic wind curve `(v^3 - v_ci^3) / (v_r^3 - v_ci^3)` |
| IEGC (Indian Grid Code) | 50 Hz, multi-stage UFLS, OFGT, 0.03 Hz AGC deadband |

---

## Testing

```bash
# Full suite
python -m pytest tests/ -v --tb=short

# Specific modules
python -m pytest tests/test_simulation.py -k "TestDynamics" -v
python -m pytest tests/test_simulation.py -k "TestPSSWashoutFilter" -v
python -m pytest tests/test_dems.py -k "TestIntegration" -v
```

**157 tests** covering:
- Grid construction and topology (buses, generators, tie-lines)
- Power flow convergence and loss calculation
- Dynamic models (swing eq, AVR, governor, PSS, AGC)
- DER operations (solar, wind, battery SOC, EV, DR)
- RL orchestrator (observation, action, reward, reset/step)
- Microgrid standalone solver (NR, DER components)
- PSS washout filter frequency response

---

## Monitoring (Docker)

```bash
docker-compose up -d
# API:        http://localhost:8000
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000 (admin/admin)
```

---

## Documentation

- [AUDIT_REPORT2.md](AUDIT_REPORT2.md) — Final audit report (all bugs resolved)
- [UNRESOLVED_BUGS.md](UNRESOLVED_BUGS.md) — Bug tracker (107/107 resolved)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — System architecture
- [docs/DER_GUIDE.md](docs/DER_GUIDE.md) — DER integration guide
- [docs/API.md](docs/API.md) — API reference
- [CONTRIBUTING.md](CONTRIBUTING.md) — Contribution guidelines

---

## License

MIT License — see [LICENSE](LICENSE)
