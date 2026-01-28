# Dynamic Energy Management System (DEMS)

[![GitHub Repo](https://img.shields.io/badge/GitHub-Byte--Bonded%2Fdems-blue)](https://github.com/Byte-Bonded/dems)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18%2B-blue)](https://reactjs.org/)

**DEMS** is a Dynamic Energy Management System built around a realistic **117-bus IEEE 39-based tri-area power grid** with integrated **Distributed Energy Resources (DER)** including Solar, Wind, Battery Storage, EV Charging, and Demand Response, plus **Reinforcement Learning** capabilities for intelligent grid optimization.

## ⚡ What Actually Works

### ✅ Fully Functional
- **117-Bus SuperGrid**: Tri-area power system based on 3× IEEE 39-bus networks
  - 117 buses, 27 generators, 8 inter-area tie-lines
  - 50 Hz Indian Grid Code operation
  - AC power flow converges in ~4 iterations (84ms)
  - Realistic voltage/frequency control with shunt compensators

- **DER Integration**: Real distributed energy resources
  - Solar PV: 205 MW capacity with irradiance-based generation
  - Wind: 250 MW capacity with wind speed curves
  - Battery Storage: 120 MWh with SOC tracking
  - **EV Charging**: 200 chargers with smart charging (15 MW peak)
  - **Demand Response**: 36.5 MW curtailable load programs
  - Integrated with power flow solver

- **Power Flow Simulation**: Production-ready pandapower integration
  - Newton-Raphson solver
  - Voltage stability analysis
  - Loss calculation
  - Tested and validated

### 🚧 In Development
- **RL-Based Optimization**: PPO/SAC agents implemented, needs SuperGrid integration
  - RL agent structure complete with stable_baselines3
  - Environment currently uses simplified 10-node model
  - **Next step**: Connect DEMSEnvironment to 117-bus SuperGrid

- **API Backend**: FastAPI structure exists but needs SuperGrid connection
  - Health endpoints working
  - **Needs**: Replace toy models with real SuperGrid interface

- **Monitoring Stack**: Docker/Prometheus/Grafana configured but not fully integrated
  - Infrastructure ready
  - **Needs**: Metrics from real power flow results

### 📋 Planned Features
- Real-time grid monitoring with Prometheus metrics
- RL agent training on actual 117-bus power flow
- Multi-agent coordination for DER control
- ~~Demand response integration~~ ✅ COMPLETED
- ~~EV charging optimization~~ ✅ COMPLETED
**DEMS** is a sophisticated Dynamic Energy Management System that uses Reinforcement Learning (RL) agents to optimize energy distribution, storage, and grid stability. The system combines advanced AI algorithms with real-time monitoring using Prometheus and Grafana.

## 🌟 Features

### Core Capabilities
- **RL-Based Optimization**: Intelligent agents trained to optimize energy distribution in real-time
- **Grid Stability Management**: Maintains frequency and voltage stability across grid nodes
- **Smart Load Balancing**: Dynamic distribution minimizes losses and maximizes efficiency
- **Energy Storage Optimization**: Intelligent charging/discharging strategies for battery systems
- **Real-Time Monitoring**: Prometheus metrics collection and Grafana dashboards

### Technical Features
- **FastAPI Backend**: High-performance REST API for energy management
- **PostgreSQL Database**: Reliable data persistence
- **Redis Caching**: Fast access to critical metrics
- **Docker Support**: Easy deployment across environments
- **GitHub Pages Website**: Modern responsive UI with React (White & Pinkish-Red theme)

## 🏗️ Project Structure

```
dems/
├── src/
│   ├── simulation/        # ✅ WORKING - Real power system
│   │   ├── supergrid.py   # 117-bus grid implementation
│   │   ├── der.py         # DER manager (Solar/Wind/BESS)
│   │   ├── power_flow.py  # AC power flow solver
│   │   └── tie_lines.py   # Inter-area connections
│   ├── grid.py            # ✅ High-level grid interface
│   ├── agent/             # 🚧 RL agent (needs integration)
│   │   ├── rl_agent.py    # PPO/SAC implementation
│   │   └── environment.py # Gym environment (10-node)
│   ├── api/               # 🚧 FastAPI (needs SuperGrid)
│   │   └── main.py
│   └── core/              # 📋 Simplified models
├── tests/                 # ✅ Unit + integration tests
│   ├── test_integration.py  # NEW - Tests real SuperGrid
│   ├── test_rl_agent.py     # Tests RL implementation
│   └── ...
├── docs/                  # ✅ Documentation
│   ├── DER_GUIDE.md       # DER integration guide
│   ├── RL_CONTROL_STRATEGY.md
│   └── ARCHITECTURE.md
├── website/               # ✅ React frontend
└── docker-compose.yml     # 🚧 Monitoring stack
│   ├── core/              # Energy and grid management
│   │   ├── energy_manager.py
│   │   └── grid_manager.py
│   ├── agent/             # RL agent implementation
│   │   ├── rl_agent.py
│   │   └── environment.py
│   ├── api/               # FastAPI application
│   │   └── main.py
│   ├── monitoring/        # Prometheus integration
│   └── models/            # Data models
├── website/               # React GitHub Pages site
│   ├── public/
│   └── src/
│       ├── components/
│       ├── pages/
│       └── styles/
├── config/                # Configuration
│   ├── prometheus.yml
│   └── config.py
├── docker/                # Docker files
├── tests/                 # Unit tests
├── docs/                  # Documentation
├── docker-compose.yml
├── requirements.txt
└── setup.py
```

## 🚀 Quick Start

### Test the Real Grid
### Backend
```bash
# Install dependencies
pip install -r requirements.txt

# Test 117-bus SuperGrid
python test_grid.py

# Test DER integration
python quick_test_der.py

# Visualize topology
python visualize_grid.py
```

### Train RL Agent (Current - 10 nodes)
```python
from src.agent import RLAgent, DEMSEnvironment

# Current: Simplified 10-node environment
env = DEMSEnvironment(num_nodes=10)
agent = RLAgent(env=env, algorithm="PPO")
agent.train(total_timesteps=10000)
```

### Use Real SuperGrid (Recommended)
```python
from src.grid import DEMSGrid

# Real 117-bus grid with DER
grid = DEMSGrid()
result = grid.run_power_flow()

print(f"Converged: {result['converged']}")
print(f"Generation: {result['total_generation_mw']:.2f} MW")
print(f"Losses: {result['total_losses_mw']:.2f} MW")

# Control DER
grid.supergrid.der_manager.update_solar_generation(hour=12, irradiance=800)
grid.supergrid.der_manager.update_wind_generation(hour=12, wind_speed=8.0)
grid.supergrid.der_manager.update_ev_charging_by_hour(hour=19)  # Peak evening
grid.supergrid.der_manager.set_demand_response_curtailment("DR_Industrial_A1", 0.5)
```

## 📊 Monitoring (In Development)

Docker stack configured but not fully connected:

```bash
docker-compose up -d

# Services:
# - API: http://localhost:8000 (🚧 needs SuperGrid)
# - Prometheus: http://localhost:9090 (🚧 needs metrics)
# - Grafana: http://localhost:3000 (🚧 needs data source)
```

## 🤖 RL Agent Implementation

**Status**: ✅ Implemented with stable_baselines3

```python
from src.agent import RLAgent, DEMSEnvironment

# Create environment (currently 10-node, upgrade to SuperGrid planned)
env = DEMSEnvironment()

# Initialize PPO or SAC agent
agent = RLAgent(env=env, algorithm="PPO", verbose=1)

# Train agent
agent.train(total_timesteps=50000)

# Evaluate performance
results = agent.evaluate(n_episodes=10)
print(f"Mean reward: {results['mean_reward']:.2f}")

# Save/load models
agent.save("models/ppo_agent")
agent.load("models/ppo_agent")
```

**Algorithms Supported**:
- **PPO** (Proximal Policy Optimization) - Default, stable
- **SAC** (Soft Actor-Critic) - For continuous control

## 🧪 Testing

```bash
# Run all tests
pytest

# Run integration tests (tests real SuperGrid)
pytest tests/test_dems.py -k TestIntegration -v

# Run RL agent tests
pytest tests/test_dems.py::TestRLAgent -v

# Coverage report
pytest --cov=src tests/
# Copy environment config
cp .env.example .env

# Run API
python -m src.api.main
# API: http://localhost:8000
```

### Frontend
```bash
cd website
npm install
npm start
# Website: http://localhost:3000
```

### Docker
```bash
# Start all services
docker-compose up -d

# Services:
# - API: http://localhost:8000
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000
# - PostgreSQL: localhost:5432
# - Redis: localhost:6379
```

## 📊 Monitoring

**Prometheus** collects these metrics:
- `dems_energy_generated_kwh` - Energy generation
- `dems_energy_consumed_kwh` - Energy consumption  
- `dems_storage_level_kwh` - Storage level
- `dems_grid_frequency_hz` - Grid frequency
- `dems_agent_reward` - Agent reward

**Grafana** visualizes the data (login: admin/admin)

## 🤖 RL Agent

The system uses Stable-Baselines3 for RL training:

```python
from src.agent import RLAgent, DEMSEnvironment

env = DEMSEnvironment(num_nodes=10)
agent = RLAgent(observation_space_size=50, action_space_size=10)

# Training and prediction logic
```

## 🌐 GitHub Pages Deployment

The website is built with React and deployed to GitHub Pages:

```bash
cd website
npm run deploy
# Site: https://Byte-Bonded.github.io/dems
```

## 🔌 API Endpoints

```bash
GET /health                    # Health check
GET /energy/state              # Current energy state
GET /grid/state                # Grid status
POST /optimization/predict     # Get optimization
GET /metrics                   # Prometheus metrics
```

## 📚 Documentation

- [DER Integration Guide](docs/DER_GUIDE.md) - How DER works with SuperGrid
- [Project Changelog](PROJECT_CHANGELOG.md) - Development summary and milestones

## 🎯 Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| 117-Bus SuperGrid | ✅ Complete | Production ready, tested |
| DER Integration | ✅ Complete | Solar, Wind, Battery working |
| Power Flow | ✅ Complete | Converges reliably |
| RL Agent | ✅ Implemented | PPO/SAC with stable_baselines3 |
| RL Environment | 🚧 Partial | Uses 10-node, needs SuperGrid |
| API Backend | 🚧 Partial | Structure exists, needs integration |
| Monitoring | 🚧 Partial | Docker setup ready, needs metrics |
| Frontend | ✅ Complete | React website ready |

## 🛠️ Next Steps (Roadmap)

1. **Connect RL Environment to SuperGrid** (High Priority)
   - Replace 10-node model with 117-bus SuperGrid
   - Use real power flow results as observations
   - Control DER as actions

2. **Integrate API with SuperGrid** (High Priority)
   - Replace simplified models with real grid interface
   - Expose power flow results via REST API

3. **Add Prometheus Metrics** (Medium Priority)
   - Export SuperGrid metrics (voltage, frequency, generation)
   - Connect Grafana dashboards

4. **Advanced Features** (Future)
   - Multi-agent RL for distributed control
   - Demand response integration
   - EV charging optimization

## 🌐 GitHub Pages

Website: [https://Byte-Bonded.github.io/dems](https://Byte-Bonded.github.io/dems)

```bash
cd website
npm install
npm run deploy
```
- [Getting Started](docs/GETTING_STARTED.md)
- [API Reference](docs/API.md)
- [Architecture](docs/ARCHITECTURE.md)

## 📝 License

MIT License - see [LICENSE](LICENSE)

## 🤝 Contributing

Contributions welcome! Please open issues and pull requests on GitHub.

---

**⚡ Building realistic power grid simulation with AI-driven optimization**

*Note: This project implements a realistic 117-bus power system. The RL integration is in progress - the grid simulation is production-ready, while the RL agent needs to be connected to the real grid for full functionality.*
Contributions welcome! Please create a pull request.

---

**⚡ Building the future of energy management with AI** 
