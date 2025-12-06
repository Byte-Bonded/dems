# Dynamic Energy Management System (DEMS)

[![GitHub Repo](https://img.shields.io/badge/GitHub-Byte--Bonded%2Fdems-blue)](https://github.com/Byte-Bonded/dems)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18%2B-blue)](https://reactjs.org/)

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

### Backend
```bash
# Install dependencies
pip install -r requirements.txt

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

- [Getting Started](docs/GETTING_STARTED.md)
- [API Reference](docs/API.md)
- [Architecture](docs/ARCHITECTURE.md)

## 📝 License

MIT License - see [LICENSE](LICENSE)

## 🤝 Contributing

Contributions welcome! Please create a pull request.

---

**⚡ Building the future of energy management with AI** 
