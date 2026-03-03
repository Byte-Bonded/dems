# DEMS Architecture Guide

## System Overview

DEMS is a layered system designed for scalability and maintainability:

```
┌─────────────────────────────────────────┐
│          User Interface Layer           │
│     (React GitHub Pages Website)        │
│    White & Pinkish-Red Modern UI        │
└──────────────┬──────────────────────────┘
               │ (HTTP/REST)
┌──────────────▼──────────────────────────┐
│          API Layer                      │
│        (FastAPI Server)                 │
│  - Health endpoints                     │
│  - Energy management                    │
│  - Grid control                         │
│  - Metrics collection                   │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│       Business Logic Layer              │
│ ┌──────────────┐  ┌───────────────────┐│
│ │ Core Module  │  │  Agent Module     ││
│ │ - Energy Mgr │  │ - RL Agent        ││
│ │ - Grid Mgr   │  │ - Gym Environment ││
│ └──────────────┘  └───────────────────┘│
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│       Data & Monitoring Layer           │
│ ┌──────────────┐  ┌───────────────────┐│
│ │ Persistence  │  │  Monitoring       ││
│ │ - PostgreSQL │  │ - Prometheus      ││
│ │ - Redis      │  │ - Grafana         ││
│ └──────────────┘  └───────────────────┘│
└─────────────────────────────────────────┘
```

## Core Components

### 1. Energy Manager (`src/core/energy_manager.py`)
- Manages energy generation and consumption
- Tracks storage levels
- Orchestrates optimization requests
- Maintains system state

### 2. Grid Manager (`src/core/grid_manager.py`)
- Manages grid nodes and topology
- Monitors frequency and voltage
- Handles load distribution
- Checks grid stability

### 3. RL Agent (`src/agent/rl_agent.py`)
- Stable-Baselines3 based agent
- Predicts optimal actions
- Learns from experiences
- Manages training state

### 4. DEMS Environment (`src/agent/environment.py`)
- Gym-compatible interface
- State and action spaces
- Reward calculation
- Episode management

### 5. API Server (`src/api/main.py`)
- FastAPI application
- REST endpoints
- Error handling
- Logging integration

### 6. Monitoring (`src/monitoring/__init__.py`)
- Prometheus client integration
- Metrics collection
- Performance tracking
- Performance decorators

## Data Flow

### Energy Optimization Flow
```
1. API Request (/optimization/predict)
   ↓
2. EnergyManager.optimize_distribution()
   ↓
3. RLAgent.predict(state)
   ↓
4. Action returned to API
   ↓
5. Optimization metrics recorded
   ↓
6. Response sent to client
```

### Monitoring Flow
```
1. System operations
   ↓
2. Metrics collected (prometheus_client)
   ↓
3. Metrics exposed at /metrics endpoint
   ↓
4. Prometheus scrapes metrics
   ↓
5. Grafana visualizes data
```

## Technology Stack

### Backend
- **Python 3.9+**: Core language
- **FastAPI**: Web framework
- **SQLAlchemy**: ORM
- **Pydantic**: Data validation

### Machine Learning
- **Stable-Baselines3**: RL algorithms
- **OpenAI Gym**: Environment interface
- **NumPy/SciPy**: Numerical computing
- **Ray RLlib**: Distributed RL (optional)

### Database
- **PostgreSQL**: Persistent data
- **Redis**: Caching layer

### Monitoring
- **Prometheus**: Metrics collection
- **Grafana**: Data visualization

### Frontend
- **React 18+**: UI framework
- **Recharts**: Data visualization
- **React Router**: Navigation
- **React Icons**: Icon library

### DevOps
- **Docker**: Containerization
- **Docker Compose**: Orchestration
- **GitHub Actions**: CI/CD

## Deployment Architecture

### Development
```
Local Machine
├── Backend (port 8000)
├── Frontend (port 3000)
├── PostgreSQL (port 5432)
└── Redis (port 6379)
```

### Docker
```
Docker Compose
├── DEMS API Container
├── Prometheus Container
├── Grafana Container
├── PostgreSQL Container
└── Redis Container
```

### Production (GitHub Pages)
```
GitHub Pages
└── React Build (Static HTML/JS/CSS)
    └── https://Byte-Bonded.github.io/dems
```

## Design Patterns

### 1. Factory Pattern
- `RLAgent` initialization with configurable parameters

### 2. Observer Pattern
- Prometheus metrics collection

### 3. Repository Pattern
- Data access through managers

### 4. Decorator Pattern
- `@track_performance` for metrics collection

### 5. Strategy Pattern
- Different optimization strategies in RLAgent

## Error Handling

```python
# API Level
try:
    result = optimize()
except Exception as e:
    logger.error(f"Optimization error: {str(e)}")
    raise HTTPException(status_code=500, detail=str(e))

# Metrics Level
optimization_errors.inc()  # Track errors
optimization_requests.inc()  # Track all requests
```

## Scalability Considerations

1. **Horizontal Scaling**
   - Multiple API instances behind load balancer
   - Distributed Redis for caching
   - PostgreSQL replication

2. **Vertical Scaling**
   - Increase container resources
   - Optimize RL model size
   - Cache frequently accessed data

3. **Performance Optimization**
   - Prometheus scrape interval tuning
   - Redis caching strategy
   - Query optimization

## Security Considerations

1. **Data Protection**
   - Environment variables for secrets
   - Database encryption
   - HTTPS for API endpoints

2. **API Security**
   - CORS configuration
   - Rate limiting (future)
   - Authentication/Authorization (future)

3. **Infrastructure**
   - Docker security best practices
   - Network isolation
   - Regular updates

## Testing Strategy

```
Unit Tests (test_*.py)
├── Core module tests
├── Agent tests
└── Environment tests

Integration Tests (future)
├── API endpoint tests
├── Database tests
└── Monitoring tests

Load Tests (future)
└── Performance benchmarks
```

## Monitoring & Observability

### Prometheus Metrics
- Energy metrics (generation, consumption, storage)
- Grid metrics (frequency, voltage)
- Agent metrics (rewards, episodes)
- Performance metrics (latency, errors)

### Grafana Dashboards
- Energy monitoring
- Grid stability
- Agent performance
- System health

### Logging
- Structured logging with JSON
- Log levels: DEBUG, INFO, WARNING, ERROR
- Centralized log aggregation (future)

## Future Enhancements

1. **Advanced RL**
   - Multi-agent RL
   - Imitation learning
   - Curriculum learning

2. **Analytics**
   - Predictive analytics
   - Anomaly detection
   - Trend analysis

3. **Integration**
   - Real grid data
   - IoT device integration
   - Weather data integration

4. **Frontend**
   - Real-time updates (WebSocket)
   - Advanced dashboards
   - Mobile app
