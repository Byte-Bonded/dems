# DEMS Monitoring Stack

Real-time monitoring for the Dynamic Energy Management System using Prometheus and Grafana.

## Quick Start

```bash
# Start the monitoring stack
./setup.sh start

# Access dashboards
# Grafana:    http://localhost:3000 (admin/dems2024)
# Prometheus: http://localhost:9090
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DEMS Application                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  FastAPI     │  │  Simulator   │  │  RL Agent    │          │
│  │  :8000       │  │  :8001       │  │  :8002       │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│         └────────────────┼─────────────────┘                   │
│                          │ /metrics                             │
└──────────────────────────┼─────────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────────┐
│                     Docker Network                              │
│  ┌──────────────────┐      ┌──────────────────┐                │
│  │   Prometheus     │      │     Grafana      │                │
│  │   :9090          │◀────▶│     :3000        │                │
│  │                  │      │                  │                │
│  │  - Scrape jobs   │      │  - Dashboards    │                │
│  │  - Alert rules   │      │  - Alerts        │                │
│  │  - TSDB storage  │      │  - Visualize     │                │
│  └──────────────────┘      └──────────────────┘                │
│                                                                 │
│  ┌──────────────────┐                                          │
│  │  Node Exporter   │  (optional host metrics)                 │
│  │  :9100           │                                          │
│  └──────────────────┘                                          │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### Prometheus (`prometheus/`)
- **prometheus.yml** - Scrape configuration for DEMS metrics
- **alerts.yml** - Alert rules for grid stability, DER, RL training

### Grafana (`grafana/`)
- **provisioning/** - Auto-configuration for datasources and dashboards
- **dashboards/** - Pre-built DEMS dashboards

## Dashboards

### DEMS Overview Dashboard
Main dashboard showing:

1. **System Overview**
   - System frequency (with IEEE 1547-2018 limits)
   - Average voltage (p.u.)
   - System losses (%)
   - Voltage violations
   - Renewable fraction
   - UFLS protection status

2. **Distributed Energy Resources**
   - Solar/Wind/Battery generation stack
   - Battery SOC gauge
   - Renewable curtailment tracking

3. **Tri-Area Interconnection**
   - Tie-line power flows (A↔B↔C)
   - Area load distribution

4. **RL Agent Performance**
   - Training steps/episodes
   - Episode reward curves
   - Policy/Value loss tracking
   - Training status indicator

5. **Economic & Environmental**
   - Operating cost ($/h)
   - System marginal cost
   - Electricity prices (TOU/RTP)
   - Carbon emissions & intensity

## Commands

```bash
# Start monitoring
./setup.sh start

# Stop monitoring
./setup.sh stop

# View logs
./setup.sh logs
./setup.sh logs grafana
./setup.sh logs prometheus

# Check status
./setup.sh status

# Full cleanup (removes data)
./setup.sh cleanup
```

## Exposing DEMS Metrics

Add the metrics exporter to your DEMS application:

```python
from src.monitoring import DEMSMetricsExporter

# Initialize exporter
exporter = DEMSMetricsExporter(orchestrator=orch, port=9091)
exporter.start()

# In simulation loop
for step in range(steps):
    obs, reward, done, info = orch.step(action)
    exporter.update()  # Push current state to Prometheus
    
# For RL training metrics
exporter.update_rl_metrics(
    training_active=True,
    steps=total_steps,
    episodes=total_episodes,
    episode_reward=ep_reward,
    policy_loss=loss_dict['policy_loss'],
    value_loss=loss_dict['value_loss'],
)
```

## Alert Rules

Pre-configured alerts for:

| Alert | Threshold | Severity |
|-------|-----------|----------|
| FrequencyDeviationWarning | ±0.2 Hz | Warning |
| FrequencyDeviationCritical | ±0.5 Hz | Critical |
| VoltageDeviationWarning | ±5% | Warning |
| VoltageDeviationCritical | ±10% | Critical |
| BatterySOCLow | <15% | Warning |
| BatterySOCCritical | <5% | Critical |
| HighRenewableCurtailment | >20% | Warning |
| AgentTrainingStalled | No progress 10m | Warning |
| PowerFlowNotConverging | Any failure | Critical |

## Configuration

### Environment Variables

Customize via environment variables in `docker-compose.yml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GF_SECURITY_ADMIN_PASSWORD` | `dems2024` | Grafana admin password |
| `GF_SERVER_ROOT_URL` | `http://localhost:3000` | Grafana URL |

### Prometheus Scrape Targets

Edit `prometheus/prometheus.yml` to add/modify scrape targets:

```yaml
scrape_configs:
  - job_name: 'my-dems-service'
    static_configs:
      - targets: ['host.docker.internal:8000']
```

## Troubleshooting

### Metrics not appearing
1. Check DEMS exporter is running: `curl http://localhost:9091/metrics`
2. Check Prometheus targets: http://localhost:9090/targets
3. Verify network connectivity between containers

### Dashboard shows "No data"
1. Ensure metric names match (check `prometheus/prometheus.yml`)
2. Verify time range in Grafana
3. Check Prometheus is scraping: http://localhost:9090/graph

### Docker issues on macOS
- Use `host.docker.internal` to access host services from containers
- Node Exporter may have limited metrics on macOS

## Files

```
monitoring/
├── docker-compose.yml          # Container orchestration
├── setup.sh                    # Management script
├── README.md                   # This file
├── prometheus/
│   ├── prometheus.yml          # Scrape configuration
│   └── alerts.yml              # Alert rules
└── grafana/
    ├── provisioning/
    │   ├── datasources/
    │   │   └── datasources.yml # Prometheus datasource
    │   └── dashboards/
    │       └── dashboards.yml  # Dashboard provider
    └── dashboards/
        └── dems_overview.json  # Main DEMS dashboard
```
