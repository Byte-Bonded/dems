# DEMS Monitoring Setup

Complete Prometheus + Grafana monitoring stack for the 117-bus DEMS SuperGrid simulation.

## Architecture

```
┌─────────────────────┐
│ DEMS Orchestrator   │
│ (orchestrator.py)   │
│                     │
│ - Power Flow        │
│ - Grid State        │
│ - DER Status        │
└──────────┬──────────┘
           │ Exposes metrics
           ▼
┌─────────────────────┐
│ Prometheus Exporter │ :8001/metrics
│ (prometheus_        │
│  exporter.py)       │
└──────────┬──────────┘
           │ Scrapes (every 2s)
           ▼
┌─────────────────────┐
│   Prometheus        │ :9090
│                     │
│ - Time-series DB    │
│ - Metrics storage   │
│ - Alerting          │
└──────────┬──────────┘
           │ Queries
           ▼
┌─────────────────────┐
│    Grafana          │ :3000
│                     │
│ - Dashboards        │
│ - Visualization     │
│ - Real-time graphs  │
└─────────────────────┘
```

## Quick Start

### 1. Start Monitoring Stack

```bash
cd scripts/monitoring
docker-compose -f docker-compose.monitoring.yml up -d
```

This starts:
- **Prometheus** on `http://localhost:9090`
- **Grafana** on `http://localhost:3000`
- **Node Exporter** on `http://localhost:9100`

### 2. Run DEMS Simulation with Monitoring

```bash
cd ../..
python scripts/monitoring/run_with_monitoring.py
```

Or integrate into your code:

```python
from scripts.monitoring.prometheus_exporter import DEMSPrometheusExporter
from src.orchestrator import GridOrchestrator

# Start exporter
exporter = DEMSPrometheusExporter(port=8001)
exporter.start()

# Create orchestrator
orchestrator = GridOrchestrator(log_level="INFO")

# Run simulation and update metrics
result = orchestrator.run_single_power_flow()
state = orchestrator.get_and_log_state()

exporter.update_from_power_flow_result(result)
exporter.update_from_grid_state(state)

# If DER is initialized
if orchestrator.grid.supergrid.der_manager:
    der_status = orchestrator.grid.supergrid.der_manager.get_status()
    exporter.update_from_der_status(der_status)
```

### 3. Access Dashboards

1. **Grafana**: Open `http://localhost:3000`
   - Username: `admin`
   - Password: `dems2024`

2. **Import DEMS Dashboard**:
   - Go to Dashboards → Import
   - Upload `grafana/dashboards/dems-grid-overview.json`

3. **Prometheus**: Open `http://localhost:9090`
   - Check targets: Status → Targets
   - Query metrics: Graph → Execute

## Metrics Exposed

### Power Flow Metrics
- `dems_total_generation_mw` - Total generation (MW)
- `dems_total_load_mw` - Total load (MW)
- `dems_total_losses_mw` - System losses (MW)
- `dems_loss_percentage` - Loss as % of generation

### Voltage Metrics
- `dems_min_voltage_pu` - Minimum bus voltage (p.u.)
- `dems_max_voltage_pu` - Maximum bus voltage (p.u.)
- `dems_avg_voltage_pu` - Average voltage (p.u.)
- `dems_voltage_violations` - Number of violations

### Frequency Metrics
- `dems_system_frequency_hz` - System frequency (Hz)
- `dems_avg_frequency_hz` - Average area frequency (Hz)

### Area Metrics (labels: area=A/B/C)
- `dems_area_generation_mw` - Generation by area
- `dems_area_load_mw` - Load by area
- `dems_area_net_export_mw` - Net export/import

### Tie-Line Metrics (labels: tie_line, from_area, to_area)
- `dems_tie_line_flow_mw` - Power flow on tie-lines
- `dems_tie_line_loading_percent` - Loading percentage

### DER Metrics
- `dems_solar_capacity_mw` / `dems_solar_output_mw`
- `dems_wind_capacity_mw` / `dems_wind_output_mw`
- `dems_battery_capacity_mwh` / `dems_battery_soc_percent`
- `dems_ev_max_power_mw` / `dems_ev_current_power_mw`
- `dems_dr_available_mw` / `dems_dr_curtailed_mw`

### Grid Security
- `dems_grid_secure` - 1=secure, 0=insecure
- `dems_line_overloads` - Number of overloaded lines
- `dems_power_flow_converged` - Convergence status

### Performance
- `dems_power_flow_duration_ms` - Power flow calc time
- `dems_power_flow_iterations` - Iterations to converge
- `dems_simulation_step_total` - Total simulation steps

## Grafana Dashboards

### Main Dashboard: DEMS Grid Overview
- **System Overview**: Generation, Load, Losses, Efficiency
- **Voltage Profile**: Min/Max/Avg voltage with violations
- **Frequency Stability**: Real-time frequency tracking
- **Area Analysis**: Per-area generation and load
- **Tie-Line Flows**: Inter-area power transfers
- **DER Status**: Solar, Wind, Battery, EV, DR
- **Grid Security**: Convergence, violations, alerts

### Custom Queries

Query total DER output:
```promql
dems_solar_output_mw + dems_wind_output_mw + dems_battery_power_mw
```

Query loss percentage over time:
```promql
rate(dems_total_losses_mw[1m]) / rate(dems_total_generation_mw[1m]) * 100
```

Check voltage violations:
```promql
dems_voltage_violations > 0
```

## Configuration Files

- `prometheus.yml` - Prometheus scrape config
- `docker-compose.monitoring.yml` - Docker services
- `grafana/provisioning/datasources/prometheus.yml` - Grafana datasource
- `grafana/dashboards/dems-grid-overview.json` - Main dashboard

## Troubleshooting

### Prometheus can't scrape metrics

1. Check exporter is running:
   ```bash
   curl http://localhost:8001/metrics
   ```

2. Check Prometheus targets:
   - Open `http://localhost:9090/targets`
   - Should show `dems-simulation` as UP

3. For Docker networking issues, use `host.docker.internal:8001` instead of `localhost:8001`

### No data in Grafana

1. Verify Prometheus datasource:
   - Grafana → Configuration → Data Sources
   - Test connection to Prometheus

2. Check metrics are being updated:
   - Run simulation with `run_with_monitoring.py`
   - Verify metrics update in Prometheus

### High memory usage

Reduce retention time in `docker-compose.monitoring.yml`:
```yaml
command:
  - "--storage.tsdb.retention.time=7d"  # Instead of 30d
```

## Advanced Usage

### Custom Alerts

Create `alerts.yml` in this directory:

```yaml
groups:
  - name: grid_alerts
    interval: 10s
    rules:
      - alert: HighVoltageViolations
        expr: dems_voltage_violations > 5
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "High number of voltage violations"
          
      - alert: PowerFlowNotConverged
        expr: dems_power_flow_converged == 0
        for: 30s
        labels:
          severity: critical
        annotations:
          summary: "Power flow failed to converge"
```

### Multiple Simulation Instances

Update `prometheus.yml` to scrape multiple instances:

```yaml
scrape_configs:
  - job_name: 'dems-simulation'
    static_configs:
      - targets: 
          - 'host.docker.internal:8001'
          - 'host.docker.internal:8002'
          - 'host.docker.internal:8003'
        labels:
          grid_type: 'supergrid'
```

## Stopping Monitoring

```bash
docker-compose -f docker-compose.monitoring.yml down

# To remove volumes (deletes all data)
docker-compose -f docker-compose.monitoring.yml down -v
```

## Integration with RL Agent

The RL agent can use these metrics for:
1. **Reward signals**: Based on losses, violations, stability
2. **State observation**: Real-time grid metrics
3. **Training monitoring**: Track agent performance

See `RL_agent.md` for recommendations on RL integration.
