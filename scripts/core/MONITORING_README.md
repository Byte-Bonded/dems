# IEEE 39-Bus System Monitoring

Complete monitoring solution for the IEEE 39-Bus power system simulation with real-time metrics export to Prometheus and visualization in Grafana.

## Overview

This monitoring system provides comprehensive observability for the IEEE 39-Bus New England test system with:

- **Power Flow Analysis**: Real-time generation, load, and losses
- **Generator Monitoring**: 10 synchronous generators with performance metrics
- **DER Systems**: 18 distributed energy resources (Solar, Wind, BESS, EV, DR)
- **System Stability**: Voltage profile, frequency monitoring, security indicators
- **Load Control**: Demonstration and metrics for DER-coordinated load response
- **Performance**: Power flow computation time and update rates

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  IEEE 39-Bus Simulation                      │
│          (ieee39_system_strict.py + dynamic_models.py)       │
│                                                              │
│  • 39 buses, 10 generators, 46 branches                     │
│  • 18 DER systems (Solar, Wind, BESS, EV, DR)              │
│  • PyPower & PandaPower integration                         │
│  • 50 Hz operation                                          │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    │ Metrics Export
                    ▼
┌─────────────────────────────────────────────────────────────┐
│           Monitoring Integration Layer                       │
│         (monitoring_integration.py)                          │
│                                                              │
│  • IEEE39BusMonitor class                                   │
│  • ~100+ Prometheus metrics                                 │
│  • Real-time state updates                                  │
│  • HTTP endpoint: :9136/metrics                             │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    │ HTTP Scraping (every 5s)
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    Prometheus                                │
│              (Time-Series Database)                          │
│                                                              │
│  • Port: 9090                                               │
│  • Retention: 15 days                                       │
│  • Scrape interval: 5s                                      │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    │ PromQL Queries
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                      Grafana                                 │
│           (Visualization & Dashboards)                       │
│                                                              │
│  • Port: 3000                                               │
│  • Login: admin/dems2024                                    │
│  • Dashboard: IEEE 39-Bus Complete                          │
│  • 26 panels with real-time data                           │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Start Everything with One Command

```bash
cd scripts/core
chmod +x start_monitoring.sh
./start_monitoring.sh
```

This will:
1. Start Prometheus and Grafana in Docker
2. Import the IEEE 39-Bus dashboard
3. Ask you to select simulation mode (Demo or Continuous)
4. Start the simulation with metrics export

### 2. Access the Dashboard

1. Open your browser: http://localhost:3000
2. Login with `admin` / `dems2024`
3. Navigate to the **IEEE 39-Bus Power System - Complete Dashboard**

### 3. Run Simulation Manually

If you prefer manual control:

```bash
# Start monitoring stack
cd scripts/monitoring
docker compose -f docker-compose.monitoring.yml up -d

# Import dashboard
cd ../core
./import_ieee39_dashboard.sh

# Run simulation - Demo Mode (load control demo)
python3 run_simulation_with_monitoring.py --mode demo

# Or Continuous Mode (ongoing simulation)
python3 run_simulation_with_monitoring.py --mode continuous --steps 100 --interval 5
```

## Files

### Core Simulation Files
- **`ieee39_system_strict.py`**: IEEE 39-Bus system implementation with PyPower/PandaPower
- **`dynamic_models.py`**: IEEE standard dynamic models (generators, DER systems)
- **`main_demo.py`**: Original demo script for load control

### Monitoring Files
- **`monitoring_integration.py`**: Prometheus metrics exporter for IEEE 39-Bus
- **`run_simulation_with_monitoring.py`**: Main script to run simulation with monitoring
- **`grafana_dashboard_ieee39.json`**: Grafana dashboard definition (26 panels)

### Helper Scripts
- **`start_monitoring.sh`**: All-in-one startup script
- **`import_ieee39_dashboard.sh`**: Dashboard import automation

## Metrics

### System Metrics
- `ieee39_total_generation_mw` - Total generation (MW)
- `ieee39_total_load_mw` - Total load (MW)
- `ieee39_total_losses_mw` - System losses (MW)
- `ieee39_system_frequency_hz` - System frequency (Hz)
- `ieee39_voltage_min_pu` / `ieee39_voltage_max_pu` - Voltage range (pu)
- `ieee39_system_secure` - Security indicator (1=secure, 0=insecure)
- `ieee39_power_flow_converged` - Convergence status (1=converged, 0=failed)

### Generator Metrics
- `ieee39_generator_power_mw{bus, gen_type}` - Generator real power
- `ieee39_generator_reactive_mvar{bus, gen_type}` - Generator reactive power
- `ieee39_generator_voltage_pu{bus, gen_type}` - Generator terminal voltage

### DER Metrics

**Solar PV:**
- `ieee39_solar_generation_mw{name, bus}` - Solar generation
- `ieee39_solar_irradiance{name}` - Irradiance (W/m²)

**Wind:**
- `ieee39_wind_generation_mw{name, bus}` - Wind generation
- `ieee39_wind_speed_mps{name}` - Wind speed (m/s)

**Battery Storage:**
- `ieee39_bess_power_mw{name, bus}` - BESS power (+ discharge, - charge)
- `ieee39_bess_soc_percent{name, bus}` - State of charge (%)
- `ieee39_bess_energy_mwh{name, bus}` - Energy stored (MWh)

**Electric Vehicles:**
- `ieee39_ev_charging_power_mw{name, bus}` - EV charging power
- `ieee39_ev_connected_count{name, bus}` - Connected EVs
- `ieee39_ev_v2g_capable_count{name, bus}` - V2G capable EVs

**Demand Response:**
- `ieee39_dr_load_mw{name, bus}` - Current load
- `ieee39_dr_reduction_mw{name, bus}` - Load reduction from baseline

### Load Control Metrics
- `ieee39_load_increase_applied_mw` - Load increase applied (MW)
- `ieee39_der_response_total_mw` - Total DER response (MW)
- `ieee39_load_control_success` - Test success indicator

### Performance Metrics
- `ieee39_power_flow_duration_seconds` - Power flow computation time
- `ieee39_update_cycle_duration_seconds` - Full update cycle time
- `ieee39_metrics_update_count` - Total metrics updates

## Dashboard Panels

The Grafana dashboard includes 26 panels organized into sections:

### 1. System Overview (Row 1)
- System Status (secure/alert indicator)
- Total Generation
- Total Load
- System Losses
- Frequency
- Voltage Range

### 2. Time Series Analysis (Rows 2-3)
- Power Flow (Generation vs Load vs Losses)
- System Frequency trends
- Voltage Profile over time

### 3. DER Overview (Row 4)
- DER Generation Mix (pie chart)
- Total DER Contribution trends
- Generator Performance (table)

### 4. Renewable Generation (Rows 5-6)
- Solar PV Generation by system
- Wind Generation by system

### 5. Energy Storage (Rows 7-8)
- Battery Power (charge/discharge)
- Battery State of Charge

### 6. Electric Vehicles (Rows 9-10)
- EV Charging Stations power
- EV Connected Count and V2G capability

### 7. Demand Response (Rows 11-12)
- DR Load Control
- DR Load Reduction

### 8. Load Control Testing (Row 13)
- Applied Load Increase
- DER Response
- Load Control Success
- Power Flow Convergence

### 9. Performance (Row 14)
- Power Flow Computation Time
- Metrics Update Rate

## Simulation Modes

### Demo Mode
Quick demonstration of load control with DER coordination:
- Establishes baseline operation
- Applies 100 MW load increase across 5 buses
- Coordinates DER response (Solar, Wind, BESS, EV V2G, DR)
- Measures and exports results
- Duration: ~30 seconds

```bash
python3 run_simulation_with_monitoring.py --mode demo
```

### Continuous Mode
Ongoing simulation with periodic power flow analysis:
- Runs for specified number of steps
- Configurable interval between steps
- Continuous metrics export
- Good for testing dashboard and monitoring
- Duration: configurable

```bash
python3 run_simulation_with_monitoring.py --mode continuous --steps 100 --interval 5
```

## Configuration

### Prometheus Scrape Config
Location: `scripts/monitoring/prometheus.yml`

```yaml
scrape_configs:
  - job_name: 'ieee39-simulation'
    static_configs:
      - targets: ['host.docker.internal:9136']
    scrape_interval: 5s
```

### Grafana Data Source
Pre-configured in `scripts/monitoring/grafana/provisioning/datasources/prometheus.yml`

### Metrics Port
Default: 9136
Change with `--port` flag:
```bash
python3 run_simulation_with_monitoring.py --mode demo --port 9999
```

## Requirements

### Python Packages
```
numpy
scipy
pypower
pandapower
prometheus-client
```

Install with:
```bash
pip install numpy scipy pypower pandapower prometheus-client
```

### Docker Services
- Prometheus
- Grafana
- Docker Compose

## Troubleshooting

### Metrics not appearing in Grafana

1. Check if simulation is running:
   ```bash
   curl http://localhost:9136/metrics | head -20
   ```

2. Check if Prometheus is scraping:
   - Open http://localhost:9090/targets
   - Ensure `ieee39-simulation` target is UP

3. Check Grafana data source:
   - Open http://localhost:3000/datasources
   - Test Prometheus connection

### Dashboard shows "No Data"

1. Ensure simulation is running with metrics export
2. Check time range in Grafana (use "Last 5 minutes")
3. Refresh the dashboard
4. Check if metrics exist in Prometheus:
   - Open http://localhost:9090/graph
   - Query: `ieee39_total_generation_mw`

### Power Flow Not Converging

- Check bus data integrity
- Verify generator limits
- Review load distribution
- Check system logs for detailed error messages

## Performance

- **Power Flow Computation**: ~50-200ms per iteration
- **Metrics Update**: ~5-10ms
- **Memory Usage**: ~200-300 MB
- **CPU Usage**: 5-10% during continuous simulation

## Future Enhancements

- [ ] Real-time contingency analysis
- [ ] Economic dispatch optimization
- [ ] State estimation integration
- [ ] PMU data simulation
- [ ] Alerts and notifications
- [ ] Historical data analysis
- [ ] Multi-scenario comparison

## Support

For issues or questions:
1. Check the logs in the terminal
2. Review Prometheus targets: http://localhost:9090/targets
3. Check Grafana dashboard for error messages
4. Verify all Docker containers are running:
   ```bash
   docker ps | grep -E "prometheus|grafana"
   ```

## License

This monitoring system is part of the DEMS (Distributed Energy Management System) project.
