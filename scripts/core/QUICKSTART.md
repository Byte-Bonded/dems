# IEEE 39-Bus Simulation with Complete Monitoring

## 🎯 Quick Start (3 Steps)

### Step 1: Start Everything
```bash
cd /home/pranaav/Work/projects/SEM6/SETandIEMS/dems/scripts/core
./start_monitoring.sh
```

This single command will:
- ✅ Start Prometheus + Grafana in Docker
- ✅ Import the IEEE 39-Bus dashboard
- ✅ Run the simulation with real-time metrics
- ✅ Keep everything running until you press Ctrl+C

### Step 2: View the Dashboard
Open your browser: **http://localhost:3000**
- Username: `admin`
- Password: `dems2024`
- Dashboard: **IEEE 39-Bus Power System - Complete Dashboard**

### Step 3: Watch Live Data
The dashboard updates every 5 seconds with:
- 📊 Total generation, load, and losses
- ⚡ System frequency (50 Hz nominal)
- 📈 Voltage profile across all 39 buses
- 🔋 DER systems (Solar, Wind, Battery, EV, Demand Response)
- 🎮 Load control demonstration results

---

## 📁 What Was Created

### Core Files in `scripts/core/`

1. **`monitoring_integration.py`** (600+ lines)
   - Prometheus metrics exporter for IEEE 39-Bus
   - ~100+ time-series metrics
   - Monitors all system components

2. **`run_simulation_with_monitoring.py`** (400+ lines)
   - Main script to run simulation with monitoring
   - Two modes: Demo and Continuous
   - Integrates simulation with metrics export

3. **`grafana_dashboard_ieee39.json`** (800+ lines)
   - Complete Grafana dashboard
   - 26 panels with real-time visualization
   - Professional power system monitoring

4. **`start_monitoring.sh`** (Bash script)
   - All-in-one startup script
   - Handles Docker, dashboard import, and simulation

5. **`import_ieee39_dashboard.sh`** (Bash script)
   - Automated dashboard import into Grafana
   - API-based, no manual steps needed

6. **`MONITORING_README.md`** (Comprehensive docs)
   - Complete documentation
   - Architecture diagrams
   - Troubleshooting guide

7. **`test_monitoring_setup.py`** (Quick test)
   - Verify monitoring setup
   - Test without full simulation

---

## 🎮 Simulation Modes

### Demo Mode (Recommended First)
**Duration:** ~30 seconds  
**Purpose:** Load control demonstration

```bash
python3 run_simulation_with_monitoring.py --mode demo
```

What it does:
1. Establishes baseline IEEE 39-Bus operation
2. Applies 100 MW load increase across 5 buses
3. Coordinates DER response:
   - Solar PV reduces curtailment
   - Wind optimizes dispatch
   - Batteries discharge (4x 15 MW)
   - EVs provide V2G power
   - Demand Response reduces load
4. Exports all metrics to Prometheus
5. Shows results in dashboard

### Continuous Mode (For Extended Testing)
**Duration:** Configurable (default: 100 steps × 5s = 8.3 minutes)  
**Purpose:** Ongoing monitoring and analysis

```bash
python3 run_simulation_with_monitoring.py --mode continuous --steps 100 --interval 5
```

What it does:
- Runs power flow analysis every 5 seconds
- Updates all metrics in real-time
- Perfect for testing dashboard and alerts
- Good for demonstration and screenshots

---

## 📊 Dashboard Overview

### 26 Panels in 14 Rows

**Row 1: System Overview**
- System Status (Secure/Alert)
- Total Generation
- Total Load
- System Losses
- Frequency
- Voltage Range

**Row 2-3: Time Series**
- Power Flow (Gen vs Load vs Losses)
- System Frequency
- Voltage Profile

**Row 4: DER Overview**
- DER Generation Mix (pie chart)
- Total DER Contribution
- Generator Performance Table

**Rows 5-12: Detailed DER Monitoring**
- Solar PV generation (2 systems)
- Wind generation (2 systems)
- Battery power and SOC (4 systems)
- EV charging (5 stations)
- Demand Response (5 programs)

**Row 13: Load Control**
- Load Increase Applied
- DER Response Total
- Success Indicator
- Power Flow Convergence

**Row 14: Performance**
- Power Flow Computation Time
- Metrics Update Rate

---

## 🔧 System Architecture

```
IEEE 39-Bus Simulation (scripts/core/)
    ↓
    ↓ [Metrics Export via monitoring_integration.py]
    ↓
Prometheus (Port 9090)
    ↓
    ↓ [PromQL Queries]
    ↓
Grafana (Port 3000)
    ↓
    ↓ [Dashboard: grafana_dashboard_ieee39.json]
    ↓
User Browser (Real-time visualization)
```

### Ports Used
- **9136**: Simulation metrics endpoint
- **9090**: Prometheus web UI
- **3000**: Grafana web UI

---

## 🧪 Test Before Running

```bash
cd /home/pranaav/Work/projects/SEM6/SETandIEMS/dems/scripts/core
python3 test_monitoring_setup.py
```

This will:
1. Check all Python dependencies
2. Create a test monitor
3. Start a test HTTP server on port 9999
4. Export test metrics
5. Verify metrics endpoint works

Press Ctrl+C to stop.

---

## 🐳 Docker Services

The monitoring stack uses Docker Compose:

```bash
# Start services
cd scripts/monitoring
docker compose -f docker-compose.monitoring.yml up -d

# Check status
docker compose -f docker-compose.monitoring.yml ps

# View logs
docker compose -f docker-compose.monitoring.yml logs -f

# Stop services
docker compose -f docker-compose.monitoring.yml down
```

---

## 📈 Key Metrics

### System Metrics
- `ieee39_total_generation_mw`
- `ieee39_total_load_mw`
- `ieee39_total_losses_mw`
- `ieee39_system_frequency_hz`
- `ieee39_voltage_min_pu` / `ieee39_voltage_max_pu`
- `ieee39_system_secure`

### Generator Metrics (10 generators)
- `ieee39_generator_power_mw{bus, gen_type}`
- `ieee39_generator_voltage_pu{bus, gen_type}`

### DER Metrics (18 systems)
- Solar: `ieee39_solar_generation_mw{name, bus}`
- Wind: `ieee39_wind_generation_mw{name, bus}`
- BESS: `ieee39_bess_power_mw{name, bus}`, `ieee39_bess_soc_percent{name, bus}`
- EV: `ieee39_ev_charging_power_mw{name, bus}`, `ieee39_ev_connected_count{name, bus}`
- DR: `ieee39_dr_load_mw{name, bus}`, `ieee39_dr_reduction_mw{name, bus}`

### Load Control Metrics
- `ieee39_load_increase_applied_mw`
- `ieee39_der_response_total_mw`
- `ieee39_load_control_success`

---

## 🔍 Accessing Metrics Directly

### View Raw Metrics
```bash
curl http://localhost:9136/metrics | grep ieee39
```

### Query in Prometheus
1. Open http://localhost:9090
2. Click "Graph"
3. Enter query: `ieee39_total_generation_mw`
4. Click "Execute"

### Example Queries
```promql
# Total generation over time
ieee39_total_generation_mw

# DER contribution percentage
(ieee39_total_der_generation_mw / ieee39_total_generation_mw) * 100

# Voltage violations count
ieee39_voltage_violations

# Frequency deviation
ieee39_frequency_deviation_hz

# Battery fleet SOC average
avg(ieee39_bess_soc_percent)
```

---

## 🛠️ Troubleshooting

### No data in Grafana?

1. **Check if simulation is running:**
   ```bash
   curl http://localhost:9136/metrics | head -20
   ```

2. **Check Prometheus is scraping:**
   - Open http://localhost:9090/targets
   - Look for `ieee39-simulation` target
   - Should show "UP" status

3. **Check Grafana time range:**
   - Set to "Last 5 minutes"
   - Click refresh button

### Prometheus not scraping?

Check `scripts/monitoring/prometheus.yml`:
```yaml
scrape_configs:
  - job_name: 'ieee39-simulation'
    static_configs:
      - targets: ['host.docker.internal:9136']
```

### Dashboard not importing?

```bash
cd scripts/core
./import_ieee39_dashboard.sh
```

Or manually:
1. Open http://localhost:3000
2. Go to Dashboards → Import
3. Upload `grafana_dashboard_ieee39.json`

---

## 🎓 Understanding the Simulation

### IEEE 39-Bus System
- **Standard:** IEEE 39-Bus New England Test System
- **Buses:** 39
- **Generators:** 10 (Bus 30-39)
- **Branches:** 46 transmission lines
- **Frequency:** 50 Hz (converted from 60 Hz standard)
- **Tools:** PyPower (power flow), PandaPower (advanced analysis)

### DER Systems (18 total)
- **Solar PV:** 2 systems (25 MW, 20 MW)
- **Wind:** 2 systems (30 MW, 25 MW)
- **Battery:** 4 systems (4x 20 MW, 80 MWh each)
- **EV Stations:** 5 aggregators (35-50 kW chargers)
- **Demand Response:** 5 programs (15 MW each)

### Load Control Demo
Demonstrates coordinated response to 100 MW load increase:
1. **Load Distribution:** Across buses 20, 21, 23, 24, 27
2. **DER Response:** ~50-60 MW from renewables, storage, V2G
3. **DR Reduction:** ~30-40 MW load reduction
4. **Result:** System remains stable with IEEE compliance

---

## 📚 Additional Documentation

- **Complete monitoring docs:** `MONITORING_README.md`
- **System architecture:** See ARCHITECTURE section above
- **IEEE 39-Bus details:** `ieee39_system_strict.py`
- **Dynamic models:** `dynamic_models.py`
- **Original demo:** `main_demo.py`

---

## 🚀 What's Next?

1. **Run the simulation** and explore the dashboard
2. **Customize metrics** in `monitoring_integration.py`
3. **Add dashboard panels** in Grafana UI (save to JSON after)
4. **Extend DER systems** in `dynamic_models.py`
5. **Add alerts** in Grafana (email, Slack, etc.)
6. **Create reports** using Grafana's reporting feature

---

## ✨ Features

✅ Complete IEEE 39-Bus simulation  
✅ 100+ Prometheus metrics  
✅ Professional Grafana dashboard (26 panels)  
✅ Real-time monitoring (5s updates)  
✅ Load control demonstration  
✅ DER coordination (Solar, Wind, BESS, EV, DR)  
✅ Power flow analysis (PyPower + PandaPower)  
✅ 50 Hz operation (international standard)  
✅ Docker-based deployment  
✅ One-command startup  
✅ Comprehensive documentation  
✅ Test scripts included  

---

**Ready to start?** Run `./start_monitoring.sh` and open http://localhost:3000! 🎉
