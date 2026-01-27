# DER (Distributed Energy Resources) Integration Guide

## Overview

The DEMS grid includes comprehensive support for five types of Distributed Energy Resources:
- ⚡ **Solar PV** - Variable generation based on solar irradiance
- 🌬️ **Wind Turbines** - Variable generation based on wind speed
- 🔋 **Battery Energy Storage (BESS)** - Controllable charge/discharge with SOC tracking
- 🚗 **EV Charging Stations** - Smart charging with flexible load control  
- 📊 **Demand Response (DR)** - Controllable load curtailment programs

## Quick Start

```python
from src.grid import DEMSGrid

# Create grid and initialize DER
grid = DEMSGrid()
der_manager = grid.supergrid.initialize_der()

# Control different DER types
der_manager.update_solar_generation(hour=12, irradiance=800)   # Solar
der_manager.update_wind_generation(hour=15, wind_speed=8.0)    # Wind
der_manager.set_battery_power("BESS_A1", power_mw=-20)         # Battery discharge
der_manager.update_ev_charging_by_hour(hour=19)                # EV peak charging
der_manager.set_demand_response_curtailment("DR_Industrial_A1", 0.5)  # 50% curtailment

# Run power flow
result = grid.run_power_flow()
print(f"Converged: {result['converged']}")
```

## Default DER Configuration

When you call `grid.supergrid.initialize_der()`, the system adds **23 DER units**:

### Total Resources
- **Solar Generation**: 205 MW
- **Wind Generation**: 250 MW
- **Battery Storage**: 120 MW power / 480 MWh energy
- **EV Charging**: 200 chargers = 15 MW peak load
- **Demand Response**: 110 MW baseline, 36.5 MW curtailable
- **Total Controllable**: 506.5 MW

See full configuration in the documentation.

## API Summary

```python
# Add DER
der_manager.add_solar_pv(bus, capacity_mw, name, area_id)
der_manager.add_wind_turbine(bus, capacity_mw, name, area_id, num_turbines)
der_manager.add_battery(bus, power_mw, energy_mwh, name, area_id)
der_manager.add_ev_charging_station(bus, num_chargers, charger_power_kw, name, area_id)
der_manager.add_demand_response(bus, baseline_load_mw, curtailable_fraction, name, area_id)

# Control DER
der_manager.set_solar_output(name, irradiance_w_m2)
der_manager.set_wind_output(name, wind_speed_m_s)
der_manager.set_battery_power(name, power_mw)
der_manager.set_ev_charging_load(name, utilization, smart_override)
der_manager.set_demand_response_curtailment(name, curtailment_fraction)

# Query State
state = der_manager.get_der_state(name)
all_states = der_manager.get_all_der_states()
```

For complete documentation with examples, see the code docstrings in `src/simulation/der.py`.
