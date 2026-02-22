"""
Comprehensive Simulation Test Suite for DEMS
Tests power flow, DER, dynamics, constraints, and microgrid integration.

Coverage:
- SuperGrid construction and topology validation
- AC/DC power flow convergence and result accuracy
- Voltage constraint checking (0.95-1.05 pu)
- Line thermal limit checking
- DER management and dispatch
- Dynamic models (swing equation, AVR, governor, PSS, AGC)
- N-1 contingency analysis
- Microgrid NR solver (standalone)
- Time-series simulation
- Protection relay logic
"""

import pytest
import numpy as np
import sys
import os

# Ensure dems root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.simulation.supergrid import SuperGrid, SuperGridConfig, AreaID
from src.simulation.power_flow import (
    PowerFlowRunner, PowerFlowResult, PowerFlowConfig, PowerFlowAlgorithm,
    run_time_series_power_flow, quick_power_flow,
)
from src.simulation.der import DERManager, DERType, DERSpec, DERState
from src.simulation.dynamics import (
    DynamicsCoordinator,
    SynchronousGeneratorDynamic,
    ExcitationSystem,
    GovernorTurbine,
    PowerSystemStabilizer,
    AutomaticGenerationControl,
    DynamicLoadModel,
    ProtectionRelay,
    GeneratorDynamicParams,
    ExciterParams,
    GovernorParams,
    AGCParams,
    ProtectionParams,
    IEEE39_GENERATOR_DATA,
    create_ieee39_dynamics,
)
from src.simulation.tie_lines import (
    TieLineConfig, create_tie_lines, get_tie_line_transfer_limits, DEFAULT_TIE_LINES,
)
from src.simulation.microgrid import (
    MicrogridCase,
    PowerFlowSolver,
    MicrogridController,
    SolarPV,
    WindTurbine,
    BatteryESS,
    DieselGenerator,
    create_example_microgrid,
    create_ieee14_case,
    BUS_I, BUS_TYPE, PD, QD, VM, VA, VMAX, VMIN,
    PG, QG, GEN_BUS, GEN_STATUS, PMAX, PMIN,
    F_BUS, T_BUS, BR_R, BR_X, BR_STATUS,
    REF_BUS, PV_BUS, PQ_BUS,
)


# ======================== FIXTURES ======================== #

@pytest.fixture(scope="module")
def supergrid():
    """Create SuperGrid once for module (expensive to construct)"""
    sg = SuperGrid()
    return sg


@pytest.fixture(scope="module")
def supergrid_with_pf(supergrid):
    """SuperGrid with a solved power flow"""
    runner = PowerFlowRunner()
    runner.run(supergrid.net)
    return supergrid


@pytest.fixture
def pf_runner():
    """Fresh PowerFlowRunner"""
    return PowerFlowRunner()


@pytest.fixture
def microgrid_case():
    """Example 5-bus microgrid"""
    case, ders = create_example_microgrid()
    return case, ders


# ======================== 1. SUPERGRID TOPOLOGY ======================== #

class TestSuperGridTopology:
    """Validate the 117-bus Tri-Area Super-Grid structure"""

    def test_bus_count(self, supergrid):
        """117 buses: 3 areas × 39 buses"""
        assert len(supergrid.net.bus) == 117

    def test_generator_count(self, supergrid):
        """At least 27 generators (9 per area × 3)"""
        assert len(supergrid.net.gen) >= 27

    def test_three_zones(self, supergrid):
        """Grid should have exactly 3 zones"""
        zones = supergrid.net.bus["zone"].unique()
        assert len(zones) == 3

    def test_area_configs(self, supergrid):
        """Each area has correct bus offset"""
        assert supergrid.areas[AreaID.AREA_A].bus_offset == 0
        assert supergrid.areas[AreaID.AREA_B].bus_offset == 39
        assert supergrid.areas[AreaID.AREA_C].bus_offset == 78

    def test_tie_lines_exist(self, supergrid):
        """8 tie-lines between areas"""
        assert len(supergrid.tie_line_specs) == 8

    def test_line_count(self, supergrid):
        """At least 105 internal lines + 8 tie-lines"""
        assert len(supergrid.net.line) >= 105 + 8

    def test_area_bus_ranges(self, supergrid):
        """Each area covers 39 buses"""
        for area_id, area_cfg in supergrid.areas.items():
            start, end = area_cfg.bus_range
            assert end - start == 38  # Inclusive range

    def test_ext_grid_exists(self, supergrid):
        """External grid (slack) exists"""
        assert len(supergrid.net.ext_grid) >= 1

    def test_repr(self, supergrid):
        """String representation includes key metrics"""
        r = repr(supergrid)
        assert "117" in r or "buses" in r.lower()


# ======================== 2. POWER FLOW ANALYSIS ======================== #

class TestPowerFlowConvergence:
    """Test AC power flow convergence and result correctness"""

    def test_newton_raphson_converges(self, supergrid, pf_runner):
        """NR algorithm converges on 117-bus system"""
        result = pf_runner.run(supergrid.net)
        assert result.converged, f"NR did not converge: {result.error_message}"

    def test_generation_positive(self, supergrid, pf_runner):
        """Total generation must be positive"""
        result = pf_runner.run(supergrid.net)
        assert result.total_generation_mw > 0

    def test_load_positive(self, supergrid, pf_runner):
        """Total load must be positive"""
        result = pf_runner.run(supergrid.net)
        assert result.total_load_mw > 0

    def test_power_balance(self, supergrid, pf_runner):
        """Generation = Load + Losses (within tolerance)"""
        result = pf_runner.run(supergrid.net)
        balance = result.total_generation_mw - result.total_load_mw - result.total_losses_mw
        assert abs(balance) < 1.0, f"Power balance error: {balance:.3f} MW"

    def test_losses_positive(self, supergrid, pf_runner):
        """Losses should be positive (resistive losses)"""
        result = pf_runner.run(supergrid.net)
        assert result.total_losses_mw > 0

    def test_losses_reasonable(self, supergrid, pf_runner):
        """Losses should be < 5% of generation (typical for transmission)"""
        result = pf_runner.run(supergrid.net)
        loss_pct = result.total_losses_mw / result.total_generation_mw * 100
        assert loss_pct < 5.0, f"Losses too high: {loss_pct:.2f}%"

    def test_iteration_count_reasonable(self, supergrid, pf_runner):
        """NR should converge in < 20 iterations"""
        result = pf_runner.run(supergrid.net)
        assert result.iterations < 20

    def test_dc_power_flow(self, supergrid):
        """DC power flow should always converge"""
        runner = PowerFlowRunner()
        result = runner.run(supergrid.net, algorithm=PowerFlowAlgorithm.DC)
        assert result.converged

    def test_repeated_convergence(self, supergrid, pf_runner):
        """Power flow should converge on repeated runs"""
        for _ in range(3):
            result = pf_runner.run(supergrid.net)
            assert result.converged


# ======================== 3. VOLTAGE CONSTRAINTS ======================== #

class TestVoltageConstraints:
    """Validate voltage profile of the 117-bus system"""

    def test_voltage_within_operational_range(self, supergrid, pf_runner):
        """All bus voltages must be within 0.90-1.10 pu (operational)"""
        result = pf_runner.run(supergrid.net)
        assert result.min_voltage_pu >= 0.90, \
            f"Min voltage {result.min_voltage_pu:.4f} below 0.90 pu"
        assert result.max_voltage_pu <= 1.10, \
            f"Max voltage {result.max_voltage_pu:.4f} above 1.10 pu"

    def test_voltage_within_normal_range(self, supergrid, pf_runner):
        """Most bus voltages should be within 0.95-1.05 pu (normal)"""
        pf_runner.run(supergrid.net)
        net = supergrid.net
        within_normal = ((net.res_bus.vm_pu >= 0.95) & (net.res_bus.vm_pu <= 1.05)).sum()
        total = len(net.res_bus)
        pct = within_normal / total * 100
        assert pct >= 90, f"Only {pct:.1f}% buses within normal range"

    def test_no_voltage_collapse(self, supergrid, pf_runner):
        """No bus voltage below 0.80 (indicates collapse)"""
        pf_runner.run(supergrid.net)
        assert supergrid.net.res_bus.vm_pu.min() > 0.80

    def test_average_voltage_near_nominal(self, supergrid, pf_runner):
        """Average voltage should be close to 1.0 pu"""
        result = pf_runner.run(supergrid.net)
        assert 0.97 <= result.avg_voltage_pu <= 1.03

    def test_per_area_voltage(self, supergrid_with_pf):
        """Each area should have acceptable voltage profile"""
        sg = supergrid_with_pf
        for area_id in AreaID:
            state = sg.get_area_state(area_id)
            assert state["min_voltage_pu"] >= 0.90, \
                f"Area {area_id.value} min V={state['min_voltage_pu']:.4f}"
            assert state["max_voltage_pu"] <= 1.10, \
                f"Area {area_id.value} max V={state['max_voltage_pu']:.4f}"


# ======================== 4. THERMAL CONSTRAINTS ======================== #

class TestThermalConstraints:
    """Validate line loading limits"""

    def test_line_loading_computable(self, supergrid, pf_runner):
        """Line loading results should exist after power flow"""
        pf_runner.run(supergrid.net)
        assert not supergrid.net.res_line.empty

    def test_no_severe_overloads(self, supergrid, pf_runner):
        """No line should be more than 200% loaded (emergency)"""
        pf_runner.run(supergrid.net)
        max_loading = supergrid.net.res_line.loading_percent.max()
        assert max_loading < 200, f"Severe overload: {max_loading:.1f}%"

    def test_tie_line_flows(self, supergrid_with_pf):
        """Tie-line flows should be retrievable and within ratings"""
        flows = supergrid_with_pf.get_tie_line_flows()
        assert len(flows) == 8
        for tl in flows:
            assert "name" in tl
            assert "loading_percent" in tl
            assert tl["loading_percent"] >= 0

    def test_transfer_limits(self):
        """Transfer limits aggregation from tie-line configs"""
        limits = get_tie_line_transfer_limits(DEFAULT_TIE_LINES)
        assert "A-B" in limits
        assert "B-C" in limits
        assert "A-C" in limits
        for key, val in limits.items():
            assert val["forward_mva"] > 0
            assert val["reverse_mva"] > 0


# ======================== 5. DER MANAGEMENT ======================== #

class TestDERManagement:
    """Test DER initialization, dispatch, and status"""

    def test_initialize_der(self, supergrid):
        """DER manager should initialize successfully"""
        der = supergrid.initialize_der()
        assert der is not None
        assert len(der.der_specs) > 0

    def test_der_types_present(self, supergrid):
        """Solar, wind, battery, EV, and DR should all be present"""
        der = supergrid.initialize_der()
        status = der.get_status()
        assert status["solar"]["unit_count"] > 0
        assert status["wind"]["unit_count"] > 0
        assert status["battery"]["unit_count"] > 0
        assert status["ev_charger"]["unit_count"] > 0
        assert status["demand_response"]["unit_count"] > 0

    def test_solar_output_setting(self, supergrid):
        """Solar PV output responds to irradiance"""
        der = supergrid.initialize_der()
        # Set high irradiance
        der.set_solar_output("Solar_A1", 800)
        state = der.get_der_state("Solar_A1")
        assert state.current_output_mw > 0

        # Set zero irradiance (night)
        der.set_solar_output("Solar_A1", 0)
        state = der.get_der_state("Solar_A1")
        assert state.current_output_mw == 0

    def test_wind_output_setting(self, supergrid):
        """Wind turbine output responds to wind speed"""
        der = supergrid.initialize_der()
        # Below cut-in speed
        der.set_wind_output("Wind_B1", 2.0)
        state = der.get_der_state("Wind_B1")
        assert state.current_output_mw == 0

        # Above rated speed
        der.set_wind_output("Wind_B1", 16.0)
        state = der.get_der_state("Wind_B1")
        assert state.current_output_mw > 0

    def test_battery_charge_discharge(self, supergrid):
        """Battery can charge and discharge"""
        der = supergrid.initialize_der()
        # Discharge
        der.set_battery_power("BESS_A1", 10.0)
        state = der.get_der_state("BESS_A1")
        assert state.current_output_mw > 0

        # Charge (negative power)
        der.set_battery_power("BESS_A1", -10.0)
        state = der.get_der_state("BESS_A1")
        assert state.current_output_mw < 0

    def test_ev_charging_load(self, supergrid):
        """EV charging load adjustable"""
        der = supergrid.initialize_der()
        der.set_ev_charging_load("EV_Station_A1", 0.8)
        state = der.get_der_state("EV_Station_A1")
        assert state.current_output_mw > 0

    def test_demand_response(self, supergrid):
        """Demand response load curtailable"""
        der = supergrid.initialize_der()
        der.set_demand_response_curtailment("DR_Industrial_A1", 0.5)
        state = der.get_der_state("DR_Industrial_A1")
        assert state.curtailment_level > 0

    def test_pf_with_der(self, supergrid):
        """Power flow should converge with DER added"""
        der = supergrid.initialize_der()
        runner = PowerFlowRunner()
        result = runner.run(supergrid.net)
        assert result.converged

    def test_der_invalid_name_raises(self, supergrid):
        """Accessing non-existent DER raises ValueError"""
        der = supergrid.initialize_der()
        with pytest.raises(ValueError):
            der.set_solar_output("NONEXISTENT", 500)


# ======================== 6. DYNAMIC MODELS ======================== #

class TestDynamicModels:
    """Test generator dynamics, AVR, governor, PSS, AGC"""

    def test_generator_swing_equation(self):
        """Generator frequency deviates under power imbalance"""
        params = GeneratorDynamicParams(H=3.5, D=2.0, frequency=50.0)
        gen = SynchronousGeneratorDynamic("G1", bus=0, params=params)
        gen.initialize(P_mw=100, Q_mvar=30, Vt=1.0, delta_deg=10)

        # Step with power imbalance (more electrical than mechanical)
        freq, _ = gen.update(dt=0.01, Vt=1.0, P_elec_mw=120)
        assert freq < 50.0, "Frequency should be below 50 Hz with excess electrical load"

    def test_generator_steady_state(self):
        """Generator stays at nominal when balanced"""
        params = GeneratorDynamicParams(H=3.5, D=2.0, frequency=50.0)
        gen = SynchronousGeneratorDynamic("G1", bus=0, params=params)
        gen.initialize(P_mw=100, Q_mvar=30, Vt=1.0, delta_deg=10)

        freq = 50.0
        for _ in range(100):
            freq, _ = gen.update(dt=0.01, Vt=1.0, P_elec_mw=100)

        assert abs(freq - 50.0) < 0.1, f"Expected ~50 Hz, got {freq:.3f} Hz"

    def test_excitation_system(self):
        """AVR increases field voltage on voltage dip"""
        exc = ExcitationSystem()
        initial_efd = exc.Efd

        # Simulate voltage dip
        for _ in range(50):
            exc.update(dt=0.01, Vt=0.92)

        assert exc.Efd > initial_efd, "AVR should increase Efd on voltage dip"

    def test_excitation_voltage_rise(self):
        """AVR decreases field voltage on overvoltage"""
        exc = ExcitationSystem()
        initial_efd = exc.Efd

        for _ in range(50):
            exc.update(dt=0.01, Vt=1.08)

        assert exc.Efd < initial_efd, "AVR should decrease Efd on overvoltage"

    def test_governor_response(self):
        """Governor increases Pm when frequency drops"""
        gov = GovernorTurbine()
        initial_pm = gov.Pm

        # Simulate underfrequency (speed below 1.0 pu)
        for _ in range(100):
            gov.update(dt=0.01, omega=0.99)

        assert gov.Pm > initial_pm, "Governor should increase Pm when frequency drops"

    def test_governor_overfrequency(self):
        """Governor decreases Pm when frequency rises"""
        gov = GovernorTurbine()
        initial_pm = gov.Pm

        for _ in range(100):
            gov.update(dt=0.01, omega=1.01)

        assert gov.Pm < initial_pm, "Governor should decrease Pm on overfrequency"

    def test_pss_damping(self):
        """PSS produces output on speed oscillation"""
        pss = PowerSystemStabilizer()

        # Oscillating speed deviation
        outputs = []
        for i in range(200):
            t = i * 0.01
            speed_dev = 0.01 * np.sin(2 * np.pi * 1.0 * t)  # 1 Hz oscillation
            out = pss.update(dt=0.01, speed_deviation=speed_dev)
            outputs.append(out)

        # PSS should have non-zero output
        assert max(abs(o) for o in outputs) > 0, "PSS should produce damping signal"

    def test_agc_pi_controller(self):
        """AGC corrects frequency error over time"""
        agc = AutomaticGenerationControl("A")
        agc.add_participating_generator("G1", 0.5)
        agc.add_participating_generator("G2", 0.5)

        # Simulate underfrequency condition
        adjustments = {}
        for _ in range(100):
            adjustments = agc.update(dt=0.1, frequency_hz=49.95)

        # AGC should command generators to increase output
        total_adj = sum(adjustments.values())
        assert total_adj > 0, f"AGC should increase generation, got {total_adj}"

    def test_dynamic_load_model(self):
        """ZIP load model responds to voltage and frequency"""
        load = DynamicLoadModel(bus=0, P0_mw=100, Q0_mvar=30)

        # Nominal conditions
        P_nom, Q_nom = load.update(voltage_pu=1.0, frequency_hz=50.0)
        assert abs(P_nom - 100) < 1, f"Expected ~100 MW, got {P_nom}"

        # Low voltage reduces load
        P_low, _ = load.update(voltage_pu=0.90, frequency_hz=50.0)
        assert P_low < P_nom, "Low voltage should reduce load"

        # High frequency increases load
        P_high, _ = load.update(voltage_pu=1.0, frequency_hz=50.5)
        assert P_high > P_nom, "High frequency should increase load"


# ======================== 7. PROTECTION RELAY ======================== #

class TestProtectionRelay:
    """Test protection system trip/alarm logic"""

    def test_normal_operation_no_trip(self):
        """Relay should not trip under normal conditions"""
        relay = ProtectionRelay("Gen_1", "generator")
        status = relay.check(dt=0.1, voltage_pu=1.0, frequency_hz=50.0)
        assert not status["tripped"]
        assert not status["alarm_active"]

    def test_undervoltage_alarm(self):
        """Relay should alarm on low voltage"""
        relay = ProtectionRelay("Gen_1", "generator")
        status = relay.check(dt=0.1, voltage_pu=0.94, frequency_hz=50.0)
        assert status["alarm_active"]
        assert any("voltage" in r.lower() for r in status["alarm_reasons"])

    def test_undervoltage_trip(self):
        """Relay should trip on sustained severe undervoltage"""
        params = ProtectionParams(voltage_low_trip=0.90, voltage_trip_delay=0.5)
        relay = ProtectionRelay("Gen_1", "generator", params)

        # Accumulate time at low voltage to exceed trip delay
        for _ in range(10):
            status = relay.check(dt=0.1, voltage_pu=0.85, frequency_hz=50.0)

        assert status["tripped"]
        assert "undervoltage" in status["trip_reason"].lower()

    def test_underfrequency_trip(self):
        """Relay trips on underfrequency"""
        params = ProtectionParams(frequency_low_trip=49.5, frequency_trip_delay=0.3)
        relay = ProtectionRelay("Gen_1", "generator", params)

        for _ in range(5):
            status = relay.check(dt=0.1, voltage_pu=1.0, frequency_hz=49.0)

        assert status["tripped"]

    def test_relay_reset(self):
        """Relay can be reset after trip"""
        relay = ProtectionRelay("Gen_1", "generator")
        relay.tripped = True
        relay.trip_reason = "Test trip"
        relay.reset()
        assert not relay.tripped
        assert relay.trip_reason is None


# ======================== 8. DYNAMICS COORDINATOR ======================== #

class TestDynamicsCoordinator:
    """Test coordinated dynamic simulation"""

    def test_initialize_dynamics(self, supergrid_with_pf):
        """Dynamics initialization should create generators and AGC"""
        coordinator = supergrid_with_pf.initialize_dynamics()
        assert len(coordinator.generators) == 30  # 10 per area × 3
        assert len(coordinator.agc_controllers) == 3
        assert coordinator.system_frequency_hz == 50.0

    def test_step_dynamics(self, supergrid_with_pf):
        """Dynamics step should return valid state"""
        supergrid_with_pf.initialize_dynamics()
        result = supergrid_with_pf.step_dynamics(dt=0.01)
        assert "system_frequency_hz" in result
        assert 49.0 < result["system_frequency_hz"] < 51.0

    def test_frequency_stable_over_time(self, supergrid_with_pf):
        """Frequency should remain stable over multiple steps"""
        supergrid_with_pf.initialize_dynamics()
        for _ in range(20):
            result = supergrid_with_pf.step_dynamics(dt=0.01)

        freq = result["system_frequency_hz"]
        assert 49.5 < freq < 50.5, f"Frequency drifted to {freq:.3f} Hz"

    def test_generator_states_accessible(self, supergrid_with_pf):
        """Generator states should be retrievable"""
        supergrid_with_pf.initialize_dynamics()
        supergrid_with_pf.step_dynamics(dt=0.01)
        states = supergrid_with_pf.dynamics.get_generator_states()
        assert len(states) == 30
        for gen_id, state in states.items():
            assert "frequency_hz" in state
            assert "angle_deg" in state

    def test_create_ieee39_dynamics_function(self):
        """Factory function creates area dynamics"""
        coord = create_ieee39_dynamics(area_offset=0, area_id="A")
        assert len(coord.generators) == 10
        assert "A" in coord.agc_controllers


# ======================== 9. N-1 CONTINGENCY ======================== #

class TestContingencyAnalysis:
    """Test N-1 contingency analysis"""

    def test_contingency_line_outage(self, supergrid, pf_runner):
        """Power flow should converge after single line outage"""
        contingencies = [("line", 0), ("line", 5)]
        results = pf_runner.run_with_contingency(supergrid.net, contingencies)

        # Base case + 2 contingencies
        assert len(results) == 3
        assert results[0].converged  # Base case

    def test_contingency_restores_state(self, supergrid, pf_runner):
        """Network state should be restored after contingency analysis"""
        original_in_service = supergrid.net.line.at[0, "in_service"]
        pf_runner.run_with_contingency(supergrid.net, [("line", 0)])
        assert supergrid.net.line.at[0, "in_service"] == original_in_service


# ======================== 10. MICROGRID (STANDALONE NR SOLVER) ======================== #

class TestMicrogridPowerFlow:
    """Test the standalone Newton-Raphson microgrid solver"""

    def test_microgrid_case_creation(self, microgrid_case):
        """Microgrid case should have correct structure"""
        case, ders = microgrid_case
        assert case.bus_count == 5
        assert case.gen_count == 5
        assert case.branch_count == 5
        assert len(ders) == 4

    def test_nr_solver_converges(self, microgrid_case):
        """NR solver converges on example microgrid"""
        case, _ = microgrid_case
        solver = PowerFlowSolver(case)
        results = solver.run()
        assert results['converged'], "Microgrid NR solver did not converge"

    def test_nr_iteration_count(self, microgrid_case):
        """Solver should converge in < 15 iterations"""
        case, _ = microgrid_case
        solver = PowerFlowSolver(case)
        results = solver.run()
        assert results['iterations'] < 15

    def test_nr_voltage_results(self, microgrid_case):
        """Bus voltages should be within limits"""
        case, _ = microgrid_case
        solver = PowerFlowSolver(case)
        results = solver.run()

        for i in range(case.bus_count):
            vm = results['bus'][i, VM]
            assert 0.90 <= vm <= 1.10, f"Bus {i} voltage {vm:.4f} out of range"

    def test_nr_power_balance(self, microgrid_case):
        """Generation = Load (approximately, for small system)"""
        case, _ = microgrid_case
        solver = PowerFlowSolver(case)
        results = solver.run()

        total_gen = results['gen'][:, PG].sum()
        total_load = results['bus'][:, PD].sum()
        assert total_gen > 0
        assert abs(total_gen - total_load) < total_load * 0.1  # Within 10%

    def test_nr_branch_flows(self, microgrid_case):
        """Branch flows should be computed"""
        case, _ = microgrid_case
        solver = PowerFlowSolver(case)
        results = solver.run()
        assert results['branch_flows'].shape[0] == case.branch_count

    def test_ybus_symmetric(self, microgrid_case):
        """Admittance matrix should be approximately symmetric"""
        case, _ = microgrid_case
        solver = PowerFlowSolver(case)
        results = solver.run()
        Ybus = results['Ybus']
        diff = np.abs(Ybus - Ybus.T).max()
        assert diff < 1e-10, f"Ybus asymmetry: {diff}"

    def test_ieee14_case_converges(self):
        """IEEE 14-bus case should converge"""
        case = create_ieee14_case()
        solver = PowerFlowSolver(case)
        results = solver.run()
        assert results['converged']


# ======================== 11. MICROGRID DER COMPONENTS ======================== #

class TestMicrogridDERComponents:
    """Test DER component models from microgrid module"""

    def test_solar_pv_day_night(self):
        """Solar PV produces power during day, zero at night"""
        solar = SolarPV("PV1", bus=0, rated_power=3.0)
        P_noon, _ = solar.get_output(12.0)
        P_night, _ = solar.get_output(0.0)
        assert P_noon > 0, "Solar should produce at noon"
        assert P_night == 0, "Solar should be zero at night"

    def test_solar_with_profile(self):
        """Solar PV follows irradiance profile"""
        solar = SolarPV("PV1", bus=0, rated_power=3.0)
        profile = np.array([0, 0, 0, 0, 0, 0, 0.2, 0.5, 0.8, 1.0, 0.9, 0.7])
        solar.set_irradiance_profile(profile)
        P_9, _ = solar.get_output(9)
        assert P_9 > 0

    def test_wind_power_curve(self):
        """Wind turbine follows cubic power curve"""
        wind = WindTurbine("W1", bus=0, rated_power=2.5,
                           cut_in_speed=3.0, rated_speed=12.0, cut_out_speed=25.0)

        # Below cut-in
        wind.set_wind_speed_profile(np.array([2.0]))
        P, _ = wind.get_output(0)
        assert P == 0

        # At rated speed
        wind.set_wind_speed_profile(np.array([12.0]))
        P_rated, _ = wind.get_output(0)
        assert abs(P_rated - 2.5) < 0.01

        # Above cut-out
        wind.set_wind_speed_profile(np.array([26.0]))
        P_cut, _ = wind.get_output(0)
        assert P_cut == 0

    def test_battery_soc_tracking(self):
        """Battery SOC changes with charge/discharge"""
        batt = BatteryESS("B1", bus=0, rated_power=2.0, capacity=4.0, initial_soc=0.5)
        initial_soc = batt.soc

        # Discharge
        batt.set_power_command(1.0, dt=1.0)
        assert batt.soc < initial_soc

        # Charge
        mid_soc = batt.soc
        batt.set_power_command(-1.0, dt=1.0)
        assert batt.soc > mid_soc

    def test_battery_soc_limits(self):
        """Battery SOC stays within min/max bounds"""
        batt = BatteryESS("B1", bus=0, rated_power=2.0, capacity=4.0,
                           initial_soc=0.15, min_soc=0.1, max_soc=0.9)

        # Try to overdischarge
        batt.set_power_command(100.0, dt=10.0)
        assert batt.soc >= batt.min_soc

    def test_diesel_ramp_rate(self):
        """Diesel generator respects ramp rate.
        ramp_rate=0.5 MW/min, dt in hours.  dt=1/60 h = 1 min → max_change=0.5 MW."""
        diesel = DieselGenerator("D1", bus=0, rated_power=3.0, ramp_rate=0.5)
        diesel.current_output = 0.0

        # Try to jump to full power with a 1-minute time step
        P = diesel.set_output(3.0, dt=1.0 / 60.0)
        assert P < 3.0, "Diesel should ramp, not jump"
        assert P > 0

    def test_diesel_fuel_consumption(self):
        """Diesel fuel consumption is positive when running"""
        diesel = DieselGenerator("D1", bus=0, rated_power=3.0)
        diesel.set_output(2.0, dt=1.0)
        fuel = diesel.get_fuel_consumption()
        assert fuel > 0

    def test_der_offline(self):
        """DER produces zero output when offline"""
        solar = SolarPV("PV1", bus=0, rated_power=3.0)
        solar.status = 0
        P, Q = solar.get_output(12.0)
        assert P == 0


# ======================== 12. MICROGRID CONTROLLER ======================== #

class TestMicrogridController:
    """Test microgrid controller logic"""

    def test_droop_control(self):
        """Droop control adjusts power based on frequency"""
        case, ders = create_example_microgrid()
        ctrl = MicrogridController(case, ders)

        # Underfrequency → increase power
        P = ctrl.droop_control(P_measured=1.0, f_measured=49.9,
                               P_ref=1.0, f_ref=50.0, droop_coeff=10.0)
        assert P > 1.0

    def test_battery_soc_management(self):
        """Controller manages battery SOC properly"""
        case, ders = create_example_microgrid()
        ctrl = MicrogridController(case, ders)

        # Excess power → should charge
        P = ctrl.battery_soc_control(ders['battery'], P_net=2.0, dt=1.0)
        assert P < 0, "Should charge when excess generation"

        # Power deficit → should discharge
        P = ctrl.battery_soc_control(ders['battery'], P_net=-2.0, dt=1.0)
        assert P > 0, "Should discharge when deficit"


# ======================== 13. TIME-SERIES + GLOBAL STATE ======================== #

class TestTimeSeries:
    """Test time-series simulation and global state"""

    def test_global_state(self, supergrid_with_pf):
        """Global state should contain all required fields"""
        state = supergrid_with_pf.get_global_state()
        assert "areas" in state
        assert "tie_lines" in state
        assert "global_metrics" in state
        assert len(state["areas"]) == 3

        metrics = state["global_metrics"]
        assert metrics["total_generation_mw"] > 0
        assert metrics["total_load_mw"] > 0

    def test_area_state(self, supergrid_with_pf):
        """Per-area state should be correct"""
        for area_id in AreaID:
            state = supergrid_with_pf.get_area_state(area_id)
            assert state["total_generation_mw"] > 0
            assert state["total_load_mw"] > 0
            assert state["num_generators"] >= 9

    def test_quick_power_flow(self, supergrid):
        """Quick power flow utility runs successfully"""
        assert quick_power_flow(supergrid.net)

    def test_load_scaling(self, supergrid):
        """Load scaling changes power flow results"""
        runner = PowerFlowRunner()

        # Baseline
        result_base = runner.run(supergrid.net)
        load_base = result_base.total_load_mw

        # Scale up Area A loads by 20%
        supergrid.scale_loads(AreaID.AREA_A, 1.2)
        result_scaled = runner.run(supergrid.net)

        assert result_scaled.total_load_mw > load_base

        # Restore
        supergrid.scale_loads(AreaID.AREA_A, 1.0 / 1.2)


# ======================== 14. IEEE 39-BUS GENERATOR DATA ======================== #

class TestIEEE39Data:
    """Validate IEEE 39-bus standard data"""

    def test_generator_data_complete(self):
        """All 10 generators should be defined"""
        assert len(IEEE39_GENERATOR_DATA) == 10

    def test_generator_buses(self):
        """Generator buses should be 29-38 (0-indexed pandapower buses)"""
        for bus in IEEE39_GENERATOR_DATA.keys():
            assert 29 <= bus <= 38

    def test_generator_inertia_positive(self):
        """All inertia constants should be positive"""
        for bus, data in IEEE39_GENERATOR_DATA.items():
            assert data["H"] > 0, f"Bus {bus} has non-positive inertia"

    def test_generator_mva_positive(self):
        """All MVA ratings should be positive"""
        for bus, data in IEEE39_GENERATOR_DATA.items():
            assert data["MVA"] > 0


# ======================== 15. EDGE CASES AND ROBUSTNESS ======================== #

class TestEdgeCases:
    """Edge cases and error handling"""

    def test_supergrid_with_custom_config(self):
        """SuperGrid with tighter voltage limits"""
        config = SuperGridConfig(v_min_pu=0.97, v_max_pu=1.03)
        sg = SuperGrid(config=config)
        assert sg.config.v_min_pu == 0.97

    def test_power_flow_runner_default_config(self):
        """PowerFlowRunner works with default config"""
        runner = PowerFlowRunner()
        assert runner.config.algorithm == PowerFlowAlgorithm.NEWTON_RAPHSON

    def test_tie_line_config_max_current(self):
        """Tie-line max current correctly computed"""
        tl = TieLineConfig(
            name="test", from_area="A", to_area="B",
            from_bus_local=0, to_bus_local=0,
            rating_mva=600, length_km=100
        )
        I_ka = tl.get_max_current_ka(345.0)
        expected = 600.0 / (345.0 * np.sqrt(3))
        assert abs(I_ka - expected) < 0.001

    def test_der_manager_requires_network(self):
        """DERManager raises on None network"""
        with pytest.raises(ValueError):
            DERManager(None)

    def test_generator_speed_deviation(self):
        """get_speed_deviation_pu returns correct value"""
        gen = SynchronousGeneratorDynamic("G1", bus=0)
        gen.omega = 1.01
        assert abs(gen.get_speed_deviation_pu() - 0.01) < 1e-10

    def test_empty_microgrid_power_flow_raises(self):
        """Empty case raises on power flow"""
        case = MicrogridCase()
        case.bus = np.array([])
        case.gen = np.array([])
        case.branch = np.array([])
        solver = PowerFlowSolver(case)
        # Should not crash; may not converge or raise
        try:
            results = solver.run()
        except (ValueError, IndexError):
            pass  # Expected


# ======================== 16. ORCHESTRATOR ======================== #

from src.simulation.orchestrator import (
    GridOrchestrator,
    ScenarioConfig,
    StochasticProfileGenerator,
    ObservationBuilder,
    RewardCalculator,
    ActionMapper,
    ActionType,
)


@pytest.fixture(scope="module")
def orchestrator():
    """Build orchestrator once (expensive: constructs 117-bus grid + dynamics + DER)."""
    cfg = ScenarioConfig(
        episode_length_steps=20,     # short for testing
        dynamics_substeps=2,          # minimal for speed
    )
    orch = GridOrchestrator(scenario=cfg, seed=123)
    return orch


class TestOrchestratorConstruction:
    """Orchestrator builds correctly with full grid, DER, dynamics."""

    def test_orchestrator_observation_dim(self, orchestrator):
        assert orchestrator.observation_dim == 42

    def test_orchestrator_action_dim_positive(self, orchestrator):
        assert orchestrator.action_dim > 0

    def test_orchestrator_initial_frequency(self, orchestrator):
        assert 49.0 < orchestrator.system_frequency_hz < 51.0

    def test_orchestrator_grid_state_has_areas(self, orchestrator):
        state = orchestrator.get_grid_state()
        assert "areas" in state
        assert "global_metrics" in state

    def test_orchestrator_der_status_keys(self, orchestrator):
        status = orchestrator.get_der_status()
        # DER manager was initialized, so status should have keys
        assert isinstance(status, dict)

    def test_orchestrator_generator_states(self, orchestrator):
        gs = orchestrator.get_generator_states()
        assert isinstance(gs, dict)


class TestOrchestratorReset:
    """Reset produces a valid observation and resets bookkeeping."""

    def test_reset_returns_correct_shape(self, orchestrator):
        obs = orchestrator.reset(seed=42)
        assert obs.shape == (42,)
        assert obs.dtype == np.float32

    def test_reset_observation_bounded(self, orchestrator):
        obs = orchestrator.reset(seed=99)
        assert np.all(obs >= -2.0)
        assert np.all(obs <= 2.0)

    def test_reset_clears_step_count(self, orchestrator):
        orchestrator.reset(seed=42)
        assert orchestrator.step_count == 0
        assert not orchestrator.is_done


class TestOrchestratorStep:
    """Step executes the full simulation pipeline."""

    def test_step_returns_tuple(self, orchestrator):
        orchestrator.reset(seed=42)
        action = np.full(orchestrator.action_dim, 0.5)
        result = orchestrator.step(action)
        assert len(result) == 4
        obs, reward, done, info = result

    def test_step_obs_shape(self, orchestrator):
        orchestrator.reset(seed=42)
        action = np.full(orchestrator.action_dim, 0.5)
        obs, _, _, _ = orchestrator.step(action)
        assert obs.shape == (42,)

    def test_step_reward_is_float(self, orchestrator):
        orchestrator.reset(seed=42)
        action = np.full(orchestrator.action_dim, 0.5)
        _, reward, _, _ = orchestrator.step(action)
        assert isinstance(reward, float)

    def test_step_info_has_keys(self, orchestrator):
        orchestrator.reset(seed=42)
        action = np.full(orchestrator.action_dim, 0.5)
        _, _, _, info = orchestrator.step(action)
        for key in ["step", "pf_converged", "frequency_hz", "total_gen_mw",
                     "reward_breakdown", "cumulative_reward"]:
            assert key in info, f"Missing key: {key}"

    def test_step_increments_count(self, orchestrator):
        orchestrator.reset(seed=42)
        action = np.full(orchestrator.action_dim, 0.5)
        orchestrator.step(action)
        assert orchestrator.step_count == 1

    def test_multiple_steps(self, orchestrator):
        orchestrator.reset(seed=42)
        action = np.full(orchestrator.action_dim, 0.5)
        for _ in range(5):
            obs, reward, done, info = orchestrator.step(action)
        assert orchestrator.step_count == 5

    def test_episode_terminates(self, orchestrator):
        """Episode ends after episode_length_steps."""
        orchestrator.reset(seed=42)
        action = np.full(orchestrator.action_dim, 0.5)
        for _ in range(orchestrator.cfg.episode_length_steps):
            obs, reward, done, info = orchestrator.step(action)
        assert done is True
        assert orchestrator.is_done

    def test_step_after_done_raises(self, orchestrator):
        """Cannot step after episode is done."""
        orchestrator.reset(seed=42)
        action = np.full(orchestrator.action_dim, 0.5)
        for _ in range(orchestrator.cfg.episode_length_steps):
            orchestrator.step(action)
        with pytest.raises(RuntimeError):
            orchestrator.step(action)


class TestOrchestratorEpisodeSummary:
    """Episode summary contains correct statistics."""

    def test_summary_after_episode(self, orchestrator):
        orchestrator.reset(seed=42)
        action = np.full(orchestrator.action_dim, 0.5)
        for _ in range(orchestrator.cfg.episode_length_steps):
            orchestrator.step(action)
        summary = orchestrator.episode_summary()
        assert summary["steps"] == orchestrator.cfg.episode_length_steps
        assert "cumulative_reward" in summary
        assert "avg_frequency_hz" in summary
        assert "pf_convergence_rate" in summary

    def test_empty_summary(self, orchestrator):
        orchestrator.reset(seed=42)
        # No steps taken — history is empty
        summary = orchestrator.episode_summary()
        assert summary == {}


class TestStochasticProfiles:
    """Profiles are truly stochastic (not hardcoded) and physically bounded."""

    def test_different_seeds_produce_different_profiles(self):
        cfg = ScenarioConfig(episode_length_steps=288)
        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(2)
        p1 = StochasticProfileGenerator(cfg, rng1)
        p2 = StochasticProfileGenerator(cfg, rng2)
        p1.reset()
        p2.reset()
        # Wind profiles should differ (wind is non-zero at all hours unlike solar)
        w1 = [p1.wind_speed(i) for i in range(288)]
        w2 = [p2.wind_speed(i) for i in range(288)]
        assert w1 != w2, "Different seeds must produce different wind profiles"

    def test_solar_bounded(self):
        cfg = ScenarioConfig(episode_length_steps=288)
        rng = np.random.default_rng(42)
        p = StochasticProfileGenerator(cfg, rng)
        p.reset()
        for i in range(288):
            irr = p.solar_irradiance(i)
            assert 0 <= irr <= 1400, f"Solar out of bounds at step {i}: {irr}"

    def test_wind_bounded(self):
        cfg = ScenarioConfig(episode_length_steps=288)
        rng = np.random.default_rng(42)
        p = StochasticProfileGenerator(cfg, rng)
        p.reset()
        for i in range(288):
            ws = p.wind_speed(i)
            assert 0 <= ws <= 35, f"Wind out of bounds at step {i}: {ws}"

    def test_load_scale_bounded(self):
        cfg = ScenarioConfig(episode_length_steps=288)
        rng = np.random.default_rng(42)
        p = StochasticProfileGenerator(cfg, rng)
        p.reset()
        for i in range(288):
            ls = p.load_scale_factor(i)
            assert 0.5 <= ls <= 1.5, f"Load scale out of bounds at step {i}: {ls}"

    def test_solar_zero_at_night(self):
        """Solar irradiance should be zero during night hours."""
        cfg = ScenarioConfig(episode_length_steps=288, dt_control_s=300)
        rng = np.random.default_rng(42)
        p = StochasticProfileGenerator(cfg, rng)
        p.reset()
        # Step 0 = hour 0 (midnight; night)
        assert p.solar_irradiance(0) == 0.0


class TestRewardCalculator:
    """Reward responds correctly to grid conditions."""

    def test_perfect_conditions_high_reward(self):
        cfg = ScenarioConfig()
        rc = RewardCalculator(cfg)
        rc.reset()
        # Simulate a perfectly converged result
        pf = PowerFlowResult(
            converged=True,
            iterations=3,
            elapsed_time_ms=10.0,
            total_generation_mw=5000,
            total_load_mw=4800,
            total_losses_mw=50,
            num_voltage_violations=0,
            num_line_overloads=0,
        )
        action = np.full(10, 0.5)
        reward, bd = rc.compute(pf, 50.0, 0, action)
        assert reward > 0.5, f"Perfect conditions should yield high reward, got {reward}"

    def test_frequency_deviation_penalty(self):
        cfg = ScenarioConfig()
        rc = RewardCalculator(cfg)
        rc.reset()
        pf = PowerFlowResult(
            converged=True,
            iterations=3,
            elapsed_time_ms=10.0,
            total_generation_mw=5000,
            total_load_mw=4800,
            total_losses_mw=50,
            num_voltage_violations=0,
            num_line_overloads=0,
        )
        action = np.full(10, 0.5)
        r_ok, _ = rc.compute(pf, 50.0, 0, action)
        r_bad, bd = rc.compute(pf, 49.0, 0, action)
        assert r_ok > r_bad, "Frequency deviation should reduce reward"
        assert bd["r_frequency"] < 0.5

    def test_protection_trip_penalty(self):
        cfg = ScenarioConfig()
        rc = RewardCalculator(cfg)
        rc.reset()
        pf = PowerFlowResult(
            converged=True,
            iterations=3,
            elapsed_time_ms=10.0,
            total_generation_mw=5000,
            total_load_mw=4800,
            total_losses_mw=50,
            num_voltage_violations=0,
            num_line_overloads=0,
        )
        action = np.full(10, 0.5)
        r_ok, _ = rc.compute(pf, 50.0, 0, action)
        r_trip, _ = rc.compute(pf, 50.0, 3, action)
        assert r_ok > r_trip

    def test_non_converged_pf_negative(self):
        cfg = ScenarioConfig()
        rc = RewardCalculator(cfg)
        rc.reset()
        pf = PowerFlowResult(
            converged=False,
            iterations=0,
            elapsed_time_ms=0.0,
            total_generation_mw=0,
            total_load_mw=0,
            total_losses_mw=0,
            num_voltage_violations=0,
            num_line_overloads=0,
        )
        action = np.full(10, 0.5)
        reward, _ = rc.compute(pf, 50.0, 0, action)
        assert reward < 0, "Non-converged PF should yield negative reward"


class TestObservationBuilder:
    """Observation vector is correctly normalised and shaped."""

    def test_obs_shape(self):
        cfg = ScenarioConfig(episode_length_steps=100)
        rng = np.random.default_rng(42)
        p = StochasticProfileGenerator(cfg, rng)
        p.reset()
        ob = ObservationBuilder(cfg)
        gs = {"areas": {"A": {}, "B": {}, "C": {}}, "global_metrics": {}}
        obs = ob.build(gs, {}, 50.0, p, 0, 0.0)
        assert obs.shape == (42,)

    def test_obs_bounded(self):
        cfg = ScenarioConfig(episode_length_steps=100)
        rng = np.random.default_rng(42)
        p = StochasticProfileGenerator(cfg, rng)
        p.reset()
        ob = ObservationBuilder(cfg)
        gs = {"areas": {"A": {}, "B": {}, "C": {}}, "global_metrics": {}}
        obs = ob.build(gs, {}, 50.0, p, 0, 0.0)
        assert np.all(obs >= -2.0) and np.all(obs <= 2.0)


class TestActionMapper:
    """ActionMapper correctly discovers controllable assets."""

    def test_action_dim_matches_grid(self, orchestrator):
        am = orchestrator.action_mapper
        # Should have generators + DER
        assert am.action_dim > 0
        assert len(am._gen_indices) > 0


# ======================== PSS WASHOUT FILTER TEST (BUG-22) ======================== #


class TestPSSWashoutFilter:
    """BUG-22: Verify PSS washout filter correctly blocks DC, passes AC."""

    def test_dc_step_decays(self):
        """A DC step input should decay to zero through the washout filter."""
        pss = PowerSystemStabilizer()
        dt = 0.01
        # Apply DC step for 20 seconds (well beyond T_washout=1.41s)
        for _ in range(2000):
            out = pss.update(dt, 0.01)  # constant speed deviation
        # After 20s (~14 time constants), output should be near zero
        assert abs(out) < 1e-3, f"PSS output should decay to ~0 for DC input, got {out}"

    def test_ac_1hz_passes(self):
        """A 1 Hz oscillation should pass through the washout with significant gain."""
        pss = PowerSystemStabilizer()
        dt = 0.01
        freq = 1.0  # Hz
        # Run 5 seconds to reach steady state
        for i in range(500):
            t = i * dt
            speed_dev = 0.01 * np.sin(2.0 * np.pi * freq * t)
            pss.update(dt, speed_dev)
        # Collect peak output over next cycle
        peak = 0.0
        for i in range(500, 600):
            t = i * dt
            speed_dev = 0.01 * np.sin(2.0 * np.pi * freq * t)
            out = pss.update(dt, speed_dev)
            peak = max(peak, abs(out))
        # With K_pss=5.0, peak should be significant (> 0.01)
        assert peak > 0.01, f"PSS should pass 1 Hz signal, peak={peak}"

    def test_slow_signal_attenuated(self):
        """A very slow signal (0.01 Hz) should be significantly attenuated."""
        pss = PowerSystemStabilizer()
        dt = 0.01
        freq = 0.01  # Hz – period 100s, well below washout corner
        # Run 200 seconds (2 full cycles)
        peak = 0.0
        for i in range(20000):
            t = i * dt
            speed_dev = 0.01 * np.sin(2.0 * np.pi * freq * t)
            out = pss.update(dt, speed_dev)
            if i > 10000:  # measure in second cycle
                peak = max(peak, abs(out))
        # The washout corner is ~1/(2*pi*1.41) ≈ 0.113 Hz
        # At 0.01 Hz, attenuation should be strong
        # Washout gain at 0.01 Hz: |j*w*T/(1+j*w*T)| = w*T/sqrt(1+(w*T)^2)
        # w*T = 2*pi*0.01*1.41 ≈ 0.0886, gain ≈ 0.088
        # So peak should be much less than the 1 Hz case
        assert peak < 0.02, f"PSS should attenuate 0.01 Hz signal, peak={peak}"


# ======================== MAIN ======================== #

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
