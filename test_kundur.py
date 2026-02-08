"""
Quick Test Script for Kundur Two-Area System
Validates power flow, DER integration, and basic functionality
"""

import numpy as np
from src.simulation import KundurTwoAreaSystem, PowerFlowRunner, KundurConfig, AreaID
from src.grid import DEMSGrid
from src.agent import KundurEnvironment

def test_basic_power_flow():
    """Test 1: Basic power flow convergence"""
    print("\n" + "="*60)
    print("TEST 1: Basic Power Flow")
    print("="*60)
    
    kundur = KundurTwoAreaSystem(enable_der=False)
    pf = PowerFlowRunner()
    result = pf.run(kundur.net, verbose=False)
    
    print(f"✓ Converged: {result.converged}")
    print(f"✓ Total Generation: {result.total_generation_mw:.2f} MW")
    print(f"✓ Total Load: {result.total_load_mw:.2f} MW")
    print(f"✓ Losses: {result.total_losses_mw:.2f} MW")
    print(f"✓ Voltage range: {result.min_voltage_pu:.3f} - {result.max_voltage_pu:.3f} pu")
    
    # Check tie-line flow
    tie_flows = kundur.get_tie_line_flow()
    total_tie_flow = sum(flow['p_from_mw'] for flow in tie_flows.values())
    print(f"✓ Tie-line flow: {total_tie_flow:.2f} MW (nominal: ~400 MW)")
    
    assert result.converged, "Power flow did not converge"
    assert 350 < total_tie_flow < 450, f"Tie-line flow out of range: {total_tie_flow}"
    print("✓ TEST 1 PASSED\n")

def test_der_integration():
    """Test 2: DER integration"""
    print("="*60)
    print("TEST 2: DER Integration")
    print("="*60)
    
    kundur = KundurTwoAreaSystem(enable_der=True)
    
    # Check DER installation
    der_state = kundur.get_der_state()
    print(f"✓ DER units installed: {len(der_state)}")
    
    for der in der_state:
        print(f"  - {der['name']}: {der['capacity_mw']} MW ({der['type']}) at Bus {der['bus']}")
    
    # Update environmental conditions
    kundur.update_der_conditions(
        solar_irradiance_w_m2=900.0,
        wind_speed_m_s=14.0,
        temperature_c=25.0
    )
    
    # Run power flow
    pf = PowerFlowRunner()
    result = pf.run(kundur.net, verbose=False)
    
    print(f"✓ Power flow converged with DER: {result.converged}")
    assert result.converged, "Power flow with DER did not converge"
    print("✓ TEST 2 PASSED\n")

def test_control_actions():
    """Test 3: Control actions (generator setpoints, battery dispatch)"""
    print("="*60)
    print("TEST 3: Control Actions")
    print("="*60)
    
    grid = DEMSGrid(enable_der=True)
    
    # Initial power flow
    result = grid.run_power_flow(verbose=False)
    initial_state = grid.get_state()
    initial_gen_p = initial_state['generators'][0]['p_mw']
    
    print(f"✓ Initial G1 power: {initial_gen_p:.2f} MW")
    
    # Change generator setpoint
    success = grid.set_generator_power("G1", 750.0)
    assert success, "Failed to set generator power"
    
    result = grid.run_power_flow(verbose=False)
    new_state = grid.get_state()
    new_gen_p = new_state['generators'][0]['p_mw']
    
    print(f"✓ New G1 power: {new_gen_p:.2f} MW (target: 750.0 MW)")
    assert abs(new_gen_p - 750.0) < 1.0, "Generator setpoint not applied correctly"
    
    # Battery dispatch
    if grid.kundur.der_manager:
        success = grid.dispatch_battery("BESS_A1", 20.0)  # Discharge 20 MW
        print(f"✓ Battery BESS_A1 dispatched: {success}")
    
    # Load perturbation
    grid.apply_load_perturbation(AreaID.AREA_1, 50.0)  # Add 50 MW load
    result = grid.run_power_flow(verbose=False)
    print(f"✓ Power flow after load perturbation: {result.converged}")
    
    print("✓ TEST 3 PASSED\n")

def test_area_metrics():
    """Test 4: Area-level metrics"""
    print("="*60)
    print("TEST 4: Area Metrics")
    print("="*60)
    
    grid = DEMSGrid(enable_der=True)
    grid.run_power_flow(verbose=False)
    
    state = grid.get_state()
    areas = state['areas']
    
    for area_name, metrics in areas.items():
        print(f"\n{area_name}:")
        print(f"  Generation: {metrics['generation_mw']:.2f} MW")
        print(f"  Load: {metrics['load_mw']:.2f} MW")
        print(f"  Avg Voltage: {metrics['avg_voltage_pu']:.3f} pu")
        
        assert 0.95 < metrics['avg_voltage_pu'] < 1.05, f"Voltage out of range in {area_name}"
    
    print("\n✓ TEST 4 PASSED\n")

def test_rl_environment():
    """Test 5: RL Environment"""
    print("="*60)
    print("TEST 5: RL Environment")
    print("="*60)
    
    env = KundurEnvironment(enable_der=True, max_steps=10)
    
    # Reset environment
    obs, info = env.reset()
    print(f"✓ Observation shape: {obs.shape}")
    print(f"✓ Expected shape: {env.observation_space.shape}")
    assert obs.shape == env.observation_space.shape, "Observation shape mismatch"
    
    # Take random actions
    total_reward = 0
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"✓ Episode terminated at step {step+1}")
            break
    
    print(f"✓ Total reward: {total_reward:.2f}")
    print(f"✓ Final converged: {info.get('converged', False)}")
    
    env.close()
    print("✓ TEST 5 PASSED\n")

def main():
    """Run all tests"""
    print("\n" + "🔋 "*20)
    print("KUNDUR TWO-AREA SYSTEM TEST SUITE")
    print("🔋 "*20)
    
    try:
        test_basic_power_flow()
        test_der_integration()
        test_control_actions()
        test_area_metrics()
        test_rl_environment()
        
        print("="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
        print("\nKundur Two-Area System is ready for use!")
        print("See docs/KUNDUR_GUIDE.md for usage examples.\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
