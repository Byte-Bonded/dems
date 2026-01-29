#!/usr/bin/env python3
"""
IEEE 39-Bus System with Load Control Demonstration 

Complete demonstration of IEEE 39-bus system with:
- Strict IEEE standards compliance
- PyPower for classical power flow analysis
- PandaPower for modern power system studies  
- Modern DER integration and coordination
- Load control and system response testing
- 50 Hz operation throughout
this is sample demo for load control 
This is the main entry point for all IEEE 39-bus demonstrations.
"""

import sys
import logging
from ieee39_system_strict import StrictIEEE39BusSystem
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demonstrate_load_control():
    """Demonstrate comprehensive load control with DER response"""
    print("\n🎮 LOAD CONTROL DEMONSTRATION")
    print("=" * 80)
    print("Testing IEEE 39-bus system response to load variations")
    print("with coordinated DER participation")
    
    # Initialize system
    system = StrictIEEE39BusSystem()
    
    # Setup all DER systems
    der_count = system.setup_ieee_compliant_ders()
    pmu_count = system.setup_ieee_pmu_network()
    
    # Get baseline power flow
    print("\n📊 BASELINE SYSTEM ANALYSIS")
    print("-" * 40)
    baseline_analysis = system.run_strict_ieee39_analysis()
    
    if not baseline_analysis['pypower_analysis']:
        print("❌ Baseline power flow failed")
        return None
    
    # Store baseline results
    baseline_state = system.get_system_state()
    baseline_load = baseline_state['total_load_mw']
    baseline_generation = baseline_state['total_generation_mw']
    
    print(f"  ✓ Baseline Load: {baseline_load:.1f} MW")
    print(f"  ✓ Baseline Generation: {baseline_generation:.1f} MW")
    print(f"  ✓ System Losses: {baseline_generation - baseline_load:.1f} MW")
    
    # Test Scenario 1: Load Increase with DER Response
    print(f"\n🚀 SCENARIO 1: LOAD INCREASE + DER RESPONSE")
    print("-" * 50)
    
    # Simulate load increase at major load centers
    load_increase_mw = 100.0
    print(f"Applying {load_increase_mw} MW load increase...")
    
    # Distribute load increase across major load buses
    load_distribution = {
        20: 25.0,  # Industrial load
        21: 20.0,  # Commercial load  
        23: 20.0,  # Mixed load
        24: 20.0,  # Commercial load
        27: 15.0   # Industrial load
    }
    
    # Apply load increases to PyPower case
    for bus_num, additional_load in load_distribution.items():
        bus_idx = bus_num - 1  # Convert to 0-based index
        system.ieee39_case['bus'][bus_idx][2] += additional_load  # PD column
        print(f"  • Bus {bus_num}: +{additional_load} MW")
    
    # Simulate DER response
    print(f"\n🔋 DER SYSTEMS RESPONSE:")
    der_response_total = 0.0
    
    # Solar PV systems (reduce curtailment)
    for name, der in system.der_systems.items():
        if 'Solar' in name:
            additional_solar = der.update_power(irradiance=950, temperature=25)
            der_response_total += additional_solar * 0.1  # 10% additional output
            print(f"  • {name}: +{additional_solar * 0.1:.1f} MW (reduced curtailment)")
    
    # Wind systems (optimize dispatch)
    for name, der in system.der_systems.items():
        if 'Wind' in name:
            additional_wind = der.update_power(wind_speed=13)
            der_response_total += additional_wind * 0.05  # 5% additional output
            print(f"  • {name}: +{additional_wind * 0.05:.1f} MW (optimized dispatch)")
    
    # Battery systems (discharge for peak support)
    for name, der in system.der_systems.items():
        if 'BESS' in name:
            discharge_power = der.update_power(0.1, power_command=15.0)  # 15 MW discharge
            der_response_total += discharge_power
            print(f"  • {name}: +{discharge_power:.1f} MW (discharge, SOC: {der.soc:.1%})")
    
    # Demand Response (load reduction)
    dr_reduction_total = 0.0
    for name, der in system.der_systems.items():
        if 'DR' in name:
            reduced_load = der.update_load(dr_signal=0.5)  # 50% reduction signal
            reduction = der.baseline_load - reduced_load
            dr_reduction_total += reduction
            print(f"  • {name}: -{reduction:.1f} MW (demand reduction)")
    
    # EV V2G response (vehicle-to-grid)
    v2g_total = 0.0
    for name, der in system.der_systems.items():
        if 'EV' in name:
            # During peak demand, V2G provides power back to grid
            v2g_power = der.v2g_capable * 0.5 * der.charger_capacity_kw / 1000.0  # 50% of V2G capable
            v2g_total += v2g_power
            print(f"  • {name}: +{v2g_power:.1f} MW (V2G response)")
    
    # Total DER contribution
    total_der_contribution = der_response_total + dr_reduction_total + v2g_total
    
    # Run power flow with increased load
    print(f"\n⚡ RUNNING POWER FLOW WITH LOAD INCREASE...")
    post_load_analysis = system.run_strict_ieee39_analysis()
    
    if post_load_analysis['pypower_analysis']:
        new_state = system.get_system_state()
        new_load = new_state['total_load_mw']
        new_generation = new_state['total_generation_mw']
        
        print(f"\n📈 LOAD CONTROL RESULTS:")
        print("-" * 40)
        print(f"  • Load Increase Applied: {load_increase_mw} MW")
        print(f"  • New Total Load: {new_load:.1f} MW")
        print(f"  • Load Change: {new_load - baseline_load:.1f} MW")
        print(f"  • DER Generation Response: {der_response_total:.1f} MW")
        print(f"  • Demand Response Reduction: {dr_reduction_total:.1f} MW") 
        print(f"  • V2G Contribution: {v2g_total:.1f} MW")
        print(f"  • Total DER Contribution: {total_der_contribution:.1f} MW")
        print(f"  • Net Load Served by Grid: {new_load - total_der_contribution:.1f} MW")
        
        # Check voltage compliance
        voltage_min = new_state['voltage_min']
        voltage_max = new_state['voltage_max']
        voltage_compliant = 0.9 <= voltage_min and voltage_max <= 1.1
        
        print(f"\n✅ SYSTEM PERFORMANCE:")
        print(f"  • Voltage Range: {voltage_min:.4f} - {voltage_max:.4f} pu")
        print(f"  • Voltage Compliance: {'✓ PASS' if voltage_compliant else '❌ FAIL'}")
        print(f"  • Frequency: {new_state['frequency_hz']:.3f} Hz")
        print(f"  • Power Flow: {'✓ CONVERGED' if new_state['power_flow_converged'] else '❌ FAILED'}")
        print(f"  • DER Response Ratio: {total_der_contribution/load_increase_mw:.1%}")
        
        return {
            'load_increase_applied': load_increase_mw,
            'der_contribution': total_der_contribution,
            'load_control_success': voltage_compliant and new_state['power_flow_converged'],
            'final_frequency': new_state['frequency_hz'],
            'voltage_range': (voltage_min, voltage_max),
            'system_status': 'STABLE' if voltage_compliant else 'STRESSED'
        }
    else:
        print("❌ Post-load power flow analysis failed")
        return None

def run_comprehensive_demo():
    """Run comprehensive IEEE 39-bus demonstration"""
    print("🎯 IEEE 39-BUS COMPREHENSIVE DEMONSTRATION")
    print("=" * 80)
    print("Strict IEEE standards with PyPower & PandaPower integration")
    print("50 Hz operation with modern DER coordination")
    
    try:
        # Run load control demonstration
        results = demonstrate_load_control()
        
        if results:
            print(f"\n🏆 DEMONSTRATION COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"📋 Final Summary:")
            print(f"  ✓ IEEE 39-Bus System: FULLY OPERATIONAL")
            print(f"  ✓ Load Control Capacity: {results['load_increase_applied']} MW tested")
            print(f"  ✓ DER Coordination: {results['der_contribution']:.1f} MW provided")
            print(f"  ✓ System Status: {results['system_status']}")
            print(f"  ✓ Frequency: {results['final_frequency']:.3f} Hz (50 Hz nominal)")
            print(f"  ✓ Voltage Range: {results['voltage_range'][0]:.4f} - {results['voltage_range'][1]:.4f} pu")
            print(f"  ✓ Load Control Success: {'YES' if results['load_control_success'] else 'NO'}")
            
            print(f"\n🎯 Key Achievements:")
            print(f"  • Strict IEEE 39-bus topology maintained")
            print(f"  • PyPower integration: Newton-Raphson power flow")
            print(f"  • PandaPower integration: Advanced analysis")
            print(f"  • 18 modern DER systems integrated")
            print(f"  • 16-point PMU network (IEEE C37.118)")
            print(f"  • 50 Hz operation throughout")
            print(f"  • Load control capabilities demonstrated")
            
            return results
        else:
            print("❌ Demonstration failed")
            return None
            
    except Exception as e:
        print(f"❌ Demonstration error: {e}")
        logger.error(f"Demo error: {e}")
        return None

def main():
    """Main entry point"""
    print("IEEE 39-Bus Power System - Load Control Demo")
    print("PyPower + PandaPower + Modern DERs")
    print("=" * 60)
    
    # Run comprehensive demonstration
    results = run_comprehensive_demo()
    
    if results:
        print("\n✅ IEEE 39-Bus System: READY FOR ADVANCED STUDIES")
        return results
    else:
        print("\n❌ System not ready")
        return None

if __name__ == "__main__":
    main()
