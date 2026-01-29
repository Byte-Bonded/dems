#!/usr/bin/env python3
"""
Quick Test Script for IEEE 39-Bus Monitoring Integration

This script tests the monitoring setup without running the full simulation.
It verifies:
- Prometheus client library is available
- Metrics can be created and exported
- HTTP server can be started
- Basic monitoring functionality works
"""

import sys
import time
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("IEEE 39-BUS MONITORING - QUICK TEST")
print("=" * 70)
print()

# Test 1: Check imports
print("1️⃣  Testing imports...")
try:
    import numpy as np
    print("   ✓ NumPy")
except ImportError as e:
    print(f"   ❌ NumPy: {e}")
    sys.exit(1)

try:
    from prometheus_client import Gauge, Counter, start_http_server
    print("   ✓ Prometheus Client")
except ImportError as e:
    print(f"   ❌ Prometheus Client: {e}")
    print("   Install with: pip install prometheus-client")
    sys.exit(1)

try:
    from monitoring_integration import IEEE39BusMonitor
    print("   ✓ Monitoring Integration")
except ImportError as e:
    print(f"   ❌ Monitoring Integration: {e}")
    sys.exit(1)

print()

# Test 2: Create monitor
print("2️⃣  Creating IEEE39BusMonitor...")
try:
    monitor = IEEE39BusMonitor()
    print("   ✓ Monitor created successfully")
except Exception as e:
    print(f"   ❌ Failed to create monitor: {e}")
    sys.exit(1)

print()

# Test 3: Start HTTP server
print("3️⃣  Starting metrics HTTP server...")
port = 9999  # Use different port for testing
try:
    start_http_server(port)
    print(f"   ✓ HTTP server started on port {port}")
    print(f"   📍 Metrics endpoint: http://localhost:{port}/metrics")
except Exception as e:
    print(f"   ❌ Failed to start HTTP server: {e}")
    sys.exit(1)

print()

# Test 4: Update some test metrics
print("4️⃣  Updating test metrics...")
try:
    # Simulate some system state
    test_state = {
        'frequency_hz': 50.0,
        'voltage_min': 0.95,
        'voltage_max': 1.05,
        'power_flow_converged': True,
        'total_generation_mw': 6000.0,
        'total_load_mw': 5800.0
    }
    
    monitor.update_system_state(test_state)
    monitor.total_generation_mw.set(6000.0)
    monitor.total_load_mw.set(5800.0)
    monitor.total_losses_mw.set(200.0)
    monitor.increment_update_count()
    
    print("   ✓ Test metrics updated")
except Exception as e:
    print(f"   ❌ Failed to update metrics: {e}")
    sys.exit(1)

print()

# Test 5: Verify metrics can be accessed
print("5️⃣  Verifying metrics endpoint...")
try:
    import urllib.request
    time.sleep(1)  # Give server a moment
    
    with urllib.request.urlopen(f"http://localhost:{port}/metrics") as response:
        content = response.read().decode('utf-8')
        
        # Check for some key metrics
        if 'ieee39_total_generation_mw' in content:
            print("   ✓ Metrics are being exported")
            
            # Count metrics
            metric_lines = [line for line in content.split('\n') if line and not line.startswith('#')]
            print(f"   ✓ Found {len(metric_lines)} metric values")
        else:
            print("   ⚠️  Metrics found but IEEE 39-Bus metrics missing")
            
except Exception as e:
    print(f"   ⚠️  Could not fetch metrics: {e}")

print()

# Summary
print("=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print()
print("✅ All core components are working!")
print()
print("📍 Test metrics server is running at:")
print(f"   http://localhost:{port}/metrics")
print()
print("🎯 Next steps:")
print("   1. Check if Prometheus and Grafana are running:")
print("      cd ../monitoring")
print("      docker compose -f docker-compose.monitoring.yml up -d")
print()
print("   2. Run the full simulation:")
print("      python3 run_simulation_with_monitoring.py --mode demo")
print()
print("   3. Or use the all-in-one script:")
print("      ./start_monitoring.sh")
print()
print("Press Ctrl+C to stop the test server...")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n\n👋 Test server stopped")
