#!/bin/bash
# Start IEEE 39-Bus Simulation with Complete Monitoring Stack
# 
# This script starts:
# 1. Prometheus (time-series database)
# 2. Grafana (visualization)
# 3. IEEE 39-Bus simulation with metrics export

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MONITORING_DIR="$PROJECT_ROOT/scripts/monitoring"

echo "================================================================"
echo "IEEE 39-BUS SYSTEM - COMPLETE MONITORING STACK"
echo "================================================================"
echo ""
echo "📂 Project Root: $PROJECT_ROOT"
echo "📂 Script Dir:   $SCRIPT_DIR"
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed or not in PATH"
    echo "   Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

echo "✓ Docker is available"

# Check if Docker Compose is available
if ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed or not in PATH"
    echo "   Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✓ Docker Compose is available"
echo ""

# Navigate to monitoring directory
cd "$MONITORING_DIR"

echo "🐳 Starting monitoring stack (Prometheus + Grafana)..."
echo "----------------------------------------------------------------"

# Start Docker Compose services
docker compose -f docker-compose.monitoring.yml up -d

echo ""
echo "⏳ Waiting for services to be ready..."
sleep 5

# Check if Prometheus is up
if curl -s http://localhost:9090/-/ready &> /dev/null; then
    echo "✓ Prometheus is ready on http://localhost:9090"
else
    echo "⚠ Prometheus may still be starting..."
fi

# Check if Grafana is up
if curl -s http://localhost:3000/api/health &> /dev/null; then
    echo "✓ Grafana is ready on http://localhost:3000"
else
    echo "⚠ Grafana may still be starting..."
fi

echo ""
echo "================================================================"
echo "MONITORING STACK IS RUNNING"
echo "================================================================"
echo ""
echo "📍 Access Points:"
echo "  • Prometheus:  http://localhost:9090"
echo "  • Grafana:     http://localhost:3000 (admin/dems2024)"
echo ""
echo "⏭️  Next Step: Import the IEEE 39-Bus dashboard"
echo ""
echo "   Option 1 - Auto Import (if script available):"
echo "   $ cd $SCRIPT_DIR"
echo "   $ ./import_ieee39_dashboard.sh"
echo ""
echo "   Option 2 - Manual Import:"
echo "   1. Open http://localhost:3000"
echo "   2. Login with admin/dems2024"
echo "   3. Go to Dashboards → Import"
echo "   4. Upload: $SCRIPT_DIR/grafana_dashboard_ieee39.json"
echo ""
echo "================================================================"
echo "NOW STARTING IEEE 39-BUS SIMULATION"
echo "================================================================"
echo ""

# Navigate to core directory
cd "$SCRIPT_DIR"

# Check if Python environment exists
if [ -f "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON="$PROJECT_ROOT/.venv/bin/python"
    echo "✓ Using virtual environment Python"
elif command -v python3 &> /dev/null; then
    PYTHON="python3"
    echo "✓ Using system Python3"
else
    echo "❌ Python3 not found"
    exit 1
fi

echo "🔧 Python: $PYTHON"
echo ""

# Check required packages
echo "📦 Checking required packages..."
required_packages="numpy scipy pypower pandapower prometheus_client"
missing_packages=""

for package in $required_packages; do
    if ! $PYTHON -c "import $package" 2>/dev/null; then
        missing_packages="$missing_packages $package"
    fi
done

if [ -n "$missing_packages" ]; then
    echo "⚠️  Missing packages:$missing_packages"
    echo "   Installing..."
    
    if [ -f "$PROJECT_ROOT/.venv/bin/pip" ]; then
        $PROJECT_ROOT/.venv/bin/pip install $missing_packages
    else
        $PYTHON -m pip install --user $missing_packages
    fi
fi

echo "✓ All required packages are installed"
echo ""

# Ask user for simulation mode
echo "================================================================"
echo "SELECT SIMULATION MODE"
echo "================================================================"
echo "1) Demo Mode     - Load control demonstration (quick)"
echo "2) Continuous    - Continuous simulation with monitoring"
echo ""
read -p "Enter choice [1-2] (default: 1): " choice
choice=${choice:-1}

case $choice in
    1)
        MODE="demo"
        echo "▶️  Running DEMO mode (Load Control Demonstration)"
        ;;
    2)
        MODE="continuous"
        read -p "Number of steps (default: 100): " steps
        steps=${steps:-100}
        read -p "Interval between steps in seconds (default: 5): " interval
        interval=${interval:-5}
        echo "▶️  Running CONTINUOUS mode ($steps steps, ${interval}s interval)"
        ;;
    *)
        echo "Invalid choice. Using demo mode."
        MODE="demo"
        ;;
esac

echo ""
echo "================================================================"
echo "STARTING SIMULATION"
echo "================================================================"
echo ""

# Run the simulation
if [ "$MODE" = "demo" ]; then
    $PYTHON run_simulation_with_monitoring.py --mode demo --port 9136
else
    $PYTHON run_simulation_with_monitoring.py --mode continuous --steps $steps --interval $interval --port 9136
fi

# The simulation will keep running until Ctrl+C
echo ""
echo "================================================================"
echo "SIMULATION STOPPED"
echo "================================================================"
echo ""
echo "📊 Monitoring stack is still running."
echo "   View metrics at:"
echo "   • Prometheus: http://localhost:9090"
echo "   • Grafana:    http://localhost:3000"
echo ""
read -p "Stop monitoring stack? [y/N]: " stop_monitoring

if [[ "$stop_monitoring" =~ ^[Yy]$ ]]; then
    echo "🛑 Stopping monitoring stack..."
    cd "$MONITORING_DIR"
    docker compose -f docker-compose.monitoring.yml down
    echo "✓ Monitoring stack stopped"
else
    echo "ℹ️  Monitoring stack still running."
    echo "   To stop later: cd $MONITORING_DIR && docker compose -f docker-compose.monitoring.yml down"
fi

echo ""
echo "👋 Done!"
