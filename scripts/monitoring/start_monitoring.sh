#!/bin/bash
# Quick start script for DEMS monitoring

echo "=========================================="
echo "DEMS Monitoring Stack Quick Start"
echo "=========================================="
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running. Please start Docker first."
    exit 1
fi

echo "✓ Docker is running"
echo ""

# Navigate to monitoring directory
cd "$(dirname "$0")"

echo "Starting Prometheus and Grafana..."
docker-compose -f docker-compose.monitoring.yml up -d

echo ""
echo "Waiting for services to start..."
sleep 5

# Check if services are running
if docker ps | grep -q "dems-prometheus"; then
    echo "✓ Prometheus is running on http://localhost:9090"
else
    echo "✗ Prometheus failed to start"
fi

if docker ps | grep -q "dems-grafana"; then
    echo "✓ Grafana is running on http://localhost:3000"
    echo "  Username: admin"
    echo "  Password: dems2024"
else
    echo "✗ Grafana failed to start"
fi

if docker ps | grep -q "dems-node-exporter"; then
    echo "✓ Node Exporter is running on http://localhost:9100"
else
    echo "✗ Node Exporter failed to start"
fi

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo "1. Run the simulation with monitoring:"
echo "   python run_with_monitoring.py"
echo ""
echo "2. Open Grafana at http://localhost:3000"
echo "3. View Prometheus at http://localhost:9090"
echo "4. Check metrics at http://localhost:8001/metrics (when simulation is running)"
echo ""
echo "To stop the monitoring stack:"
echo "   docker-compose -f docker-compose.monitoring.yml down"
echo ""
