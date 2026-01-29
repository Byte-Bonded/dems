#!/bin/bash
# Import IEEE 39-Bus Dashboard into Grafana
# This script automatically imports the dashboard JSON into Grafana

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DASHBOARD_FILE="$SCRIPT_DIR/grafana_dashboard_ieee39.json"
GRAFANA_URL="http://localhost:3000"
GRAFANA_USER="admin"
GRAFANA_PASSWORD="dems2024"

echo "================================================================"
echo "IMPORT IEEE 39-BUS DASHBOARD TO GRAFANA"
echo "================================================================"
echo ""

# Check if Grafana is accessible
if ! curl -s "$GRAFANA_URL/api/health" &> /dev/null; then
    echo "❌ Grafana is not accessible at $GRAFANA_URL"
    echo "   Please start Grafana first:"
    echo "   $ cd scripts/monitoring"
    echo "   $ docker compose -f docker-compose.monitoring.yml up -d"
    exit 1
fi

echo "✓ Grafana is accessible"

# Check if dashboard file exists
if [ ! -f "$DASHBOARD_FILE" ]; then
    echo "❌ Dashboard file not found: $DASHBOARD_FILE"
    exit 1
fi

echo "✓ Dashboard file found: $DASHBOARD_FILE"
echo ""

# Wait for Grafana to be fully ready
echo "⏳ Waiting for Grafana to be fully ready..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -s -u "$GRAFANA_USER:$GRAFANA_PASSWORD" "$GRAFANA_URL/api/datasources" &> /dev/null; then
        echo "✓ Grafana is ready"
        break
    fi
    attempt=$((attempt + 1))
    echo "   Attempt $attempt/$max_attempts..."
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo "❌ Grafana did not become ready in time"
    exit 1
fi

echo ""
echo "📤 Importing dashboard..."

# Import dashboard using Grafana API
response=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -u "$GRAFANA_USER:$GRAFANA_PASSWORD" \
    -d @"$DASHBOARD_FILE" \
    "$GRAFANA_URL/api/dashboards/db")

# Check response
if echo "$response" | grep -q '"status":"success"'; then
    echo "✓ Dashboard imported successfully!"
    
    # Extract dashboard URL
    dashboard_url=$(echo "$response" | grep -o '"url":"[^"]*"' | cut -d'"' -f4)
    
    echo ""
    echo "================================================================"
    echo "DASHBOARD IMPORTED"
    echo "================================================================"
    echo ""
    echo "📊 Dashboard URL:"
    echo "   $GRAFANA_URL$dashboard_url"
    echo ""
    echo "🔐 Login Credentials:"
    echo "   Username: $GRAFANA_USER"
    echo "   Password: $GRAFANA_PASSWORD"
    echo ""
    echo "💡 The dashboard includes:"
    echo "   • System overview (generation, load, losses, frequency)"
    echo "   • Voltage monitoring (all 39 buses)"
    echo "   • Generator performance (10 generators)"
    echo "   • DER systems (Solar, Wind, BESS, EV, DR)"
    echo "   • Load control test metrics"
    echo "   • Performance metrics"
    echo ""
elif echo "$response" | grep -q '"message":"Dashboard already exists"'; then
    echo "ℹ️  Dashboard already exists in Grafana"
    echo ""
    echo "📊 Access the dashboard at:"
    echo "   $GRAFANA_URL/dashboards"
    echo ""
else
    echo "❌ Failed to import dashboard"
    echo "Response: $response"
    exit 1
fi

echo "================================================================"
echo "NEXT STEPS"
echo "================================================================"
echo ""
echo "1. Open the dashboard: $GRAFANA_URL"
echo "2. Run the simulation to see live data:"
echo "   $ cd $SCRIPT_DIR"
echo "   $ ./start_monitoring.sh"
echo ""
echo "   Or run directly:"
echo "   $ python3 run_simulation_with_monitoring.py --mode demo"
echo ""
