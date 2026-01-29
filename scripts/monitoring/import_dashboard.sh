#!/bin/bash
# Import DEMS Dashboard into Grafana

echo "=========================================="
echo "DEMS Grafana Dashboard Import Guide"
echo "=========================================="
echo ""

DASHBOARD_FILE="./grafana/dashboards/dems-grid-overview.json"

if [ ! -f "$DASHBOARD_FILE" ]; then
    echo "ERROR: Dashboard file not found at $DASHBOARD_FILE"
    exit 1
fi

echo "Dashboard file location:"
echo "  $(pwd)/$DASHBOARD_FILE"
echo ""

echo "To import the dashboard into Grafana:"
echo ""
echo "METHOD 1: Web UI Import (Recommended)"
echo "  1. Open Grafana: http://localhost:3000"
echo "  2. Login with admin/dems2024"
echo "  3. Click '+' icon → 'Import dashboard'"
echo "  4. Click 'Upload JSON file'"
echo "  5. Select: $(pwd)/$DASHBOARD_FILE"
echo "  6. Click 'Load' then 'Import'"
echo ""

echo "METHOD 2: Auto-provisioning (requires restart)"
echo "  The dashboard should auto-load from:"
echo "  /var/lib/grafana/dashboards/ (inside container)"
echo ""
echo "  To trigger auto-load:"
echo "  $ docker-compose -f docker-compose.monitoring.yml restart grafana"
echo ""

echo "METHOD 3: API Import (for automation)"
cat << 'SCRIPT'
  $ curl -X POST http://admin:dems2024@localhost:3000/api/dashboards/db \
    -H "Content-Type: application/json" \
    -d @grafana/dashboards/dems-grid-overview.json
SCRIPT

echo ""
echo "=========================================="
echo "After import, the dashboard will be available at:"
echo "http://localhost:3000/d/dems-grid-overview"
echo "=========================================="
