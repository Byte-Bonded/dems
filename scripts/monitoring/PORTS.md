# Quick Reference: DEMS Monitoring Ports

## Port Configuration

**NEW PORT: 9136** (changed from 8001)

### Services

- **DEMS Prometheus Exporter**: http://localhost:**9136**/metrics
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/dems2024)
- **Node Exporter**: http://localhost:9100

### Testing

**Test Monitoring (standalone)**:
```bash
cd scripts/monitoring
/path/to/.venv/bin/python test_monitoring.py
```

**Full Simulation**:
```bash
cd scripts/monitoring
/path/to/.venv/bin/python run_with_monitoring.py --steps 50 --interval 2
```

**Check Metrics**:
```bash
curl http://localhost:9136/metrics
```

### Grafana Dashboard Import

**Location**: `scripts/monitoring/grafana/dashboards/dems-grid-overview.json`

**Import Steps**:
1. Open http://localhost:3000
2. Login: admin/dems2024
3. Click + → Import dashboard
4. Upload `dems-grid-overview.json`
5. Click Import

### Prometheus Configuration

Prometheus scrapes metrics from: `host.docker.internal:9136`

See: `scripts/monitoring/prometheus.yml`

### Required Dependencies

```bash
pip install prometheus-client gymnasium
```

Or with venv:
```bash
.venv/bin/pip install prometheus-client gymnasium
```
