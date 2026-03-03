# DEMS API Reference

## Base URL
```
http://localhost:8000
```

## Authentication
Currently no authentication required. Add JWT in production.

## Response Format
All responses are JSON:
```json
{
  "status": "success|error",
  "data": {},
  "error": null
}
```

## Endpoints

### Health & Status

#### GET /
Root endpoint listing all available endpoints.

**Response:**
```json
{
  "name": "DEMS API",
  "version": "0.1.0",
  "status": "running",
  "endpoints": [
    "/health",
    "/energy/state",
    "/grid/state",
    "/optimization/predict",
    "/metrics"
  ]
}
```

#### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "energy_manager": "active",
  "grid_manager": "active",
  "rl_agent": "active"
}
```

### Energy Management

#### GET /energy/state
Get current energy system state.

**Response:**
```json
{
  "timestamp": "2024-01-01T12:00:00",
  "storage_level": 750,
  "storage_capacity": 1000,
  "grid_size": 10
}
```

**Parameters:**
- None

**Status Codes:**
- 200: Success
- 500: Server error

### Grid Management

#### GET /grid/state
Get current grid state and metrics.

**Response:**
```json
{
  "num_nodes": 10,
  "total_load": 850,
  "total_capacity": 1000,
  "utilization": 85.0,
  "avg_frequency": 50.0
}
```

**Parameters:**
- None

**Status Codes:**
- 200: Success
- 500: Server error

### Optimization

#### POST /optimization/predict
Get optimization prediction from RL agent.

**Request:**
```json
{}
```

**Response:**
```json
{
  "status": "success",
  "optimization": {
    "status": "pending",
    "recommendation": null
  },
  "agent_info": "RLAgent(obs_size=50, action_size=10)"
}
```

**Status Codes:**
- 200: Success
- 500: Optimization error

### Metrics

#### GET /metrics
Get system metrics in Prometheus format.

**Response:**
```json
{
  "energy": {
    "generated": 150.5,
    "consumed": 120.3,
    "storage_level": 500.0
  },
  "grid": {
    "frequency": 50.0,
    "voltage": 230.0,
    "utilization": 60.5
  },
  "agent": {
    "training_episodes": 0,
    "avg_reward": 0.0
  }
}
```

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Invalid request parameters"
}
```

### 404 Not Found
```json
{
  "detail": "Endpoint not found"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Internal server error message"
}
```

## Rate Limiting
Not currently implemented. To be added in production.

## Pagination
Not applicable for current endpoints.

## Data Types

### Timestamp
ISO 8601 format: `2024-01-01T12:00:00`

### Energy Values
Float in kWh: `850.5`

### Percentage
Float 0-100: `85.5`

### Frequency
Float in Hz: `50.0`

### Voltage
Float in V: `230.5`

## Examples

### Using curl
```bash
# Get energy state
curl -X GET http://localhost:8000/energy/state

# Get grid state
curl -X GET http://localhost:8000/grid/state

# Get optimization
curl -X POST http://localhost:8000/optimization/predict

# Get metrics
curl -X GET http://localhost:8000/metrics
```

### Using Python requests
```python
import requests

base_url = "http://localhost:8000"

# Get energy state
response = requests.get(f"{base_url}/energy/state")
print(response.json())

# Get optimization
response = requests.post(f"{base_url}/optimization/predict")
print(response.json())
```

### Using JavaScript fetch
```javascript
// Get energy state
fetch('http://localhost:8000/energy/state')
  .then(r => r.json())
  .then(data => console.log(data));

// Get optimization
fetch('http://localhost:8000/optimization/predict', {
  method: 'POST'
})
  .then(r => r.json())
  .then(data => console.log(data));
```

## WebSocket Endpoints
Not currently implemented. To be added for real-time updates.

## Versioning
Current API version: v1 (no prefix in URL yet)

## Changelog

### v0.1.0 (2024-01-01)
- Initial API with core endpoints
- Energy and grid management
- RL agent optimization
- Prometheus metrics
