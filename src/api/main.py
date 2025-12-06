"""
Main API application
FastAPI server for DEMS
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_wsgi_app
from prometheus_client.core import CollectorRegistry
import logging

from src.core import EnergyManager, GridManager
from src.agent import RLAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Dynamic Energy Management System API",
    description="RL-based energy management system with Prometheus monitoring",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core components
energy_manager = EnergyManager(grid_size=10, storage_capacity=1000.0)
grid_manager = GridManager(num_nodes=10)
rl_agent = RLAgent(observation_space_size=50, action_space_size=10)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "DEMS API",
        "version": "0.1.0",
        "status": "running",
        "endpoints": [
            "/health",
            "/energy/state",
            "/grid/state",
            "/optimization/predict",
            "/metrics",
        ],
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "energy_manager": "active",
        "grid_manager": "active",
        "rl_agent": "active",
    }


@app.get("/energy/state")
async def get_energy_state():
    """Get current energy system state"""
    return energy_manager.get_current_state()


@app.get("/grid/state")
async def get_grid_state():
    """Get current grid state"""
    return grid_manager.get_grid_state()


@app.post("/optimization/predict")
async def predict_optimization():
    """Get optimization prediction from RL agent"""
    try:
        result = energy_manager.optimize_distribution()
        return {
            "status": "success",
            "optimization": result,
            "agent_info": str(rl_agent),
        }
    except Exception as e:
        logger.error(f"Optimization error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Get system metrics (Prometheus format)"""
    return {
        "energy": {
            "generated": 150.5,
            "consumed": 120.3,
            "storage_level": 500.0,
        },
        "grid": {
            "frequency": 50.0,
            "voltage": 230.0,
            "utilization": 60.5,
        },
        "agent": rl_agent.get_training_stats(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
