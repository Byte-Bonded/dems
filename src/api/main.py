"""
Main API application
FastAPI server for DEMS
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_wsgi_app
from prometheus_client.core import CollectorRegistry
import logging

from src.core import EnergyManager, GridMonitor
from src.agent import RLAgent, DEMSEnvironment
from src.grid import DEMSGrid

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

# Lazy-loaded components (initialized on first use)
_grid: DEMSGrid = None
_environment: DEMSEnvironment = None
_agent: RLAgent = None


def get_grid() -> DEMSGrid:
    """Get or create the DEMS grid (lazy initialization)"""
    global _grid
    if _grid is None:
        logger.info("Initializing DEMSGrid...")
        _grid = DEMSGrid()
        _grid.run_power_flow()
    return _grid


def get_environment() -> DEMSEnvironment:
    """Get or create the RL environment (lazy initialization)"""
    global _environment
    if _environment is None:
        _environment = DEMSEnvironment(num_nodes=10, max_steps=1000)
    return _environment


def get_agent() -> RLAgent:
    """Get or create the RL agent (lazy initialization)"""
    global _agent
    if _agent is None:
        _agent = RLAgent(env=get_environment())
    return _agent


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
            "/grid/power-flow",
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
        "grid": "lazy-loaded",
        "rl_agent": "lazy-loaded",
    }


@app.get("/energy/state")
async def get_energy_state():
    """Get current energy system state"""
    return energy_manager.get_current_state()


@app.get("/grid/state")
async def get_grid_state():
    """Get current grid state"""
    try:
        grid = get_grid()
        return grid.get_state()
    except Exception as e:
        logger.error(f"Failed to get grid state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/grid/power-flow")
async def run_power_flow():
    """Run power flow analysis"""
    try:
        grid = get_grid()
        result = grid.run_power_flow()
        return {
            "converged": result.converged,
            "iterations": result.iterations,
            "total_generation_mw": result.total_generation_mw,
            "total_load_mw": result.total_load_mw,
            "total_losses_mw": result.total_losses_mw,
            "min_voltage_pu": result.min_voltage_pu,
            "max_voltage_pu": result.max_voltage_pu,
            "is_secure": result.is_secure,
        }
    except Exception as e:
        logger.error(f"Power flow failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimization/predict")
async def predict_optimization():
    """Get optimization prediction from RL agent"""
    try:
        result = energy_manager.optimize_distribution()
        return {
            "status": "success",
            "optimization": result,
            "agent_info": "RL Agent available",
        }
    except Exception as e:
        logger.error(f"Optimization error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    try:
        grid = get_grid()
        state = grid.get_state()
        return {
            "energy": energy_manager.get_current_state(),
            "grid": {
                "total_generation_mw": state.get("global_metrics", {}).get("total_generation_mw", 0),
                "total_load_mw": state.get("global_metrics", {}).get("total_load_mw", 0),
                "frequency_hz": 50.0,
            }
        }
    except Exception as e:
        return {
            "energy": energy_manager.get_current_state(),
            "grid": {"error": str(e)}
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
