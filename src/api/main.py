"""
Main API application
FastAPI server for DEMS
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import threading

from src.grid import DEMSGrid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Dynamic Energy Management System API",
    description="Hierarchical multi-agent RL energy management system",
    version="0.2.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded components (initialized on first use)
_grid: DEMSGrid = None
_grid_lock = threading.Lock()


def get_grid() -> DEMSGrid:
    """Get or create the DEMS grid (lazy initialization with thread safety)"""
    global _grid
    if _grid is None:
        with _grid_lock:
            if _grid is None:
                logger.info("Initializing DEMSGrid...")
                _grid = DEMSGrid()
                _grid.run_power_flow()
    return _grid


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "DEMS API",
        "version": "0.2.0",
        "status": "running",
        "endpoints": [
            "/health",
            "/grid/state",
            "/grid/power-flow",
            "/metrics",
        ],
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "grid": "lazy-loaded",
        "multi_agent": "pending",
    }


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


@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    try:
        grid = get_grid()
        state = grid.get_state()
        return {
            "grid": {
                "total_generation_mw": state.get("global_metrics", {}).get("total_generation_mw", 0),
                "total_load_mw": state.get("global_metrics", {}).get("total_load_mw", 0),
                "frequency_hz": 50.0,
            }
        }
    except Exception as e:
        return {
            "grid": {"error": str(e)}
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
