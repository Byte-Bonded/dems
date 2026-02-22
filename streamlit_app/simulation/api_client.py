"""
API client for connecting the Streamlit dashboard to the FastAPI backend.
Falls back gracefully when the API is unreachable.
"""

import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://localhost:8000"


class APIClient:
    """Thin HTTP wrapper around the DEMS FastAPI endpoints."""

    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str) -> Optional[Dict]:
        try:
            r = requests.get(f"{self.base_url}{path}", timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            logger.warning("API GET %s failed: %s", path, exc)
            return None

    def _post(self, path: str, data: Optional[Dict] = None) -> Optional[Dict]:
        try:
            r = requests.post(f"{self.base_url}{path}", json=data or {}, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            logger.warning("API POST %s failed: %s", path, exc)
            return None

    # --- endpoints -------
    def health(self) -> Optional[Dict]:
        return self._get("/health")

    def grid_state(self) -> Optional[Dict]:
        return self._get("/grid/state")

    def run_power_flow(self) -> Optional[Dict]:
        return self._post("/grid/power-flow")

    def energy_state(self) -> Optional[Dict]:
        return self._get("/energy/state")

    def metrics(self) -> Optional[Dict]:
        return self._get("/metrics")

    def is_reachable(self) -> bool:
        return self.health() is not None
