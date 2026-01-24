"""
Grid Management System
Handles grid stability, frequency control, and load balancing
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class GridNode:
    """Represents a node in the energy grid"""
    node_id: int
    voltage: float
    frequency: float
    load: float
    capacity: float


class GridManager:
    """
    Manages the energy grid
    Monitors and controls grid stability
    """

    def __init__(self, num_nodes: int = 10):
        self.num_nodes = num_nodes
        self.nodes: Dict[int, GridNode] = {}
        self._initialize_nodes()

    def _initialize_nodes(self) -> None:
        """Initialize grid nodes"""
        for i in range(self.num_nodes):
            self.nodes[i] = GridNode(
                node_id=i,
                voltage=230.0,
                frequency=50.0,
                load=0.0,
                capacity=100.0,
            )

    def get_grid_state(self) -> Dict:
        """Get current grid state"""
        total_load = sum(node.load for node in self.nodes.values())
        total_capacity = sum(node.capacity for node in self.nodes.values())

        return {
            "num_nodes": self.num_nodes,
            "total_load": total_load,
            "total_capacity": total_capacity,
            "utilization": (total_load / total_capacity * 100) if total_capacity > 0 else 0,
            "avg_frequency": sum(node.frequency for node in self.nodes.values()) / self.num_nodes,
        }

    def update_node_load(self, node_id: int, load: float) -> bool:
        """Update load on a specific node"""
        if node_id in self.nodes:
            self.nodes[node_id].load = min(load, self.nodes[node_id].capacity)
            return True
        return False

    def check_stability(self) -> bool:
        """Check if grid is stable"""
        avg_freq = sum(node.frequency for node in self.nodes.values()) / self.num_nodes
        return 49.5 <= avg_freq <= 50.5
