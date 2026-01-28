"""
Grid Monitoring and Stability Analysis
Provides monitoring, metrics collection, and stability analysis for the DEMS grid

This module wraps around the main SuperGrid simulation to provide:
- Real-time stability monitoring
- Metrics collection and history tracking
- Stability thresholds and alerting
- Grid health assessment

Note: This is a monitoring layer - use DEMSGrid for actual grid control.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class StabilityStatus(Enum):
    """Grid stability status levels"""
    SECURE = "secure"           # All parameters within normal limits
    ALERT = "alert"             # Some parameters approaching limits
    EMERGENCY = "emergency"     # Parameters exceeding limits
    CRITICAL = "critical"       # Imminent collapse risk


@dataclass
class StabilityThresholds:
    """Configurable thresholds for stability monitoring (Indian Grid Code)"""
    # Frequency thresholds (Hz)
    freq_nominal: float = 50.0
    freq_warn_low: float = 49.7
    freq_warn_high: float = 50.3
    freq_trip_low: float = 49.5
    freq_trip_high: float = 50.5
    
    # Voltage thresholds (per-unit)
    volt_nominal: float = 1.0
    volt_warn_low: float = 0.95
    volt_warn_high: float = 1.05
    volt_trip_low: float = 0.90
    volt_trip_high: float = 1.10
    
    # Loading thresholds (percent)
    loading_warn: float = 80.0
    loading_trip: float = 100.0


@dataclass
class GridMetricsSnapshot:
    """Point-in-time snapshot of grid metrics"""
    timestamp: datetime
    
    # System-wide metrics
    total_generation_mw: float
    total_load_mw: float
    total_losses_mw: float
    
    # Voltage metrics
    min_voltage_pu: float
    max_voltage_pu: float
    avg_voltage_pu: float
    num_voltage_violations: int
    
    # Frequency metrics
    system_frequency_hz: float
    
    # Loading metrics
    max_line_loading_pct: float
    max_trafo_loading_pct: float
    num_overloads: int
    
    # Stability assessment
    stability_status: StabilityStatus
    
    # Optional detailed data
    area_metrics: Optional[Dict[str, Dict]] = None


class GridMonitor:
    """
    Real-time grid monitoring and stability analysis
    
    Provides monitoring services for the DEMS SuperGrid:
    - Continuous stability assessment
    - Metrics history tracking
    - Threshold-based alerting
    - Health scoring
    
    Example:
        >>> from src.grid import DEMSGrid
        >>> grid = DEMSGrid()
        >>> monitor = GridMonitor(thresholds=StabilityThresholds())
        >>> 
        >>> result = grid.run_power_flow()
        >>> state = grid.get_state()
        >>> snapshot = monitor.analyze(result, state)
        >>> print(f"Status: {snapshot.stability_status.value}")
    """
    
    def __init__(
        self,
        thresholds: Optional[StabilityThresholds] = None,
        history_size: int = 1000
    ):
        """
        Initialize Grid Monitor
        
        Args:
            thresholds: Stability thresholds (uses Indian Grid Code defaults if None)
            history_size: Maximum number of snapshots to retain
        """
        self.thresholds = thresholds or StabilityThresholds()
        self.history_size = history_size
        self.metrics_history: List[GridMetricsSnapshot] = []
        self._alerts: List[Dict[str, Any]] = []
        
        logger.info("GridMonitor initialized with thresholds: "
                   f"freq=[{self.thresholds.freq_trip_low}, {self.thresholds.freq_trip_high}] Hz, "
                   f"volt=[{self.thresholds.volt_trip_low}, {self.thresholds.volt_trip_high}] pu")
    
    def analyze(
        self,
        power_flow_result: Any,  # PowerFlowResult
        grid_state: Dict,
        system_frequency_hz: float = 50.0
    ) -> GridMetricsSnapshot:
        """
        Analyze current grid state and assess stability
        
        Args:
            power_flow_result: Result from DEMSGrid.run_power_flow()
            grid_state: State from DEMSGrid.get_state()
            system_frequency_hz: Current system frequency
            
        Returns:
            GridMetricsSnapshot with stability assessment
        """
        # Extract metrics from power flow result
        snapshot = GridMetricsSnapshot(
            timestamp=datetime.now(),
            total_generation_mw=power_flow_result.total_generation_mw,
            total_load_mw=power_flow_result.total_load_mw,
            total_losses_mw=power_flow_result.total_losses_mw,
            min_voltage_pu=power_flow_result.min_voltage_pu,
            max_voltage_pu=power_flow_result.max_voltage_pu,
            avg_voltage_pu=power_flow_result.avg_voltage_pu,
            num_voltage_violations=power_flow_result.num_voltage_violations,
            system_frequency_hz=system_frequency_hz,
            max_line_loading_pct=0.0,  # Would need to extract from net
            max_trafo_loading_pct=0.0,
            num_overloads=power_flow_result.num_line_overloads + power_flow_result.num_trafo_overloads,
            stability_status=StabilityStatus.SECURE,
            area_metrics=grid_state.get('areas')
        )
        
        # Assess stability
        snapshot.stability_status = self._assess_stability(snapshot)
        
        # Generate alerts if needed
        self._check_and_generate_alerts(snapshot)
        
        # Store in history
        self._add_to_history(snapshot)
        
        return snapshot
    
    def _assess_stability(self, snapshot: GridMetricsSnapshot) -> StabilityStatus:
        """
        Assess grid stability based on current metrics
        
        Returns:
            StabilityStatus enum value
        """
        th = self.thresholds
        
        # Check for critical conditions (trip limits exceeded)
        if (snapshot.system_frequency_hz <= th.freq_trip_low or 
            snapshot.system_frequency_hz >= th.freq_trip_high):
            return StabilityStatus.CRITICAL
            
        if (snapshot.min_voltage_pu <= th.volt_trip_low or 
            snapshot.max_voltage_pu >= th.volt_trip_high):
            return StabilityStatus.CRITICAL
        
        # Check for emergency conditions (violations present)
        if snapshot.num_voltage_violations > 0 or snapshot.num_overloads > 0:
            return StabilityStatus.EMERGENCY
        
        # Check for alert conditions (approaching limits)
        if (snapshot.system_frequency_hz <= th.freq_warn_low or 
            snapshot.system_frequency_hz >= th.freq_warn_high):
            return StabilityStatus.ALERT
            
        if (snapshot.min_voltage_pu <= th.volt_warn_low or 
            snapshot.max_voltage_pu >= th.volt_warn_high):
            return StabilityStatus.ALERT
        
        # All good
        return StabilityStatus.SECURE
    
    def _check_and_generate_alerts(self, snapshot: GridMetricsSnapshot) -> None:
        """Generate alerts for concerning conditions"""
        th = self.thresholds
        alerts = []
        
        # Frequency alerts
        if snapshot.system_frequency_hz < th.freq_warn_low:
            alerts.append({
                "type": "LOW_FREQUENCY",
                "severity": "WARNING" if snapshot.system_frequency_hz > th.freq_trip_low else "CRITICAL",
                "value": snapshot.system_frequency_hz,
                "threshold": th.freq_warn_low,
                "message": f"System frequency {snapshot.system_frequency_hz:.2f} Hz below warning threshold"
            })
        elif snapshot.system_frequency_hz > th.freq_warn_high:
            alerts.append({
                "type": "HIGH_FREQUENCY",
                "severity": "WARNING" if snapshot.system_frequency_hz < th.freq_trip_high else "CRITICAL",
                "value": snapshot.system_frequency_hz,
                "threshold": th.freq_warn_high,
                "message": f"System frequency {snapshot.system_frequency_hz:.2f} Hz above warning threshold"
            })
        
        # Voltage alerts
        if snapshot.min_voltage_pu < th.volt_warn_low:
            alerts.append({
                "type": "LOW_VOLTAGE",
                "severity": "WARNING" if snapshot.min_voltage_pu > th.volt_trip_low else "CRITICAL",
                "value": snapshot.min_voltage_pu,
                "threshold": th.volt_warn_low,
                "message": f"Minimum voltage {snapshot.min_voltage_pu:.3f} pu below warning threshold"
            })
        
        if snapshot.max_voltage_pu > th.volt_warn_high:
            alerts.append({
                "type": "HIGH_VOLTAGE",
                "severity": "WARNING" if snapshot.max_voltage_pu < th.volt_trip_high else "CRITICAL",
                "value": snapshot.max_voltage_pu,
                "threshold": th.volt_warn_high,
                "message": f"Maximum voltage {snapshot.max_voltage_pu:.3f} pu above warning threshold"
            })
        
        # Overload alerts
        if snapshot.num_overloads > 0:
            alerts.append({
                "type": "OVERLOAD",
                "severity": "CRITICAL",
                "value": snapshot.num_overloads,
                "message": f"{snapshot.num_overloads} line/transformer overloads detected"
            })
        
        # Log and store alerts
        for alert in alerts:
            if alert["severity"] == "CRITICAL":
                logger.error(f"ALERT [{alert['type']}]: {alert['message']}")
            else:
                logger.warning(f"ALERT [{alert['type']}]: {alert['message']}")
        
        self._alerts.extend(alerts)
    
    def _add_to_history(self, snapshot: GridMetricsSnapshot) -> None:
        """Add snapshot to history, maintaining size limit"""
        self.metrics_history.append(snapshot)
        
        # Trim history if needed
        if len(self.metrics_history) > self.history_size:
            self.metrics_history = self.metrics_history[-self.history_size:]
    
    def get_health_score(self) -> float:
        """
        Calculate overall grid health score (0-100)
        
        Returns:
            Health score from 0 (critical) to 100 (optimal)
        """
        if not self.metrics_history:
            return 100.0  # No data = assume healthy
        
        latest = self.metrics_history[-1]
        th = self.thresholds
        
        score = 100.0
        
        # Frequency score (0-30 points)
        freq_deviation = abs(latest.system_frequency_hz - th.freq_nominal)
        freq_score = max(0, 30 - (freq_deviation * 60))  # Lose 6 points per 0.1 Hz
        score = score - 30 + freq_score
        
        # Voltage score (0-30 points)
        volt_deviation = max(
            abs(latest.min_voltage_pu - th.volt_nominal),
            abs(latest.max_voltage_pu - th.volt_nominal)
        )
        volt_score = max(0, 30 - (volt_deviation * 300))  # Lose 3 points per 0.01 pu
        score = score - 30 + volt_score
        
        # Violations score (0-20 points)
        violation_penalty = (latest.num_voltage_violations + latest.num_overloads) * 5
        score = score - min(20, violation_penalty)
        
        # Losses score (0-20 points) - penalize > 5% losses
        if latest.total_generation_mw > 0:
            loss_pct = (latest.total_losses_mw / latest.total_generation_mw) * 100
            if loss_pct > 5:
                score = score - min(20, (loss_pct - 5) * 4)
        
        return max(0, min(100, score))
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """Get most recent alerts"""
        return self._alerts[-limit:]
    
    def clear_alerts(self) -> None:
        """Clear alert history"""
        self._alerts = []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistical summary of recent metrics
        
        Returns:
            Dictionary with min/max/avg statistics
        """
        if not self.metrics_history:
            return {}
        
        recent = self.metrics_history[-100:]  # Last 100 snapshots
        
        return {
            "sample_count": len(recent),
            "time_range": {
                "start": recent[0].timestamp.isoformat(),
                "end": recent[-1].timestamp.isoformat()
            },
            "frequency_hz": {
                "min": min(s.system_frequency_hz for s in recent),
                "max": max(s.system_frequency_hz for s in recent),
                "avg": sum(s.system_frequency_hz for s in recent) / len(recent)
            },
            "voltage_pu": {
                "min": min(s.min_voltage_pu for s in recent),
                "max": max(s.max_voltage_pu for s in recent),
                "avg": sum(s.avg_voltage_pu for s in recent) / len(recent)
            },
            "generation_mw": {
                "min": min(s.total_generation_mw for s in recent),
                "max": max(s.total_generation_mw for s in recent),
                "avg": sum(s.total_generation_mw for s in recent) / len(recent)
            },
            "health_score": self.get_health_score()
        }


# Backward compatibility - keep old class name as alias
# DEPRECATED: Use GridMonitor instead
class GridManager(GridMonitor):
    """
    DEPRECATED: Use GridMonitor instead.
    
    This class is kept for backward compatibility only.
    """
    def __init__(self, num_nodes: int = 10, **kwargs):
        logger.warning(
            "GridManager is deprecated. Use GridMonitor instead. "
            "The num_nodes parameter is ignored - monitoring is based on SuperGrid."
        )
        super().__init__(**kwargs)
