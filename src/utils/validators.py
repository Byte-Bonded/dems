"""
DEMS Validators
Input validation utilities for grid operations
"""

from typing import Union, Optional, Sequence
from src.simulation.kundur import AreaID


def validate_area_id(area_id: str, valid_areas: Optional[Sequence[str]] = None) -> str:
    """
    Validate an area identifier
    
    Args:
        area_id: The area ID to validate
        valid_areas: Optional sequence of valid area IDs (defaults to Area1, Area2)
        
    Returns:
        The validated area ID
        
    Raises:
        ValueError: If area_id is not valid
    """
    if valid_areas is None:
        valid_areas = [a.value for a in AreaID]
    
    if area_id not in valid_areas:
        raise ValueError(
            f"Invalid area_id '{area_id}'. Must be one of: {valid_areas}"
        )
    return area_upper


def validate_numeric_range(
    value: Union[int, float],
    name: str,
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
    allow_zero: bool = True,
) -> Union[int, float]:
    """
    Validate a numeric value is within an acceptable range
    
    Args:
        value: The value to validate
        name: Name of the parameter (for error messages)
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        allow_zero: Whether zero is a valid value
        
    Returns:
        The validated value
        
    Raises:
        ValueError: If value is outside the valid range
        TypeError: If value is not numeric
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    
    if not allow_zero and value == 0:
        raise ValueError(f"{name} cannot be zero")
    
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be <= {max_value}, got {value}")
    
    return value


def validate_bus_index(
    bus_idx: int,
    max_bus: int,
    name: str = "bus_idx"
) -> int:
    """
    Validate a bus index is within the network
    
    Args:
        bus_idx: The bus index to validate
        max_bus: Maximum valid bus index
        name: Name of the parameter (for error messages)
        
    Returns:
        The validated bus index
        
    Raises:
        ValueError: If bus index is invalid
    """
    if not isinstance(bus_idx, int):
        raise TypeError(f"{name} must be an integer, got {type(bus_idx).__name__}")
    
    if bus_idx < 0:
        raise ValueError(f"{name} must be non-negative, got {bus_idx}")
    
    if bus_idx > max_bus:
        raise ValueError(f"{name} {bus_idx} exceeds network size (max: {max_bus})")
    
    return bus_idx


def validate_percentage(value: float, name: str = "value") -> float:
    """
    Validate a value is a valid percentage (0.0 to 1.0 or 0 to 100)
    
    Args:
        value: The percentage value
        name: Name of the parameter
        
    Returns:
        Normalized percentage (0.0 to 1.0)
        
    Raises:
        ValueError: If value is not a valid percentage
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    
    # Auto-convert if given as 0-100 range
    if value > 1.0 and value <= 100:
        value = value / 100.0
    
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0 and 1 (or 0-100), got {value}")
    
    return float(value)
