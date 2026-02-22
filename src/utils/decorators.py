"""
DEMS Decorators
Reusable function decorators for logging, validation, and error handling
"""

import logging
import time
import sys
from functools import wraps
from typing import Callable, Any, Optional, TypeVar

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    try:
        from typing_extensions import ParamSpec
    except ImportError:
        # Fallback: define a minimal ParamSpec stub for Python 3.9
        class _ParamSpecStub:
            def __init__(self, name):
                self.args = Any
                self.kwargs = Any
        ParamSpec = _ParamSpecStub  # type: ignore

P = ParamSpec('P')
T = TypeVar('T')

logger = logging.getLogger(__name__)


def log_operation(func: Callable[P, T]) -> Callable[P, T]:
    """
    Decorator to log grid operations with execution details
    
    Logs function entry, successful completion, and any errors.
    """
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        func_name = func.__name__
        logger.info(f"Starting {func_name}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"Completed {func_name} successfully")
            return result
        except Exception as e:
            logger.error(f"Error in {func_name}: {str(e)}")
            raise
    return wrapper


def measure_performance(func: Callable[P, T]) -> Callable[P, T]:
    """
    Decorator to measure and log execution time
    
    Logs execution time at DEBUG level in milliseconds.
    """
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        logger.debug(f"{func.__name__} took {elapsed*1000:.2f}ms")
        return result
    return wrapper


def validate_converged(func: Callable[P, T]) -> Callable[P, T]:
    """
    Decorator to ensure power flow has converged before state operations.
    
    Requires the decorated method's first argument (self) to have:
    - _last_result attribute with converged and error_message properties
    
    Raises:
        RuntimeError: If power flow has not been run or did not converge
    """
    @wraps(func)
    def wrapper(self: Any, *args: P.args, **kwargs: P.kwargs) -> T:
        if not hasattr(self, "_last_result"):
            raise RuntimeError(
                f"{func.__name__} requires 'self' to define a '_last_result' "
                "attribute with 'converged' and 'error_message' properties."
            )
        last_result = self._last_result
        if last_result is None:
            raise RuntimeError(
                f"{func.__name__} called before running power flow. "
                "Call run_power_flow() first."
            )
        if not last_result.converged:
            raise RuntimeError(
                f"{func.__name__} called with non-converged power flow. "
                f"Last power flow converged={last_result.converged}, "
                f"error={last_result.error_message}"
            )
        return func(self, *args, **kwargs)
    return wrapper


def safe_grid_operation(func: Callable[P, T]) -> Callable[P, T]:
    """
    Decorator for safe grid operations with automatic error recovery
    
    On failure, attempts to reset the grid and retry the operation once.
    Requires the decorated method's first argument (self) to have a reset() method.
    """
    @wraps(func)
    def wrapper(self: Any, *args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            logger.error(f"Grid operation {func.__name__} failed: {str(e)}")
            logger.info("Attempting automatic recovery...")
            try:
                self.reset()
                logger.info("Grid reset successful, retrying operation")
                return func(self, *args, **kwargs)
            except Exception as recovery_error:
                logger.critical(f"Recovery failed: {str(recovery_error)}")
                raise
    return wrapper


def retry_on_failure(
    max_retries: int = 3,
    delay: float = 0.1,
    exceptions: tuple = (Exception,),
    backoff: float = 2.0
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to retry a function on failure with exponential backoff
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        exceptions: Tuple of exception types to catch and retry
        backoff: Multiplier for delay after each retry
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            current_delay = delay
            last_exception: Optional[Exception] = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                            f"Retrying in {current_delay:.2f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries + 1} attempts")
            
            raise last_exception  # type: ignore
        return wrapper
    return decorator
