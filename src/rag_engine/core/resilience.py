"""
Error handling and resilience patterns for the RAG engine
"""

import time
import asyncio
import threading
from typing import Any, Callable, Optional, Dict, List, Union, Type
from functools import wraps
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
import random
import contextlib
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from .logging import get_logger
from .monitoring import record_metric
from .exceptions import RAGEngineError, ExternalServiceError, ConfigurationError, ResilienceError


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class RetryConfig:
    """Configuration for retry logic"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: List[Type[Exception]] = field(default_factory=lambda: [Exception])
    timeout: Optional[float] = None  # Per-attempt timeout


@dataclass
class BulkheadConfig:
    """Configuration for bulkhead isolation pattern"""
    max_concurrent_calls: int = 10
    queue_size: int = 100
    timeout: float = 30.0
    name: str = "default"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: Type[Exception] = Exception
    name: str = "default"


class CircuitBreaker:
    """Circuit breaker implementation for external service calls"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.logger = get_logger(f"circuit_breaker.{config.name}")
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.logger.info(f"Circuit breaker {self.config.name} transitioning to HALF_OPEN")
                else:
                    self.logger.warning(f"Circuit breaker {self.config.name} is OPEN, rejecting call")
                    record_metric(f"circuit_breaker_rejected_calls", 1.0, "counter", {"name": self.config.name})
                    raise ExternalServiceError(f"Circuit breaker {self.config.name} is open")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt to reset"""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = datetime.now(timezone.utc) - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.recovery_timeout
    
    def _on_success(self):
        """Handle successful call"""
        with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
                self.logger.info(f"Circuit breaker {self.config.name} reset to CLOSED")
            
            self.failure_count = 0
            record_metric(f"circuit_breaker_successful_calls", 1.0, "counter", {"name": self.config.name})
    
    def _on_failure(self):
        """Handle failed call"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now(timezone.utc)
            
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                self.logger.warning(f"Circuit breaker {self.config.name} opened after {self.failure_count} failures")
            
            record_metric(f"circuit_breaker_failed_calls", 1.0, "counter", {"name": self.config.name})
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        return {
            "name": self.config.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "failure_threshold": self.config.failure_threshold,
            "recovery_timeout": self.config.recovery_timeout
        }


class RetryHandler:
    """Handles retry logic with exponential backoff"""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.logger = get_logger("retry_handler")
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                # Apply timeout if configured
                if self.config.timeout:
                    timeout_handler = TimeoutHandler(self.config.timeout)
                    result = timeout_handler.execute(func, *args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                if attempt > 0:
                    self.logger.info(f"Function succeeded on attempt {attempt + 1}")
                    record_metric("retry_success", 1.0, "counter", {"attempt": str(attempt + 1)})
                return result
            
            except Exception as e:
                last_exception = e
                
                # Check if exception is retryable
                if not any(isinstance(e, exc_type) for exc_type in self.config.retryable_exceptions):
                    self.logger.error(f"Non-retryable exception: {e}")
                    raise
                
                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    self.logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                    record_metric("retry_attempt", 1.0, "counter", {"attempt": str(attempt + 1)})
                    time.sleep(delay)
                else:
                    self.logger.error(f"All {self.config.max_attempts} attempts failed")
                    record_metric("retry_exhausted", 1.0, "counter")
        
        raise last_exception
    
    async def execute_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with retry logic"""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                # Apply timeout if configured
                if self.config.timeout:
                    timeout_handler = TimeoutHandler(self.config.timeout)
                    result = await timeout_handler.execute_async(func, *args, **kwargs)
                else:
                    result = await func(*args, **kwargs)
                
                if attempt > 0:
                    self.logger.info(f"Async function succeeded on attempt {attempt + 1}")
                    record_metric("async_retry_success", 1.0, "counter", {"attempt": str(attempt + 1)})
                return result
            
            except Exception as e:
                last_exception = e
                
                if not any(isinstance(e, exc_type) for exc_type in self.config.retryable_exceptions):
                    self.logger.error(f"Non-retryable async exception: {e}")
                    raise
                
                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    self.logger.warning(f"Async attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                    record_metric("async_retry_attempt", 1.0, "counter", {"attempt": str(attempt + 1)})
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"All {self.config.max_attempts} async attempts failed")
                    record_metric("async_retry_exhausted", 1.0, "counter")
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for next retry attempt"""
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        delay = min(delay, self.config.max_delay)
        
        if self.config.jitter:
            # Add jitter to prevent thundering herd
            jitter = random.uniform(0, delay * 0.1)
            delay += jitter
        
        return delay


class Bulkhead:
    """Bulkhead isolation pattern to prevent resource exhaustion"""
    
    def __init__(self, config: BulkheadConfig):
        self.config = config
        self.executor = ThreadPoolExecutor(
            max_workers=config.max_concurrent_calls,
            thread_name_prefix=f"bulkhead_{config.name}"
        )
        self.logger = get_logger(f"bulkhead.{config.name}")
        self._active_calls = 0
        self._lock = threading.Lock()
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with bulkhead isolation"""
        with self._lock:
            if self._active_calls >= self.config.max_concurrent_calls:
                self.logger.warning(f"Bulkhead {self.config.name} at capacity, rejecting call")
                record_metric("bulkhead_rejected_calls", 1.0, "counter", {"name": self.config.name})
                raise ResilienceError(f"Bulkhead {self.config.name} at capacity")
            
            self._active_calls += 1
        
        try:
            future = self.executor.submit(func, *args, **kwargs)
            result = future.result(timeout=self.config.timeout)
            record_metric("bulkhead_successful_calls", 1.0, "counter", {"name": self.config.name})
            return result
        except FutureTimeoutError:
            self.logger.error(f"Bulkhead {self.config.name} call timed out after {self.config.timeout}s")
            record_metric("bulkhead_timeout_calls", 1.0, "counter", {"name": self.config.name})
            raise ResilienceError(f"Bulkhead {self.config.name} call timed out")
        except Exception as e:
            self.logger.error(f"Bulkhead {self.config.name} call failed: {e}")
            record_metric("bulkhead_failed_calls", 1.0, "counter", {"name": self.config.name})
            raise
        finally:
            with self._lock:
                self._active_calls -= 1
    
    def get_status(self) -> Dict[str, Any]:
        """Get bulkhead status"""
        return {
            "name": self.config.name,
            "max_concurrent_calls": self.config.max_concurrent_calls,
            "active_calls": self._active_calls,
            "timeout": self.config.timeout
        }
    
    def shutdown(self):
        """Shutdown the bulkhead executor"""
        self.executor.shutdown(wait=True)


class TimeoutHandler:
    """Handles function execution timeouts"""
    
    def __init__(self, timeout: float):
        self.timeout = timeout
        self.logger = get_logger("timeout_handler")
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with timeout"""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                result = future.result(timeout=self.timeout)
                return result
            except FutureTimeoutError:
                self.logger.error(f"Function execution timed out after {self.timeout}s")
                record_metric("timeout_exceeded", 1.0, "counter", {"timeout": str(self.timeout)})
                raise ResilienceError(f"Function execution timed out after {self.timeout}s")
    
    async def execute_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with timeout"""
        try:
            result = await asyncio.wait_for(func(*args, **kwargs), timeout=self.timeout)
            return result
        except asyncio.TimeoutError:
            self.logger.error(f"Async function execution timed out after {self.timeout}s")
            record_metric("async_timeout_exceeded", 1.0, "counter", {"timeout": str(self.timeout)})
            raise ResilienceError(f"Async function execution timed out after {self.timeout}s")


class GracefulDegradation:
    """Handles graceful degradation strategies"""
    
    def __init__(self):
        self.fallback_strategies: Dict[str, Callable] = {}
        self.degradation_levels: Dict[str, List[Callable]] = {}
        self.logger = get_logger("graceful_degradation")
    
    def register_fallback(self, operation: str, fallback_func: Callable):
        """Register a fallback strategy for an operation"""
        self.fallback_strategies[operation] = fallback_func
        self.logger.info(f"Registered fallback strategy for operation: {operation}")
    
    def register_degradation_levels(self, operation: str, degradation_funcs: List[Callable]):
        """Register multiple degradation levels for an operation"""
        self.degradation_levels[operation] = degradation_funcs
        self.logger.info(f"Registered {len(degradation_funcs)} degradation levels for operation: {operation}")
    
    def execute_with_fallback(self, operation: str, primary_func: Callable, *args, **kwargs) -> Any:
        """Execute function with fallback strategy"""
        try:
            return primary_func(*args, **kwargs)
        except Exception as e:
            self.logger.warning(f"Primary operation '{operation}' failed: {e}")
            
            # Try degradation levels first if available
            if operation in self.degradation_levels:
                for level, degradation_func in enumerate(self.degradation_levels[operation]):
                    try:
                        self.logger.info(f"Trying degradation level {level + 1} for operation: {operation}")
                        result = degradation_func(*args, **kwargs)
                        record_metric("degradation_level_used", 1.0, "counter", {
                            "operation": operation, 
                            "level": str(level + 1)
                        })
                        return result
                    except Exception as degradation_error:
                        self.logger.warning(f"Degradation level {level + 1} failed for '{operation}': {degradation_error}")
                        continue
            
            # Fall back to simple fallback if degradation levels fail or don't exist
            if operation in self.fallback_strategies:
                self.logger.info(f"Executing fallback strategy for operation: {operation}")
                record_metric("fallback_executed", 1.0, "counter", {"operation": operation})
                try:
                    return self.fallback_strategies[operation](*args, **kwargs)
                except Exception as fallback_error:
                    self.logger.error(f"Fallback strategy for '{operation}' also failed: {fallback_error}")
                    record_metric("fallback_failed", 1.0, "counter", {"operation": operation})
                    raise
            else:
                self.logger.error(f"No fallback strategy registered for operation: {operation}")
                raise


class ResilienceManager:
    """Central manager for all resilience patterns"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_handlers: Dict[str, RetryHandler] = {}
        self.bulkheads: Dict[str, Bulkhead] = {}
        self.degradation = GracefulDegradation()
        self.logger = get_logger("resilience_manager")
    
    def create_circuit_breaker(self, name: str, config: CircuitBreakerConfig) -> CircuitBreaker:
        """Create and register a circuit breaker"""
        config.name = name
        breaker = CircuitBreaker(config)
        self.circuit_breakers[name] = breaker
        self.logger.info(f"Created circuit breaker: {name}")
        return breaker
    
    def create_retry_handler(self, name: str, config: RetryConfig) -> RetryHandler:
        """Create and register a retry handler"""
        handler = RetryHandler(config)
        self.retry_handlers[name] = handler
        self.logger.info(f"Created retry handler: {name}")
        return handler
    
    def create_bulkhead(self, name: str, config: BulkheadConfig) -> Bulkhead:
        """Create and register a bulkhead"""
        config.name = name
        bulkhead = Bulkhead(config)
        self.bulkheads[name] = bulkhead
        self.logger.info(f"Created bulkhead: {name}")
        return bulkhead
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name"""
        return self.circuit_breakers.get(name)
    
    def get_retry_handler(self, name: str) -> Optional[RetryHandler]:
        """Get retry handler by name"""
        return self.retry_handlers.get(name)
    
    def get_bulkhead(self, name: str) -> Optional[Bulkhead]:
        """Get bulkhead by name"""
        return self.bulkheads.get(name)
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all resilience components"""
        return {
            "circuit_breakers": {name: cb.get_state() for name, cb in self.circuit_breakers.items()},
            "retry_handlers": list(self.retry_handlers.keys()),
            "bulkheads": {name: bh.get_status() for name, bh in self.bulkheads.items()},
            "fallback_strategies": list(self.degradation.fallback_strategies.keys()),
            "degradation_levels": list(self.degradation.degradation_levels.keys())
        }
    
    def shutdown(self):
        """Shutdown all resilience components"""
        for bulkhead in self.bulkheads.values():
            bulkhead.shutdown()
        self.logger.info("All resilience components shut down")


# Global resilience manager
_resilience_manager = None


def get_resilience_manager() -> ResilienceManager:
    """Get the global resilience manager instance"""
    global _resilience_manager
    if _resilience_manager is None:
        _resilience_manager = ResilienceManager()
    return _resilience_manager


def with_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator to add circuit breaker protection to a function"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_resilience_manager()
            breaker = manager.get_circuit_breaker(name)
            
            if breaker is None:
                breaker_config = config or CircuitBreakerConfig(name=name)
                breaker = manager.create_circuit_breaker(name, breaker_config)
            
            return breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator


def with_retry(name: str, config: Optional[RetryConfig] = None):
    """Decorator to add retry logic to a function"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_resilience_manager()
            handler = manager.get_retry_handler(name)
            
            if handler is None:
                retry_config = config or RetryConfig()
                handler = manager.create_retry_handler(name, retry_config)
            
            return handler.execute(func, *args, **kwargs)
        return wrapper
    return decorator


def with_fallback(operation: str, fallback_func: Callable):
    """Decorator to add fallback strategy to a function"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_resilience_manager()
            manager.degradation.register_fallback(operation, fallback_func)
            return manager.degradation.execute_with_fallback(operation, func, *args, **kwargs)
        return wrapper
    return decorator


def resilient_call(
    func: Callable,
    circuit_breaker_name: Optional[str] = None,
    retry_name: Optional[str] = None,
    fallback_operation: Optional[str] = None,
    fallback_func: Optional[Callable] = None,
    *args,
    **kwargs
) -> Any:
    """
    Execute a function with full resilience patterns
    
    Args:
        func: Function to execute
        circuit_breaker_name: Name of circuit breaker to use
        retry_name: Name of retry handler to use
        fallback_operation: Name of fallback operation
        fallback_func: Fallback function to use
        *args, **kwargs: Arguments to pass to the function
    
    Returns:
        Result of function execution
    """
    manager = get_resilience_manager()
    
    # Create a wrapped function that applies all patterns
    def execute_with_patterns():
        current_func = func
        
        # Apply fallback if specified
        if fallback_operation and fallback_func:
            manager.degradation.register_fallback(fallback_operation, fallback_func)
            current_func = lambda: manager.degradation.execute_with_fallback(fallback_operation, func, *args, **kwargs)
        
        # Apply retry if specified
        if retry_name:
            handler = manager.get_retry_handler(retry_name)
            if handler is None:
                handler = manager.create_retry_handler(retry_name, RetryConfig())
            
            if fallback_operation and fallback_func:
                return handler.execute(current_func)
            else:
                return handler.execute(func, *args, **kwargs)
        
        # Apply circuit breaker if specified
        if circuit_breaker_name:
            breaker = manager.get_circuit_breaker(circuit_breaker_name)
            if breaker is None:
                breaker = manager.create_circuit_breaker(circuit_breaker_name, CircuitBreakerConfig(name=circuit_breaker_name))
            
            if retry_name or (fallback_operation and fallback_func):
                return breaker.call(current_func)
            else:
                return breaker.call(func, *args, **kwargs)
        
        # No patterns applied, just execute the function
        if fallback_operation and fallback_func:
            return current_func()
        else:
            return func(*args, **kwargs)
    
    return execute_with_patterns()


def with_bulkhead(name: str, config: Optional[BulkheadConfig] = None):
    """Decorator to add bulkhead isolation to a function"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_resilience_manager()
            bulkhead = manager.get_bulkhead(name)
            
            if bulkhead is None:
                bulkhead_config = config or BulkheadConfig(name=name)
                bulkhead = manager.create_bulkhead(name, bulkhead_config)
            
            return bulkhead.execute(func, *args, **kwargs)
        return wrapper
    return decorator


def with_timeout(timeout: float):
    """Decorator to add timeout to a function"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            timeout_handler = TimeoutHandler(timeout)
            return timeout_handler.execute(func, *args, **kwargs)
        return wrapper
    return decorator


def with_degradation_levels(operation: str, degradation_funcs: List[Callable]):
    """Decorator to add multiple degradation levels to a function"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_resilience_manager()
            manager.degradation.register_degradation_levels(operation, degradation_funcs)
            return manager.degradation.execute_with_fallback(operation, func, *args, **kwargs)
        return wrapper
    return decorator


# Context managers for resilience patterns
@contextlib.contextmanager
def circuit_breaker_context(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Context manager for circuit breaker protection"""
    manager = get_resilience_manager()
    breaker = manager.get_circuit_breaker(name)
    
    if breaker is None:
        breaker_config = config or CircuitBreakerConfig(name=name)
        breaker = manager.create_circuit_breaker(name, breaker_config)
    
    yield breaker


@contextlib.contextmanager
def bulkhead_context(name: str, config: Optional[BulkheadConfig] = None):
    """Context manager for bulkhead isolation"""
    manager = get_resilience_manager()
    bulkhead = manager.get_bulkhead(name)
    
    if bulkhead is None:
        bulkhead_config = config or BulkheadConfig(name=name)
        bulkhead = manager.create_bulkhead(name, bulkhead_config)
    
    yield bulkhead


# Advanced resilience patterns
class AdaptiveCircuitBreaker(CircuitBreaker):
    """Circuit breaker that adapts its thresholds based on success rate"""
    
    def __init__(self, config: CircuitBreakerConfig, adaptation_window: int = 100):
        super().__init__(config)
        self.adaptation_window = adaptation_window
        self.success_history: List[bool] = []
        self.original_threshold = config.failure_threshold
    
    def _on_success(self):
        super()._on_success()
        self.success_history.append(True)
        self._adapt_threshold()
    
    def _on_failure(self):
        super()._on_failure()
        self.success_history.append(False)
        self._adapt_threshold()
    
    def _adapt_threshold(self):
        """Adapt failure threshold based on recent success rate"""
        if len(self.success_history) > self.adaptation_window:
            self.success_history = self.success_history[-self.adaptation_window:]
        
        if len(self.success_history) >= 20:  # Minimum samples for adaptation
            success_rate = sum(self.success_history) / len(self.success_history)
            
            if success_rate > 0.95:  # Very high success rate
                self.config.failure_threshold = min(self.original_threshold + 2, 10)
            elif success_rate < 0.8:  # Low success rate
                self.config.failure_threshold = max(self.original_threshold - 1, 2)
            else:
                self.config.failure_threshold = self.original_threshold


class RateLimiter:
    """Token bucket rate limiter for controlling request rates"""
    
    def __init__(self, max_tokens: int, refill_rate: float, name: str = "default"):
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate  # tokens per second
        self.tokens = max_tokens
        self.last_refill = time.time()
        self.name = name
        self.logger = get_logger(f"rate_limiter.{name}")
        self._lock = threading.Lock()
    
    def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens from the bucket"""
        with self._lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                record_metric("rate_limiter_allowed", 1.0, "counter", {"name": self.name})
                return True
            else:
                record_metric("rate_limiter_rejected", 1.0, "counter", {"name": self.name})
                self.logger.warning(f"Rate limiter {self.name} rejected request - insufficient tokens")
                return False
    
    def _refill(self):
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate
        
        self.tokens = min(self.max_tokens, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def get_status(self) -> Dict[str, Any]:
        """Get rate limiter status"""
        return {
            "name": self.name,
            "max_tokens": self.max_tokens,
            "current_tokens": self.tokens,
            "refill_rate": self.refill_rate
        }


def with_rate_limit(max_tokens: int, refill_rate: float, name: str = "default"):
    """Decorator to add rate limiting to a function"""
    rate_limiter = RateLimiter(max_tokens, refill_rate, name)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not rate_limiter.acquire():
                raise ResilienceError(f"Rate limit exceeded for {name}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Health check integration
class HealthAwareCircuitBreaker(CircuitBreaker):
    """Circuit breaker that considers external health checks"""
    
    def __init__(self, config: CircuitBreakerConfig, health_check_func: Optional[Callable] = None):
        super().__init__(config)
        self.health_check_func = health_check_func
        self.last_health_check = None
        self.health_check_interval = 30.0  # seconds
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with health-aware circuit breaker protection"""
        # Check external health if configured
        if self.health_check_func and self._should_check_health():
            try:
                if not self.health_check_func():
                    self.logger.warning(f"Health check failed for {self.config.name}, opening circuit")
                    self.state = CircuitBreakerState.OPEN
                    self.last_failure_time = datetime.now(timezone.utc)
                    raise ExternalServiceError(f"Health check failed for {self.config.name}")
                else:
                    self.last_health_check = time.time()
            except Exception as e:
                self.logger.error(f"Health check error for {self.config.name}: {e}")
        
        return super().call(func, *args, **kwargs)
    
    def _should_check_health(self) -> bool:
        """Check if health check should be performed"""
        if self.last_health_check is None:
            return True
        
        return (time.time() - self.last_health_check) > self.health_check_interval