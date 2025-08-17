# Error Handling and Resilience

This document describes the comprehensive error handling and resilience patterns implemented in the RAG engine to ensure production-ready reliability and fault tolerance.

## Overview

The RAG engine implements multiple resilience patterns to handle various failure scenarios:

- **Circuit Breaker Pattern**: Prevents cascading failures by temporarily blocking calls to failing services
- **Retry Logic with Exponential Backoff**: Automatically retries failed operations with increasing delays
- **Graceful Degradation**: Provides fallback strategies when primary operations fail
- **Bulkhead Isolation**: Isolates resources to prevent resource exhaustion
- **Rate Limiting**: Controls request rates to prevent overwhelming services
- **Timeout Handling**: Prevents operations from hanging indefinitely
- **Health-Aware Patterns**: Integrates with health checks for intelligent failure handling

## Core Components

### Circuit Breaker

The circuit breaker pattern prevents cascading failures by monitoring the failure rate of external service calls and temporarily blocking requests when failures exceed a threshold.

```python
from src.rag_engine.core.resilience import CircuitBreakerConfig, get_resilience_manager

# Configure circuit breaker
config = CircuitBreakerConfig(
    failure_threshold=5,        # Open circuit after 5 failures
    recovery_timeout=60.0,      # Try to recover after 60 seconds
    expected_exception=Exception,
    name="external_api"
)

manager = get_resilience_manager()
breaker = manager.create_circuit_breaker("external_api", config)

# Use circuit breaker
try:
    result = breaker.call(external_api_function, *args, **kwargs)
except ExternalServiceError:
    # Circuit is open, handle gracefully
    pass
```

#### Circuit Breaker States

- **CLOSED**: Normal operation, requests pass through
- **OPEN**: Circuit is open, requests are rejected immediately
- **HALF_OPEN**: Testing if service has recovered

### Retry Logic with Exponential Backoff

Automatically retries failed operations with configurable backoff strategies.

```python
from src.rag_engine.core.resilience import RetryConfig, RetryHandler

config = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=60.0,
    exponential_base=2.0,
    jitter=True,
    retryable_exceptions=[ConnectionError, TimeoutError]
)

handler = RetryHandler(config)
result = handler.execute(unreliable_function, *args, **kwargs)
```

#### Retry Features

- **Exponential Backoff**: Delays increase exponentially between retries
- **Jitter**: Random variation in delays to prevent thundering herd
- **Exception Filtering**: Only retry specific exception types
- **Timeout Support**: Per-attempt timeouts
- **Async Support**: Full async/await support

### Graceful Degradation

Provides fallback strategies when primary operations fail.

```python
from src.rag_engine.core.resilience import GracefulDegradation

degradation = GracefulDegradation()

def primary_service():
    # Primary implementation
    return call_external_api()

def fallback_service():
    # Fallback implementation
    return get_cached_response()

# Register fallback
degradation.register_fallback("api_operation", fallback_service)

# Execute with fallback
result = degradation.execute_with_fallback("api_operation", primary_service)
```

#### Degradation Levels

Support for multiple degradation levels:

```python
def level_1_degradation():
    return get_reduced_quality_response()

def level_2_degradation():
    return get_minimal_response()

degradation.register_degradation_levels("complex_operation", [
    level_1_degradation,
    level_2_degradation
])
```

### Bulkhead Isolation

Isolates resources to prevent resource exhaustion from affecting the entire system.

```python
from src.rag_engine.core.resilience import BulkheadConfig, Bulkhead

config = BulkheadConfig(
    max_concurrent_calls=10,
    queue_size=100,
    timeout=30.0,
    name="api_bulkhead"
)

bulkhead = Bulkhead(config)
result = bulkhead.execute(resource_intensive_function, *args, **kwargs)
```

### Rate Limiting

Controls request rates using token bucket algorithm.

```python
from src.rag_engine.core.resilience import RateLimiter, with_rate_limit

# Manual rate limiting
rate_limiter = RateLimiter(max_tokens=10, refill_rate=1.0)

if rate_limiter.acquire():
    result = make_api_call()
else:
    raise ResilienceError("Rate limit exceeded")

# Decorator-based rate limiting
@with_rate_limit(max_tokens=5, refill_rate=0.5)
def rate_limited_function():
    return expensive_operation()
```

### Timeout Handling

Prevents operations from hanging indefinitely.

```python
from src.rag_engine.core.resilience import TimeoutHandler, with_timeout

# Manual timeout handling
timeout_handler = TimeoutHandler(30.0)
result = timeout_handler.execute(slow_function, *args, **kwargs)

# Decorator-based timeout
@with_timeout(10.0)
def timeout_protected_function():
    return potentially_slow_operation()
```

## Advanced Patterns

### Adaptive Circuit Breaker

Circuit breaker that adapts its failure threshold based on success rate.

```python
from src.rag_engine.core.resilience import AdaptiveCircuitBreaker

breaker = AdaptiveCircuitBreaker(config, adaptation_window=100)
# Automatically adjusts failure threshold based on recent success rate
```

### Health-Aware Circuit Breaker

Circuit breaker that considers external health checks.

```python
from src.rag_engine.core.resilience import HealthAwareCircuitBreaker

def health_check():
    return check_service_health()

breaker = HealthAwareCircuitBreaker(config, health_check)
# Opens circuit if health check fails
```

## Decorators and Utilities

### Resilience Decorators

```python
from src.rag_engine.core.resilience import (
    with_circuit_breaker,
    with_retry,
    with_fallback,
    with_bulkhead,
    with_timeout,
    with_rate_limit
)

@with_circuit_breaker("api_service")
@with_retry("api_retry")
@with_timeout(30.0)
def resilient_api_call():
    return call_external_api()
```

### Comprehensive Resilient Call

Single function that applies multiple resilience patterns:

```python
from src.rag_engine.core.resilience import resilient_call

result = resilient_call(
    function_to_call,
    circuit_breaker_name="my_cb",
    retry_name="my_retry",
    fallback_operation="my_fallback",
    fallback_func=fallback_function,
    *args, **kwargs
)
```

### Context Managers

```python
from src.rag_engine.core.resilience import circuit_breaker_context, bulkhead_context

# Circuit breaker context
with circuit_breaker_context("api_cb") as breaker:
    result = breaker.call(api_function)

# Bulkhead context
with bulkhead_context("api_bulkhead") as bulkhead:
    result = bulkhead.execute(resource_function)
```

## Configuration

### Resilience Manager

Central manager for all resilience components:

```python
from src.rag_engine.core.resilience import get_resilience_manager

manager = get_resilience_manager()

# Create components
breaker = manager.create_circuit_breaker("api_cb", circuit_breaker_config)
handler = manager.create_retry_handler("api_retry", retry_config)
bulkhead = manager.create_bulkhead("api_bulkhead", bulkhead_config)

# Get status
status = manager.get_status()
print(f"Active circuit breakers: {list(status['circuit_breakers'].keys())}")

# Cleanup
manager.shutdown()
```

### Configuration Classes

All resilience patterns use configuration classes for type safety:

```python
from src.rag_engine.core.resilience import (
    CircuitBreakerConfig,
    RetryConfig,
    BulkheadConfig
)

# Circuit breaker configuration
cb_config = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=60.0,
    expected_exception=ExternalServiceError,
    name="my_service"
)

# Retry configuration
retry_config = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=60.0,
    exponential_base=2.0,
    jitter=True,
    retryable_exceptions=[ConnectionError, TimeoutError],
    timeout=30.0
)

# Bulkhead configuration
bulkhead_config = BulkheadConfig(
    max_concurrent_calls=10,
    queue_size=100,
    timeout=30.0,
    name="resource_pool"
)
```

## Error Classification

The system uses a hierarchy of exception types for proper error handling:

```python
from src.rag_engine.core.exceptions import (
    RAGEngineError,           # Base exception
    ExternalServiceError,     # External service failures
    ResilienceError,          # Resilience pattern errors
    ConfigurationError,       # Configuration issues
    # ... other specific exceptions
)
```

## Monitoring and Metrics

All resilience patterns integrate with the monitoring system:

```python
# Metrics are automatically recorded for:
# - Circuit breaker state changes
# - Retry attempts and outcomes
# - Rate limiting decisions
# - Timeout occurrences
# - Fallback usage

# Access metrics through the monitoring system
from src.rag_engine.core.monitoring import get_metrics_collector

collector = get_metrics_collector()
metrics = collector.get_metrics()
```

## Async Support

Full support for async operations:

```python
import asyncio
from src.rag_engine.core.resilience import RetryHandler, TimeoutHandler

# Async retry
handler = RetryHandler(retry_config)
result = await handler.execute_async(async_function, *args, **kwargs)

# Async timeout
timeout_handler = TimeoutHandler(30.0)
result = await timeout_handler.execute_async(async_function, *args, **kwargs)
```

## Best Practices

### 1. Choose Appropriate Patterns

- **Circuit Breaker**: For external service calls that can fail
- **Retry**: For transient failures (network issues, temporary unavailability)
- **Fallback**: When degraded functionality is acceptable
- **Bulkhead**: For resource-intensive operations
- **Rate Limiting**: To protect against overload

### 2. Configure Sensible Defaults

```python
# Production-ready defaults
circuit_breaker_config = CircuitBreakerConfig(
    failure_threshold=5,      # Allow some failures before opening
    recovery_timeout=60.0,    # Give service time to recover
    expected_exception=ExternalServiceError
)

retry_config = RetryConfig(
    max_attempts=3,           # Don't retry forever
    base_delay=1.0,          # Start with reasonable delay
    max_delay=60.0,          # Cap maximum delay
    exponential_base=2.0,    # Standard exponential backoff
    jitter=True              # Prevent thundering herd
)
```

### 3. Monitor and Alert

- Monitor circuit breaker state changes
- Alert on high retry rates
- Track fallback usage
- Monitor timeout occurrences

### 4. Test Failure Scenarios

```python
# Test circuit breaker opening
def test_circuit_breaker_opens():
    # Simulate failures and verify circuit opens
    pass

# Test retry exhaustion
def test_retry_exhaustion():
    # Verify behavior when all retries fail
    pass

# Test fallback activation
def test_fallback_activation():
    # Verify fallback is used when primary fails
    pass
```

### 5. Graceful Degradation Strategy

Design your system with multiple levels of degradation:

1. **Full functionality**: All services working
2. **Reduced functionality**: Some services failing, using fallbacks
3. **Minimal functionality**: Most services failing, basic responses only
4. **Emergency mode**: System barely functional but still responsive

## Example Usage

See `examples/example_error_handling_resilience.py` for comprehensive examples of all resilience patterns in action.

## Testing

The resilience system includes comprehensive tests covering:

- Unit tests for all resilience patterns
- Integration tests for pattern combinations
- Error scenario testing
- Performance and load testing
- Async operation testing

Run tests with:

```bash
uv run python -m pytest tests/core/test_resilience.py -v
```

## Performance Considerations

- Circuit breakers have minimal overhead when closed
- Retry logic adds latency but improves reliability
- Bulkheads use thread pools - configure appropriately
- Rate limiters use efficient token bucket algorithm
- Monitoring has minimal performance impact

## Troubleshooting

### Circuit Breaker Not Opening

- Check failure threshold configuration
- Verify exception types match expected_exception
- Monitor actual failure rates

### Retries Not Working

- Verify exception types are in retryable_exceptions list
- Check retry configuration (max_attempts, delays)
- Monitor retry metrics

### Fallbacks Not Activating

- Ensure fallback functions are registered
- Check exception handling in primary functions
- Verify fallback function signatures

### High Resource Usage

- Review bulkhead configurations
- Check for resource leaks in long-running operations
- Monitor thread pool usage

## Conclusion

The RAG engine's error handling and resilience system provides comprehensive protection against various failure modes while maintaining high performance and observability. By combining multiple resilience patterns, the system can handle complex failure scenarios and provide graceful degradation when needed.