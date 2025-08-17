#!/usr/bin/env python3
"""
Example demonstrating comprehensive error handling and resilience patterns
in the RAG engine.

This example shows how to use:
- Circuit breaker pattern for external service calls
- Retry logic with exponential backoff
- Graceful degradation strategies
- Bulkhead isolation
- Rate limiting
- Timeout handling
- Comprehensive error handling scenarios
"""

import asyncio
import time
import random
from typing import List, Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_engine.core.resilience import (
    # Core resilience patterns
    CircuitBreakerConfig,
    RetryConfig,
    BulkheadConfig,
    
    # Resilience manager
    get_resilience_manager,
    
    # Decorators and utilities
    with_circuit_breaker,
    with_retry,
    with_fallback,
    with_bulkhead,
    with_timeout,
    with_rate_limit,
    resilient_call,
    
    # Advanced patterns
    RateLimiter,
    AdaptiveCircuitBreaker,
    HealthAwareCircuitBreaker,
    
    # Context managers
    circuit_breaker_context,
    bulkhead_context
)

from src.rag_engine.core.exceptions import (
    ExternalServiceError,
    ResilienceError
)


class ExternalAPIService:
    """Simulated external API service with various failure modes"""
    
    def __init__(self, failure_rate: float = 0.3):
        self.failure_rate = failure_rate
        self.call_count = 0
        self.is_healthy = True
    
    def call_api(self, data: str) -> str:
        """Simulate API call with potential failures"""
        self.call_count += 1
        
        if not self.is_healthy:
            raise ExternalServiceError("Service is unhealthy")
        
        if random.random() < self.failure_rate:
            failure_type = random.choice([
                "network_timeout",
                "service_unavailable", 
                "rate_limited",
                "internal_error"
            ])
            
            if failure_type == "network_timeout":
                time.sleep(0.1)  # Simulate slow response
                raise TimeoutError("Network timeout")
            elif failure_type == "service_unavailable":
                raise ExternalServiceError("Service temporarily unavailable")
            elif failure_type == "rate_limited":
                raise ExternalServiceError("Rate limit exceeded")
            else:
                raise Exception("Internal server error")
        
        return f"API response for: {data}"
    
    async def async_call_api(self, data: str) -> str:
        """Async version of API call"""
        await asyncio.sleep(0.01)  # Simulate network delay
        return self.call_api(data)
    
    def health_check(self) -> bool:
        """Health check for the service"""
        return self.is_healthy
    
    def set_health(self, healthy: bool):
        """Set service health status"""
        self.is_healthy = healthy


def demonstrate_circuit_breaker():
    """Demonstrate circuit breaker pattern"""
    print("\n=== Circuit Breaker Pattern ===")
    
    # Create a service that fails frequently
    service = ExternalAPIService(failure_rate=0.8)
    
    # Configure circuit breaker
    config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=2.0,
        expected_exception=Exception,
        name="api_service"
    )
    
    manager = get_resilience_manager()
    breaker = manager.create_circuit_breaker("api_service", config)
    
    print("Making calls to unreliable service...")
    
    for i in range(10):
        try:
            result = breaker.call(service.call_api, f"request_{i}")
            print(f"Call {i+1}: SUCCESS - {result}")
        except ExternalServiceError as e:
            print(f"Call {i+1}: REJECTED - {e}")
        except Exception as e:
            print(f"Call {i+1}: FAILED - {e}")
        
        # Show circuit breaker state
        state = breaker.get_state()
        print(f"  Circuit state: {state['state']}, failures: {state['failure_count']}")
        
        time.sleep(0.1)
    
    print(f"Total API calls made: {service.call_count}")


def demonstrate_retry_with_backoff():
    """Demonstrate retry logic with exponential backoff"""
    print("\n=== Retry with Exponential Backoff ===")
    
    service = ExternalAPIService(failure_rate=0.6)
    
    # Configure retry with exponential backoff
    config = RetryConfig(
        max_attempts=4,
        base_delay=0.1,
        max_delay=2.0,
        exponential_base=2.0,
        jitter=True,
        retryable_exceptions=[ExternalServiceError, TimeoutError]
    )
    
    manager = get_resilience_manager()
    handler = manager.create_retry_handler("api_retry", config)
    
    print("Attempting API call with retry logic...")
    start_time = time.time()
    
    try:
        result = handler.execute(service.call_api, "retry_test")
        print(f"SUCCESS: {result}")
    except Exception as e:
        print(f"FAILED after all retries: {e}")
    
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f}s")
    print(f"Total API calls made: {service.call_count}")


def demonstrate_graceful_degradation():
    """Demonstrate graceful degradation strategies"""
    print("\n=== Graceful Degradation ===")
    
    service = ExternalAPIService(failure_rate=1.0)  # Always fails
    
    def primary_operation(data: str) -> str:
        return service.call_api(data)
    
    def fallback_operation(data: str) -> str:
        return f"Cached response for: {data}"
    
    def emergency_fallback(data: str) -> str:
        return f"Emergency response for: {data}"
    
    # Using decorator approach
    @with_fallback("api_operation", fallback_operation)
    def api_call_with_fallback(data: str) -> str:
        return primary_operation(data)
    
    print("Primary service call (will fail)...")
    result = api_call_with_fallback("test_data")
    print(f"Result: {result}")
    
    # Using resilient_call with multiple fallback levels
    print("\nUsing resilient_call with comprehensive patterns...")
    result = resilient_call(
        lambda: primary_operation("comprehensive_test"),
        circuit_breaker_name="degradation_cb",
        retry_name="degradation_retry", 
        fallback_operation="api_degradation",
        fallback_func=lambda: emergency_fallback("comprehensive_test")
    )
    print(f"Result: {result}")


def demonstrate_bulkhead_isolation():
    """Demonstrate bulkhead isolation pattern"""
    print("\n=== Bulkhead Isolation ===")
    
    service = ExternalAPIService(failure_rate=0.2)
    
    # Configure bulkhead
    config = BulkheadConfig(
        max_concurrent_calls=3,
        timeout=1.0,
        name="api_bulkhead"
    )
    
    manager = get_resilience_manager()
    bulkhead = manager.create_bulkhead("api_bulkhead", config)
    
    def slow_operation(data: str) -> str:
        time.sleep(0.5)  # Simulate slow operation
        return service.call_api(data)
    
    print("Testing bulkhead isolation...")
    
    import threading
    results = []
    errors = []
    
    def worker(worker_id: int):
        try:
            result = bulkhead.execute(slow_operation, f"worker_{worker_id}")
            results.append(f"Worker {worker_id}: {result}")
        except Exception as e:
            errors.append(f"Worker {worker_id}: {e}")
    
    # Launch more workers than bulkhead capacity
    threads = []
    for i in range(6):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    print("Results:")
    for result in results:
        print(f"  {result}")
    
    print("Errors:")
    for error in errors:
        print(f"  {error}")
    
    # Show bulkhead status
    status = bulkhead.get_status()
    print(f"Bulkhead status: {status}")
    
    bulkhead.shutdown()


def demonstrate_rate_limiting():
    """Demonstrate rate limiting"""
    print("\n=== Rate Limiting ===")
    
    service = ExternalAPIService(failure_rate=0.1)
    
    # Create rate limiter (2 requests per second)
    rate_limiter = RateLimiter(max_tokens=2, refill_rate=2.0, name="api_limiter")
    
    @with_rate_limit(max_tokens=3, refill_rate=1.0, name="decorated_limiter")
    def rate_limited_api_call(data: str) -> str:
        return service.call_api(data)
    
    print("Testing manual rate limiter...")
    for i in range(5):
        if rate_limiter.acquire():
            try:
                result = service.call_api(f"manual_{i}")
                print(f"Request {i+1}: SUCCESS - {result}")
            except Exception as e:
                print(f"Request {i+1}: FAILED - {e}")
        else:
            print(f"Request {i+1}: RATE LIMITED")
        
        time.sleep(0.2)
    
    print("\nTesting decorator-based rate limiting...")
    for i in range(5):
        try:
            result = rate_limited_api_call(f"decorated_{i}")
            print(f"Request {i+1}: SUCCESS - {result}")
        except ResilienceError as e:
            print(f"Request {i+1}: RATE LIMITED - {e}")
        except Exception as e:
            print(f"Request {i+1}: FAILED - {e}")
        
        time.sleep(0.3)


def demonstrate_timeout_handling():
    """Demonstrate timeout handling"""
    print("\n=== Timeout Handling ===")
    
    service = ExternalAPIService(failure_rate=0.0)  # No failures, just slow
    
    def slow_operation(data: str) -> str:
        time.sleep(2.0)  # Simulate very slow operation
        return service.call_api(data)
    
    @with_timeout(1.0)
    def timeout_protected_call(data: str) -> str:
        return slow_operation(data)
    
    print("Testing timeout protection...")
    try:
        result = timeout_protected_call("timeout_test")
        print(f"SUCCESS: {result}")
    except ResilienceError as e:
        print(f"TIMEOUT: {e}")


async def demonstrate_async_resilience():
    """Demonstrate async resilience patterns"""
    print("\n=== Async Resilience Patterns ===")
    
    service = ExternalAPIService(failure_rate=0.5)
    
    # Configure async retry
    config = RetryConfig(
        max_attempts=3,
        base_delay=0.1,
        retryable_exceptions=[ExternalServiceError, TimeoutError]
    )
    
    manager = get_resilience_manager()
    handler = manager.create_retry_handler("async_retry", config)
    
    print("Testing async retry...")
    try:
        result = await handler.execute_async(service.async_call_api, "async_test")
        print(f"SUCCESS: {result}")
    except Exception as e:
        print(f"FAILED: {e}")


def demonstrate_health_aware_circuit_breaker():
    """Demonstrate health-aware circuit breaker"""
    print("\n=== Health-Aware Circuit Breaker ===")
    
    service = ExternalAPIService(failure_rate=0.1)
    
    config = CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=1.0,
        name="health_aware"
    )
    
    breaker = HealthAwareCircuitBreaker(config, service.health_check)
    
    print("Service is healthy - calls should succeed...")
    for i in range(3):
        try:
            result = breaker.call(service.call_api, f"healthy_{i}")
            print(f"Call {i+1}: SUCCESS - {result}")
        except Exception as e:
            print(f"Call {i+1}: FAILED - {e}")
    
    print("\nMarking service as unhealthy...")
    service.set_health(False)
    
    for i in range(3):
        try:
            result = breaker.call(service.call_api, f"unhealthy_{i}")
            print(f"Call {i+1}: SUCCESS - {result}")
        except Exception as e:
            print(f"Call {i+1}: FAILED - {e}")


def demonstrate_comprehensive_integration():
    """Demonstrate comprehensive integration of all patterns"""
    print("\n=== Comprehensive Integration ===")
    
    service = ExternalAPIService(failure_rate=0.4)
    
    def primary_service_call(data: str) -> str:
        return service.call_api(data)
    
    def fallback_service_call(data: str) -> str:
        return f"Fallback response for: {data}"
    
    # Configure comprehensive resilience
    manager = get_resilience_manager()
    
    # Custom retry configuration
    retry_config = RetryConfig(
        max_attempts=3,
        base_delay=0.1,
        exponential_base=2.0,
        retryable_exceptions=[ExternalServiceError, TimeoutError]
    )
    manager.create_retry_handler("comprehensive_retry", retry_config)
    
    # Custom circuit breaker configuration
    cb_config = CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=2.0
    )
    manager.create_circuit_breaker("comprehensive_cb", cb_config)
    
    print("Making comprehensive resilient calls...")
    
    for i in range(10):
        try:
            result = resilient_call(
                lambda: primary_service_call(f"comprehensive_{i}"),
                circuit_breaker_name="comprehensive_cb",
                retry_name="comprehensive_retry",
                fallback_operation="comprehensive_fallback",
                fallback_func=lambda: fallback_service_call(f"comprehensive_{i}")
            )
            print(f"Call {i+1}: SUCCESS - {result}")
        except Exception as e:
            print(f"Call {i+1}: FAILED - {e}")
        
        time.sleep(0.1)
    
    # Show final status
    status = manager.get_status()
    print(f"\nFinal resilience status:")
    print(f"Circuit breakers: {list(status['circuit_breakers'].keys())}")
    print(f"Retry handlers: {status['retry_handlers']}")
    print(f"Fallback strategies: {status['fallback_strategies']}")


def demonstrate_context_managers():
    """Demonstrate resilience context managers"""
    print("\n=== Context Managers ===")
    
    service = ExternalAPIService(failure_rate=0.3)
    
    # Using circuit breaker context manager
    print("Using circuit breaker context manager...")
    with circuit_breaker_context("context_cb") as breaker:
        for i in range(3):
            try:
                result = breaker.call(service.call_api, f"context_{i}")
                print(f"Call {i+1}: SUCCESS - {result}")
            except Exception as e:
                print(f"Call {i+1}: FAILED - {e}")
    
    # Using bulkhead context manager
    print("\nUsing bulkhead context manager...")
    with bulkhead_context("context_bulkhead") as bulkhead:
        try:
            result = bulkhead.execute(service.call_api, "bulkhead_test")
            print(f"Bulkhead call: SUCCESS - {result}")
        except Exception as e:
            print(f"Bulkhead call: FAILED - {e}")


async def main():
    """Main demonstration function"""
    print("RAG Engine Error Handling and Resilience Demonstration")
    print("=" * 60)
    
    # Run all demonstrations
    demonstrate_circuit_breaker()
    demonstrate_retry_with_backoff()
    demonstrate_graceful_degradation()
    demonstrate_bulkhead_isolation()
    demonstrate_rate_limiting()
    demonstrate_timeout_handling()
    await demonstrate_async_resilience()
    demonstrate_health_aware_circuit_breaker()
    demonstrate_comprehensive_integration()
    demonstrate_context_managers()
    
    print("\n" + "=" * 60)
    print("Demonstration completed!")
    
    # Show final system status
    manager = get_resilience_manager()
    status = manager.get_status()
    
    print(f"\nFinal System Status:")
    print(f"Active circuit breakers: {len(status['circuit_breakers'])}")
    print(f"Active retry handlers: {len(status['retry_handlers'])}")
    print(f"Active bulkheads: {len(status['bulkheads'])}")
    print(f"Registered fallback strategies: {len(status['fallback_strategies'])}")
    
    # Cleanup
    manager.shutdown()
    print("System shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())