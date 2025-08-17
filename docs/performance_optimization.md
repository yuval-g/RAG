# Performance Optimization Guide

This document describes the performance optimization features implemented in the RAG system, including connection pooling, caching, async operations, and scaling strategies.

## Overview

The RAG system includes several performance optimization features designed to improve throughput, reduce latency, and enable horizontal scaling:

1. **Connection Pooling** - Reuse connections to external services
2. **Caching Layer** - Cache frequently accessed data and results
3. **Async/Await Support** - Enable concurrent operations
4. **Batch Processing** - Process multiple items together for efficiency
5. **Performance Monitoring** - Track and analyze system performance

## Connection Pooling

### Overview
Connection pooling reduces the overhead of creating and destroying connections to external services like LLM providers, embedding services, and vector stores.

### Configuration
```yaml
# Performance Configuration
connection_pool_size: 20          # Maximum connections per pool
connection_timeout: 30.0          # Timeout for acquiring connections
enable_connection_pooling: true   # Enable/disable connection pooling
```

### Usage
```python
from src.rag_engine.core.performance import ConnectionPool, ConnectionPoolConfig

# Create connection pool
config = ConnectionPoolConfig(
    max_connections=10,
    min_connections=2,
    connection_timeout=30.0
)

def create_llm_connection():
    # Your connection creation logic
    return LLMConnection()

pool = ConnectionPool(create_llm_connection, config)

# Use connection
connection = pool.get_connection()
try:
    # Use the connection
    result = connection.generate("prompt")
finally:
    pool.return_connection(connection)
```

### Benefits
- Reduces connection establishment overhead
- Limits resource usage through connection limits
- Improves response times for frequent operations
- Provides connection validation and health checking

## Caching Layer

### Overview
The caching system stores frequently accessed data in memory to reduce computation and external API calls.

### Configuration
```yaml
# Performance Configuration
enable_caching: true    # Enable/disable caching
cache_size: 10000      # Maximum number of cached items
cache_ttl: 3600.0      # Default time-to-live in seconds
```

### Usage
```python
from src.rag_engine.core.performance import cached, get_cache_manager

# Using decorator
@cached(ttl=1800.0)  # Cache for 30 minutes
def expensive_operation(param):
    # Expensive computation
    return result

# Manual cache management
cache = get_cache_manager()
cache.set("key", "value", ttl=3600.0)
value = cache.get("key")
```

### Cache Types
- **Query Results** - Cache RAG query responses
- **Embeddings** - Cache document and query embeddings
- **LLM Responses** - Cache language model outputs
- **Retrieval Results** - Cache document retrieval results

### Benefits
- Reduces API calls and costs
- Improves response times for repeated queries
- Reduces computational overhead
- Configurable TTL and size limits

## Async/Await Support

### Overview
Async operations enable concurrent processing of multiple requests, improving system throughput.

### Configuration
```yaml
# Performance Configuration
async_enabled: true              # Enable async operations
max_concurrent_requests: 50      # Maximum concurrent requests
```

### Usage
```python
from src.rag_engine.core.async_engine import AsyncRAGEngine, process_queries_concurrently

# Create async RAG engine
async with AsyncRAGEngine(config) as engine:
    # Process single query
    response = await engine.query("What is machine learning?")
    
    # Process multiple queries concurrently
    queries = ["Query 1", "Query 2", "Query 3"]
    responses = await process_queries_concurrently(
        engine, queries, max_concurrent=10
    )
```

### Async Components
- **AsyncRAGEngine** - Async version of the main RAG engine
- **AsyncConnectionPool** - Async connection pooling
- **AsyncBatchProcessor** - Async batch processing
- **Performance Tracking** - Async performance monitoring

### Benefits
- Higher throughput for concurrent requests
- Better resource utilization
- Improved scalability
- Non-blocking I/O operations

## Batch Processing

### Overview
Batch processing groups multiple operations together to improve efficiency and reduce overhead.

### Configuration
```yaml
# Performance Configuration
batch_size: 20    # Number of items to process in each batch
```

### Usage
```python
from src.rag_engine.core.performance import BatchProcessor

def process_batch(items):
    # Process multiple items together
    return [f"processed_{item}" for item in items]

processor = BatchProcessor(
    batch_size=10,
    max_wait_time=1.0,
    processor_func=process_batch
)

# Submit items for batch processing
result = processor.submit("item1", "id1")
```

### Batch Operations
- **Embedding Generation** - Batch multiple texts for embedding
- **Document Indexing** - Index multiple documents together
- **Query Processing** - Process multiple queries in batches
- **Evaluation** - Batch evaluation of test cases

### Benefits
- Reduces per-item overhead
- Better utilization of external APIs
- Improved throughput for bulk operations
- Configurable batch sizes and timeouts

## Performance Monitoring

### Overview
The system includes comprehensive performance monitoring to track metrics and identify bottlenecks.

### Configuration
```yaml
# Production Configuration
enable_metrics: true    # Enable performance metrics collection
```

### Usage
```python
from src.rag_engine.core.performance import track_performance, get_performance_tracker

# Using decorator
@track_performance("my_operation")
def my_function():
    # Function implementation
    pass

# Manual tracking
tracker = get_performance_tracker()
metrics = tracker.get_metrics("my_operation")
stats = tracker.get_operation_stats("my_operation")
```

### Tracked Metrics
- **Response Times** - Operation duration and latency
- **Throughput** - Operations per second
- **Success Rates** - Success/failure ratios
- **Resource Usage** - Memory and CPU utilization
- **Cache Performance** - Hit rates and efficiency

### Benefits
- Identify performance bottlenecks
- Monitor system health
- Track performance trends
- Optimize resource allocation

## Benchmarking

### Running Benchmarks
Use the provided benchmarking script to analyze system performance:

```bash
# Run all benchmarks
python scripts/performance_benchmark.py

# Save results to file
python scripts/performance_benchmark.py --output results.json

# Use custom configuration
python scripts/performance_benchmark.py --config config/config.performance.yaml
```

### Benchmark Categories
1. **Cache Performance** - Read/write operations, hit rates
2. **Connection Pool Performance** - Acquisition times, reuse rates
3. **Concurrent Operations** - Scalability under load
4. **Async vs Sync** - Performance comparison
5. **Memory Usage** - Memory consumption patterns
6. **Scalability** - Performance with increasing load

### Performance Targets
- **Cache Hit Rate** - > 80%
- **Connection Reuse Rate** - > 70%
- **P95 Response Time** - < 500ms
- **Async Speedup** - > 2x for I/O operations
- **Memory Efficiency** - < 1KB per cached item

## Optimization Strategies

### 1. Connection Pool Tuning
- Set `max_connections` based on external service limits
- Use `min_connections` to maintain warm connections
- Monitor acquisition times and adjust pool size

### 2. Cache Optimization
- Set appropriate TTL based on data freshness requirements
- Size cache based on available memory
- Use cache warming for frequently accessed data

### 3. Async Configuration
- Enable async for I/O-heavy workloads
- Set `max_concurrent_requests` based on system capacity
- Use connection pooling with async operations

### 4. Batch Processing
- Use larger batch sizes for throughput-critical operations
- Balance batch size with latency requirements
- Configure timeouts based on processing patterns

### 5. Resource Management
- Monitor memory usage and adjust cache sizes
- Use connection limits to prevent resource exhaustion
- Implement graceful degradation for overload scenarios

## Troubleshooting

### Common Issues

#### High Latency
- Check connection pool acquisition times
- Verify cache hit rates
- Monitor external service response times
- Consider increasing connection pool size

#### Memory Issues
- Reduce cache size or TTL
- Monitor connection pool memory usage
- Check for memory leaks in long-running processes

#### Low Throughput
- Enable async operations
- Increase batch sizes
- Optimize connection pool configuration
- Use caching for repeated operations

#### Connection Errors
- Check connection pool limits
- Verify external service availability
- Monitor connection timeout settings
- Implement retry logic with backoff

### Monitoring Commands
```python
# Check performance metrics
tracker = get_performance_tracker()
print(tracker.get_operation_stats("query"))

# Check cache performance
cache = get_cache_manager()
print(cache.get_stats())

# Check connection pool status
print(pool.get_stats())
```

## Best Practices

1. **Enable Caching** - For production workloads with repeated queries
2. **Use Connection Pooling** - For all external service connections
3. **Monitor Performance** - Continuously track key metrics
4. **Tune Configuration** - Adjust settings based on workload patterns
5. **Test Under Load** - Use benchmarking to validate performance
6. **Plan for Scale** - Design for horizontal scaling from the start
7. **Implement Graceful Degradation** - Handle overload scenarios
8. **Regular Optimization** - Periodically review and optimize settings

## Configuration Examples

### High-Throughput Configuration
```yaml
# Optimized for maximum throughput
connection_pool_size: 50
cache_size: 50000
async_enabled: true
max_concurrent_requests: 100
batch_size: 50
enable_connection_pooling: true
```

### Low-Latency Configuration
```yaml
# Optimized for minimum latency
connection_pool_size: 20
cache_size: 10000
cache_ttl: 1800.0
async_enabled: true
max_concurrent_requests: 20
batch_size: 10
```

### Memory-Constrained Configuration
```yaml
# Optimized for limited memory
connection_pool_size: 5
cache_size: 1000
cache_ttl: 900.0
async_enabled: false
max_concurrent_requests: 5
batch_size: 5
```

This performance optimization system provides the foundation for building scalable, high-performance RAG applications that can handle production workloads efficiently.