# Monitoring and Health Checks

This document describes the comprehensive monitoring and health check system implemented for the production RAG system.

## Overview

The monitoring system provides:

- **Metrics Collection**: Response times, accuracy scores, resource usage
- **Health Checks**: System component status monitoring
- **HTTP Endpoints**: RESTful API for health and metrics
- **Prometheus Integration**: Industry-standard metrics export
- **Resource Monitoring**: CPU, memory, and disk usage tracking

## Architecture

### Core Components

1. **MetricsCollector**: Collects and stores time-series metrics
2. **ResourceMonitor**: Monitors system resource usage
3. **HealthChecker**: Performs health checks on system components
4. **PrometheusExporter**: Exports metrics in Prometheus format
5. **RAGMetricsCollector**: Specialized metrics for RAG operations
6. **HealthCheckServer**: HTTP server for health and metrics endpoints

### Key Classes

```python
from src.rag_engine.core.monitoring import (
    MonitoringManager,
    get_monitoring_manager,
    record_rag_query_metrics,
    record_rag_indexing_metrics
)
from src.rag_engine.core.health_api import (
    HealthCheckServer,
    HealthCheckClient,
    start_health_server,
    stop_health_server
)
```

## Usage

### Basic Setup

```python
from src.rag_engine.core.monitoring import get_monitoring_manager
from src.rag_engine.core.health_api import start_health_server

# Initialize monitoring with configuration
config = {
    'max_metric_points': 10000,
    'resource_monitor_interval': 30.0,
    'memory_threshold': 85.0,
    'cpu_threshold': 80.0,
    'disk_threshold': 90.0
}

# Get monitoring manager
manager = get_monitoring_manager(config)

# Start monitoring
manager.start()

# Start health check server
server = start_health_server(host="0.0.0.0", port=8080, monitoring_manager=manager)
```

### Recording RAG Metrics

```python
from src.rag_engine.core.monitoring import record_rag_query_metrics, record_rag_indexing_metrics

# Record query metrics
record_rag_query_metrics(
    response_time=1.2,
    confidence_score=0.85,
    retrieved_docs_count=5,
    query_length=45,
    answer_length=180,
    status="success"
)

# Record indexing metrics
record_rag_indexing_metrics(
    docs_processed=25,
    processing_time=8.5,
    chunk_count=150,
    status="success"
)
```

### Custom Health Checks

```python
def check_database_connection():
    """Custom health check for database"""
    try:
        # Your database connection logic here
        return True
    except Exception:
        return False

# Register custom health check
manager.health_checker.register_check("database", check_database_connection)
```

## HTTP Endpoints

The health check server provides the following endpoints:

### Health Endpoints

#### `GET /health`
Basic health check returning overall system status.

**Response:**
```json
{
  "status": "healthy",
  "message": "All systems operational",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### `GET /health/detailed`
Detailed health check with individual component status.

**Response:**
```json
{
  "status": "healthy",
  "message": "All systems operational",
  "timestamp": "2024-01-01T12:00:00Z",
  "checks": {
    "memory_usage": true,
    "disk_space": true,
    "cpu_usage": true,
    "database": true
  },
  "system_info": {
    "platform": "Linux-5.4.0-x86_64",
    "python_version": "3.11.0",
    "cpu_count": 8,
    "memory_total_gb": 32.0,
    "disk_total_gb": 500.0
  }
}
```

### Metrics Endpoints

#### `GET /metrics`
All metrics in JSON format.

**Response:**
```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "metrics": {
    "rag_query_duration_seconds": {
      "count": 100,
      "min": 0.5,
      "max": 3.2,
      "avg": 1.2,
      "latest": 1.1
    },
    "system_cpu_percent": {
      "latest": 25.5
    }
  }
}
```

#### `GET /metrics/prometheus`
Metrics in Prometheus format for scraping.

**Response:**
```
# HELP rag_engine_info RAG Engine information
# TYPE rag_engine_info gauge
rag_engine_info{version="1.0.0"} 1

# RAG Query Metrics
rag_query_duration_seconds 1.2
rag_queries_total{status="success"} 95
rag_queries_total{status="error"} 5
rag_confidence_score 0.85

# System Metrics
system_cpu_percent 25.5
system_memory_percent 45.2
system_memory_used_mb 14500.0
```

#### `GET /metrics/rag`
RAG-specific performance metrics.

**Response:**
```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "rag_performance": {
    "response_time": {
      "avg_seconds": 1.2,
      "min_seconds": 0.5,
      "max_seconds": 3.2,
      "latest_seconds": 1.1
    },
    "accuracy": {
      "avg_confidence": 0.85,
      "min_confidence": 0.65,
      "max_confidence": 0.98,
      "latest_confidence": 0.87
    },
    "queries": {
      "total_queries": 100,
      "query_count": 100
    },
    "retrieval": {
      "avg_docs_retrieved": 4.2,
      "min_docs_retrieved": 1,
      "max_docs_retrieved": 10
    }
  }
}
```

#### `GET /status`
Complete system status combining health and performance data.

## Metrics Types

### RAG-Specific Metrics

| Metric Name | Type | Description |
|-------------|------|-------------|
| `rag_query_duration_seconds` | Histogram | Time taken to process queries |
| `rag_queries_total` | Counter | Total number of queries processed |
| `rag_confidence_score` | Histogram | Confidence scores of responses |
| `rag_retrieved_docs_count` | Histogram | Number of documents retrieved |
| `rag_query_length_chars` | Histogram | Length of input queries |
| `rag_answer_length_chars` | Histogram | Length of generated answers |
| `rag_rolling_avg_accuracy` | Gauge | Rolling average accuracy |
| `rag_rolling_avg_response_time` | Gauge | Rolling average response time |

### System Metrics

| Metric Name | Type | Description |
|-------------|------|-------------|
| `system_cpu_percent` | Gauge | CPU usage percentage |
| `system_memory_percent` | Gauge | Memory usage percentage |
| `system_memory_used_mb` | Gauge | Memory used in MB |
| `system_memory_available_mb` | Gauge | Available memory in MB |
| `system_disk_percent` | Gauge | Disk usage percentage |
| `system_disk_used_gb` | Gauge | Disk used in GB |
| `system_disk_free_gb` | Gauge | Free disk space in GB |
| `process_cpu_percent` | Gauge | Process CPU usage |
| `process_memory_mb` | Gauge | Process memory usage |
| `process_threads` | Gauge | Number of process threads |

### Health Check Metrics

| Metric Name | Type | Description |
|-------------|------|-------------|
| `health_check_{name}` | Gauge | Individual health check status (1=pass, 0=fail) |
| `system_health` | Gauge | Overall system health (1=healthy, 0=unhealthy) |

## Configuration

### Monitoring Manager Configuration

```python
config = {
    # Metrics storage
    'max_metric_points': 10000,  # Maximum points per metric
    
    # Resource monitoring
    'resource_monitor_interval': 30.0,  # Seconds between collections
    
    # Health check thresholds
    'memory_threshold': 85.0,    # Memory usage threshold (%)
    'cpu_threshold': 80.0,       # CPU usage threshold (%)
    'disk_threshold': 90.0,      # Disk usage threshold (%)
}
```

### Health Server Configuration

```python
# Start health server
server = start_health_server(
    host="0.0.0.0",      # Bind address
    port=8080,           # Port number
    monitoring_manager=manager  # Optional custom manager
)
```

## Integration with Monitoring Systems

### Prometheus Integration

1. **Configure Prometheus** to scrape the `/metrics/prometheus` endpoint:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'rag-engine'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics/prometheus'
    scrape_interval: 30s
```

2. **Set up Grafana dashboards** using the exported metrics.

3. **Configure alerts** based on health check status and performance metrics.

### Health Check Integration

1. **Load Balancer Health Checks**: Use `/health` endpoint
2. **Kubernetes Liveness Probe**: Use `/health` endpoint
3. **Kubernetes Readiness Probe**: Use `/health/detailed` endpoint
4. **Monitoring Dashboards**: Use `/status` endpoint

## Best Practices

### Metrics Collection

1. **Use appropriate metric types**:
   - Counters for cumulative values (total queries)
   - Gauges for current values (CPU usage)
   - Histograms for distributions (response times)

2. **Add meaningful labels**:
   ```python
   record_rag_query_metrics(
       response_time=1.2,
       confidence_score=0.85,
       retrieved_docs_count=5,
       query_length=45,
       answer_length=180,
       status="success"  # Important for filtering
   )
   ```

3. **Monitor error rates**:
   ```python
   # Record failed operations
   record_rag_query_metrics(
       response_time=5.0,
       confidence_score=0.3,
       retrieved_docs_count=0,
       query_length=40,
       answer_length=50,
       status="error"  # Track failures
   )
   ```

### Health Checks

1. **Keep checks lightweight**: Health checks should complete quickly
2. **Check critical dependencies**: Database, external APIs, disk space
3. **Use meaningful names**: Clear, descriptive check names
4. **Handle exceptions**: Health checks should never crash

### Performance

1. **Configure appropriate intervals**: Balance monitoring detail with performance
2. **Limit metric retention**: Use `max_metric_points` to control memory usage
3. **Use resource monitoring**: Monitor the monitoring system itself

## Troubleshooting

### Common Issues

1. **High Memory Usage**:
   - Reduce `max_metric_points`
   - Increase `resource_monitor_interval`
   - Check for metric label cardinality

2. **Health Check Failures**:
   - Check individual health check logs
   - Verify system resource thresholds
   - Review custom health check implementations

3. **Missing Metrics**:
   - Verify monitoring manager is started
   - Check metric recording calls
   - Review Prometheus export format

### Debugging

1. **Enable debug logging**:
   ```python
   import logging
   logging.getLogger("rag_engine.monitoring").setLevel(logging.DEBUG)
   ```

2. **Check health status programmatically**:
   ```python
   health_status = manager.get_health_status()
   print(f"Status: {health_status.status}")
   print(f"Checks: {health_status.checks}")
   ```

3. **Inspect metrics directly**:
   ```python
   metrics_summary = manager.get_metrics_summary()
   for name, summary in metrics_summary.items():
       print(f"{name}: {summary}")
   ```

## Examples

See `examples/example_monitoring_and_health_checks.py` for a complete demonstration of the monitoring system functionality.

## Testing

The monitoring system includes comprehensive tests:

- Unit tests: `tests/core/test_monitoring.py`, `tests/core/test_health_api.py`
- Integration tests: `tests/integration/test_monitoring_integration.py`

Run tests with:
```bash
uv run python -m pytest tests/core/test_monitoring.py -v
uv run python -m pytest tests/core/test_health_api.py -v
uv run python -m pytest tests/integration/test_monitoring_integration.py -v
```