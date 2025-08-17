"""
Monitoring and metrics collection for the RAG engine
"""

import time
import psutil
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque
import json
from pathlib import Path

from .logging import get_logger


@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class HealthStatus:
    """System health status"""
    status: str  # "healthy", "degraded", "unhealthy"
    checks: Dict[str, bool]
    message: str
    timestamp: datetime


class MetricsCollector:
    """Collects and stores metrics for the RAG engine"""
    
    def __init__(self, max_points: int = 10000):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.logger = get_logger("monitoring")
        self._lock = threading.Lock()
    
    def record_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Record a counter metric"""
        with self._lock:
            key = self._make_key(name, labels)
            self.counters[key] += value
            self.metrics[key].append(MetricPoint(datetime.now(timezone.utc), self.counters[key], labels or {}))
    
    def record_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a gauge metric"""
        with self._lock:
            key = self._make_key(name, labels)
            self.gauges[key] = value
            self.metrics[key].append(MetricPoint(datetime.now(timezone.utc), value, labels or {}))
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram metric"""
        with self._lock:
            key = self._make_key(name, labels)
            self.histograms[key].append(value)
            self.metrics[key].append(MetricPoint(datetime.now(timezone.utc), value, labels or {}))
    
    def record_timing(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None):
        """Record a timing metric"""
        self.record_histogram(f"{name}_duration_seconds", duration, labels)
        self.record_counter(f"{name}_total", 1.0, labels)
    
    def get_metric_summary(self, name: str, labels: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Get summary statistics for a metric"""
        key = self._make_key(name, labels)
        points = list(self.metrics[key])
        
        if not points:
            return {}
        
        values = [p.value for p in points]
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'latest': values[-1] if values else 0,
            'timestamp': points[-1].timestamp.isoformat() if points else None
        }
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all metrics summaries"""
        with self._lock:
            return {name: self.get_metric_summary(name.split('|')[0], 
                                                self._parse_labels(name.split('|')[1]) if '|' in name else None)
                   for name in self.metrics.keys()}
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Create a unique key for metric storage"""
        if labels:
            label_str = '|'.join(f"{k}={v}" for k, v in sorted(labels.items()))
            return f"{name}|{label_str}"
        return name
    
    def _parse_labels(self, label_str: str) -> Dict[str, str]:
        """Parse labels from string"""
        if not label_str:
            return {}
        return dict(item.split('=') for item in label_str.split('|'))


class ResourceMonitor:
    """Monitors system resource usage"""
    
    def __init__(self, metrics_collector: MetricsCollector, interval: float = 30.0):
        self.metrics = metrics_collector
        self.interval = interval
        self.logger = get_logger("resource_monitor")
        self._running = False
        self._thread = None
    
    def start(self):
        """Start resource monitoring"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        self.logger.info("Resource monitoring started")
    
    def stop(self):
        """Stop resource monitoring"""
        self._running = False
        if self._thread:
            self._thread.join()
        self.logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                self._collect_system_metrics()
                time.sleep(self.interval)
            except Exception as e:
                self.logger.exception(f"Error in resource monitoring: {e}")
                time.sleep(self.interval)
    
    def _collect_system_metrics(self):
        """Collect system resource metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics.record_gauge("system_cpu_percent", cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.metrics.record_gauge("system_memory_percent", memory.percent)
        self.metrics.record_gauge("system_memory_used_mb", memory.used / 1024 / 1024)
        self.metrics.record_gauge("system_memory_available_mb", memory.available / 1024 / 1024)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        self.metrics.record_gauge("system_disk_percent", (disk.used / disk.total) * 100)
        self.metrics.record_gauge("system_disk_used_gb", disk.used / 1024 / 1024 / 1024)
        self.metrics.record_gauge("system_disk_free_gb", disk.free / 1024 / 1024 / 1024)
        
        # Process-specific metrics
        process = psutil.Process()
        self.metrics.record_gauge("process_cpu_percent", process.cpu_percent())
        self.metrics.record_gauge("process_memory_mb", process.memory_info().rss / 1024 / 1024)
        self.metrics.record_gauge("process_threads", process.num_threads())


class HealthChecker:
    """Performs health checks on system components"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.logger = get_logger("health_checker")
        self.checks: Dict[str, Callable[[], bool]] = {}
    
    def register_check(self, name: str, check_func: Callable[[], bool]):
        """Register a health check function"""
        self.checks[name] = check_func
        self.logger.info(f"Registered health check: {name}")
    
    def run_checks(self) -> HealthStatus:
        """Run all health checks"""
        results = {}
        overall_healthy = True
        
        for name, check_func in self.checks.items():
            try:
                result = check_func()
                results[name] = result
                if not result:
                    overall_healthy = False
                self.metrics.record_gauge(f"health_check_{name}", 1.0 if result else 0.0)
            except Exception as e:
                self.logger.exception(f"Health check '{name}' failed: {e}")
                results[name] = False
                overall_healthy = False
                self.metrics.record_gauge(f"health_check_{name}", 0.0)
        
        # Determine overall status
        if overall_healthy:
            status = "healthy"
            message = "All systems operational"
        elif any(results.values()):
            status = "degraded"
            message = "Some systems experiencing issues"
        else:
            status = "unhealthy"
            message = "Multiple system failures detected"
        
        health_status = HealthStatus(
            status=status,
            checks=results,
            message=message,
            timestamp=datetime.now(timezone.utc)
        )
        
        self.metrics.record_gauge("system_health", 1.0 if overall_healthy else 0.0)
        return health_status


class PrometheusExporter:
    """Exports metrics in Prometheus format"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.logger = get_logger("prometheus_exporter")
    
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        # Add help and type information
        lines.append("# HELP rag_engine_info RAG Engine information")
        lines.append("# TYPE rag_engine_info gauge")
        lines.append('rag_engine_info{version="1.0.0"} 1')
        lines.append("")
        
        # Export all metrics
        all_metrics = self.metrics.get_all_metrics()
        
        for metric_name, summary in all_metrics.items():
            if '|' in metric_name:
                name, label_str = metric_name.split('|', 1)
                labels = self.metrics._parse_labels(label_str)
                label_pairs = ','.join(f'{k}="{v}"' for k, v in labels.items())
                metric_line = f'{name}{{{label_pairs}}} {summary.get("latest", 0)}'
            else:
                metric_line = f'{metric_name} {summary.get("latest", 0)}'
            
            lines.append(metric_line)
        
        return '\n'.join(lines)
    
    def save_metrics(self, file_path: str):
        """Save metrics to file"""
        metrics_text = self.export_metrics()
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(metrics_text)


class RAGMetricsCollector:
    """Specialized metrics collector for RAG-specific metrics"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.logger = get_logger("rag_metrics")
        self._accuracy_window = deque(maxlen=100)  # Keep last 100 accuracy scores
        self._response_time_window = deque(maxlen=100)  # Keep last 100 response times
    
    def record_query_metrics(self, response_time: float, confidence_score: float, 
                           retrieved_docs_count: int, query_length: int, 
                           answer_length: int, status: str = "success"):
        """Record metrics for a RAG query"""
        labels = {"status": status}
        
        # Response time metrics
        self.metrics.record_histogram("rag_query_duration_seconds", response_time, labels)
        self.metrics.record_counter("rag_queries_total", 1.0, labels)
        
        # Quality metrics
        self.metrics.record_histogram("rag_confidence_score", confidence_score, labels)
        self.metrics.record_histogram("rag_retrieved_docs_count", retrieved_docs_count, labels)
        
        # Content metrics
        self.metrics.record_histogram("rag_query_length_chars", query_length, labels)
        self.metrics.record_histogram("rag_answer_length_chars", answer_length, labels)
        
        # Update gauges for latest values
        self.metrics.record_gauge("rag_latest_response_time", response_time)
        self.metrics.record_gauge("rag_latest_confidence", confidence_score)
        
        # Track accuracy and response time trends
        if status == "success":
            self._accuracy_window.append(confidence_score)
            self._response_time_window.append(response_time)
            
            # Calculate rolling averages
            if self._accuracy_window:
                avg_accuracy = sum(self._accuracy_window) / len(self._accuracy_window)
                self.metrics.record_gauge("rag_rolling_avg_accuracy", avg_accuracy)
            
            if self._response_time_window:
                avg_response_time = sum(self._response_time_window) / len(self._response_time_window)
                self.metrics.record_gauge("rag_rolling_avg_response_time", avg_response_time)
    
    def record_indexing_metrics(self, docs_processed: int, processing_time: float, 
                              chunk_count: int, status: str = "success"):
        """Record metrics for document indexing"""
        labels = {"status": status}
        
        self.metrics.record_counter("rag_documents_indexed_total", docs_processed, labels)
        self.metrics.record_histogram("rag_indexing_duration_seconds", processing_time, labels)
        self.metrics.record_histogram("rag_chunks_created", chunk_count, labels)
    
    def record_retrieval_metrics(self, query_time: float, docs_found: int, 
                               rerank_time: Optional[float] = None, status: str = "success"):
        """Record metrics for document retrieval"""
        labels = {"status": status}
        
        self.metrics.record_histogram("rag_retrieval_duration_seconds", query_time, labels)
        self.metrics.record_histogram("rag_retrieval_docs_found", docs_found, labels)
        
        if rerank_time is not None:
            self.metrics.record_histogram("rag_rerank_duration_seconds", rerank_time, labels)
    
    def record_generation_metrics(self, generation_time: float, prompt_tokens: int, 
                                completion_tokens: int, status: str = "success"):
        """Record metrics for response generation"""
        labels = {"status": status}
        
        self.metrics.record_histogram("rag_generation_duration_seconds", generation_time, labels)
        self.metrics.record_histogram("rag_prompt_tokens", prompt_tokens, labels)
        self.metrics.record_histogram("rag_completion_tokens", completion_tokens, labels)
        
        total_tokens = prompt_tokens + completion_tokens
        self.metrics.record_histogram("rag_total_tokens", total_tokens, labels)
    
    def record_evaluation_metrics(self, framework: str, metric_name: str, 
                                score: float, evaluation_time: float):
        """Record evaluation metrics"""
        labels = {"framework": framework, "metric": metric_name}
        
        self.metrics.record_histogram(f"rag_evaluation_score", score, labels)
        self.metrics.record_histogram(f"rag_evaluation_duration_seconds", evaluation_time, labels)
    
    def get_rag_performance_summary(self) -> Dict[str, Any]:
        """Get summary of RAG performance metrics"""
        summary = {}
        
        # Response time statistics
        response_time_summary = self.metrics.get_metric_summary("rag_query_duration_seconds")
        if response_time_summary:
            summary["response_time"] = {
                "avg_seconds": response_time_summary.get("avg", 0),
                "min_seconds": response_time_summary.get("min", 0),
                "max_seconds": response_time_summary.get("max", 0),
                "latest_seconds": response_time_summary.get("latest", 0)
            }
        
        # Accuracy statistics
        confidence_summary = self.metrics.get_metric_summary("rag_confidence_score")
        if confidence_summary:
            summary["accuracy"] = {
                "avg_confidence": confidence_summary.get("avg", 0),
                "min_confidence": confidence_summary.get("min", 0),
                "max_confidence": confidence_summary.get("max", 0),
                "latest_confidence": confidence_summary.get("latest", 0)
            }
        
        # Query statistics
        queries_summary = self.metrics.get_metric_summary("rag_queries_total")
        if queries_summary:
            summary["queries"] = {
                "total_queries": queries_summary.get("latest", 0),
                "query_count": queries_summary.get("count", 0)
            }
        
        # Retrieval statistics
        retrieval_summary = self.metrics.get_metric_summary("rag_retrieved_docs_count")
        if retrieval_summary:
            summary["retrieval"] = {
                "avg_docs_retrieved": retrieval_summary.get("avg", 0),
                "min_docs_retrieved": retrieval_summary.get("min", 0),
                "max_docs_retrieved": retrieval_summary.get("max", 0)
            }
        
        return summary


class MonitoringManager:
    """Central manager for all monitoring functionality"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metrics = MetricsCollector(max_points=self.config.get('max_metric_points', 10000))
        self.resource_monitor = ResourceMonitor(
            self.metrics, 
            interval=self.config.get('resource_monitor_interval', 30.0)
        )
        self.health_checker = HealthChecker(self.metrics)
        self.prometheus_exporter = PrometheusExporter(self.metrics)
        self.rag_metrics = RAGMetricsCollector(self.metrics)
        self.logger = get_logger("monitoring_manager")
        
        # Register default health checks
        self._register_default_health_checks()
    
    def start(self):
        """Start all monitoring components"""
        self.resource_monitor.start()
        self.logger.info("Monitoring system started")
    
    def stop(self):
        """Stop all monitoring components"""
        self.resource_monitor.stop()
        self.logger.info("Monitoring system stopped")
    
    def get_health_status(self) -> HealthStatus:
        """Get current system health status"""
        return self.health_checker.run_checks()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        return self.metrics.get_all_metrics()
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        return self.prometheus_exporter.export_metrics()
    
    def get_rag_performance_summary(self) -> Dict[str, Any]:
        """Get RAG-specific performance summary"""
        return self.rag_metrics.get_rag_performance_summary()
    
    def record_rag_query(self, response_time: float, confidence_score: float, 
                        retrieved_docs_count: int, query_length: int, 
                        answer_length: int, status: str = "success"):
        """Record RAG query metrics"""
        self.rag_metrics.record_query_metrics(
            response_time, confidence_score, retrieved_docs_count, 
            query_length, answer_length, status
        )
    
    def record_rag_indexing(self, docs_processed: int, processing_time: float, 
                           chunk_count: int, status: str = "success"):
        """Record RAG indexing metrics"""
        self.rag_metrics.record_indexing_metrics(
            docs_processed, processing_time, chunk_count, status
        )
    
    def record_rag_retrieval(self, query_time: float, docs_found: int, 
                            rerank_time: Optional[float] = None, status: str = "success"):
        """Record RAG retrieval metrics"""
        self.rag_metrics.record_retrieval_metrics(
            query_time, docs_found, rerank_time, status
        )
    
    def record_rag_generation(self, generation_time: float, prompt_tokens: int, 
                             completion_tokens: int, status: str = "success"):
        """Record RAG generation metrics"""
        self.rag_metrics.record_generation_metrics(
            generation_time, prompt_tokens, completion_tokens, status
        )
    
    def record_rag_evaluation(self, framework: str, metric_name: str, 
                             score: float, evaluation_time: float):
        """Record RAG evaluation metrics"""
        self.rag_metrics.record_evaluation_metrics(
            framework, metric_name, score, evaluation_time
        )
    
    def _register_default_health_checks(self):
        """Register default health checks"""
        
        def check_memory_usage():
            """Check if memory usage is below threshold"""
            memory = psutil.virtual_memory()
            return memory.percent < self.config.get('memory_threshold', 90.0)
        
        def check_disk_space():
            """Check if disk space is available"""
            disk = psutil.disk_usage('/')
            usage_percent = (disk.used / disk.total) * 100
            return usage_percent < self.config.get('disk_threshold', 90.0)
        
        def check_cpu_usage():
            """Check if CPU usage is reasonable"""
            cpu_percent = psutil.cpu_percent(interval=1)
            return cpu_percent < self.config.get('cpu_threshold', 90.0)
        
        self.health_checker.register_check("memory_usage", check_memory_usage)
        self.health_checker.register_check("disk_space", check_disk_space)
        self.health_checker.register_check("cpu_usage", check_cpu_usage)


# Global monitoring instance
_monitoring_manager = None


def get_monitoring_manager(config: Optional[Dict[str, Any]] = None) -> MonitoringManager:
    """Get the global monitoring manager instance"""
    global _monitoring_manager
    if _monitoring_manager is None:
        _monitoring_manager = MonitoringManager(config)
    return _monitoring_manager


def record_metric(name: str, value: float, metric_type: str = "gauge", labels: Optional[Dict[str, str]] = None):
    """Convenience function to record a metric"""
    manager = get_monitoring_manager()
    
    if metric_type == "counter":
        manager.metrics.record_counter(name, value, labels)
    elif metric_type == "gauge":
        manager.metrics.record_gauge(name, value, labels)
    elif metric_type == "histogram":
        manager.metrics.record_histogram(name, value, labels)
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")


def time_operation(operation_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to time operations and record metrics"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                record_metric(f"{operation_name}_duration_seconds", duration, "histogram", labels)
                record_metric(f"{operation_name}_total", 1.0, "counter", {**(labels or {}), "status": "success"})
                return result
            except Exception as e:
                duration = time.time() - start_time
                record_metric(f"{operation_name}_duration_seconds", duration, "histogram", labels)
                record_metric(f"{operation_name}_total", 1.0, "counter", {**(labels or {}), "status": "error"})
                raise
        return wrapper
    return decorator


def record_rag_query_metrics(response_time: float, confidence_score: float, 
                            retrieved_docs_count: int, query_length: int, 
                            answer_length: int, status: str = "success"):
    """Convenience function to record RAG query metrics"""
    manager = get_monitoring_manager()
    manager.record_rag_query(
        response_time, confidence_score, retrieved_docs_count, 
        query_length, answer_length, status
    )


def record_rag_indexing_metrics(docs_processed: int, processing_time: float, 
                               chunk_count: int, status: str = "success"):
    """Convenience function to record RAG indexing metrics"""
    manager = get_monitoring_manager()
    manager.record_rag_indexing(docs_processed, processing_time, chunk_count, status)


def record_rag_retrieval_metrics(query_time: float, docs_found: int, 
                                rerank_time: Optional[float] = None, status: str = "success"):
    """Convenience function to record RAG retrieval metrics"""
    manager = get_monitoring_manager()
    manager.record_rag_retrieval(query_time, docs_found, rerank_time, status)


def record_rag_generation_metrics(generation_time: float, prompt_tokens: int, 
                                 completion_tokens: int, status: str = "success"):
    """Convenience function to record RAG generation metrics"""
    manager = get_monitoring_manager()
    manager.record_rag_generation(generation_time, prompt_tokens, completion_tokens, status)


def record_rag_evaluation_metrics(framework: str, metric_name: str, 
                                 score: float, evaluation_time: float):
    """Convenience function to record RAG evaluation metrics"""
    manager = get_monitoring_manager()
    manager.record_rag_evaluation(framework, metric_name, score, evaluation_time)


def get_rag_performance_summary() -> Dict[str, Any]:
    """Convenience function to get RAG performance summary"""
    manager = get_monitoring_manager()
    return manager.get_rag_performance_summary()