"""
Tests for the monitoring system
"""

import pytest
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.rag_engine.core.monitoring import (
    MetricPoint,
    HealthStatus,
    MetricsCollector,
    ResourceMonitor,
    HealthChecker,
    PrometheusExporter,
    MonitoringManager,
    get_monitoring_manager,
    record_metric,
    time_operation
)


class TestMetricPoint:
    """Test MetricPoint dataclass"""
    
    def test_metric_point_creation(self):
        """Test metric point creation"""
        timestamp = datetime.utcnow()
        labels = {"service": "test", "version": "1.0"}
        
        point = MetricPoint(timestamp=timestamp, value=42.5, labels=labels)
        
        assert point.timestamp == timestamp
        assert point.value == 42.5
        assert point.labels == labels
    
    def test_metric_point_default_labels(self):
        """Test metric point with default labels"""
        timestamp = datetime.utcnow()
        point = MetricPoint(timestamp=timestamp, value=10.0)
        
        assert point.labels == {}


class TestHealthStatus:
    """Test HealthStatus dataclass"""
    
    def test_health_status_creation(self):
        """Test health status creation"""
        timestamp = datetime.utcnow()
        checks = {"database": True, "cache": False}
        
        status = HealthStatus(
            status="degraded",
            checks=checks,
            message="Some services down",
            timestamp=timestamp
        )
        
        assert status.status == "degraded"
        assert status.checks == checks
        assert status.message == "Some services down"
        assert status.timestamp == timestamp


class TestMetricsCollector:
    """Test metrics collector functionality"""
    
    def test_counter_recording(self):
        """Test counter metric recording"""
        collector = MetricsCollector(max_points=100)
        
        collector.record_counter("test_counter", 1.0)
        collector.record_counter("test_counter", 2.0)
        
        summary = collector.get_metric_summary("test_counter")
        assert summary["count"] == 2
        assert summary["latest"] == 3.0  # Cumulative
    
    def test_gauge_recording(self):
        """Test gauge metric recording"""
        collector = MetricsCollector(max_points=100)
        
        collector.record_gauge("test_gauge", 10.0)
        collector.record_gauge("test_gauge", 20.0)
        
        summary = collector.get_metric_summary("test_gauge")
        assert summary["count"] == 2
        assert summary["latest"] == 20.0  # Latest value
    
    def test_histogram_recording(self):
        """Test histogram metric recording"""
        collector = MetricsCollector(max_points=100)
        
        collector.record_histogram("test_histogram", 1.0)
        collector.record_histogram("test_histogram", 2.0)
        collector.record_histogram("test_histogram", 3.0)
        
        summary = collector.get_metric_summary("test_histogram")
        assert summary["count"] == 3
        assert summary["min"] == 1.0
        assert summary["max"] == 3.0
        assert summary["avg"] == 2.0
    
    def test_timing_recording(self):
        """Test timing metric recording"""
        collector = MetricsCollector(max_points=100)
        
        collector.record_timing("test_operation", 1.5)
        
        duration_summary = collector.get_metric_summary("test_operation_duration_seconds")
        total_summary = collector.get_metric_summary("test_operation_total")
        
        assert duration_summary["latest"] == 1.5
        assert total_summary["latest"] == 1.0
    
    def test_labels_support(self):
        """Test metrics with labels"""
        collector = MetricsCollector(max_points=100)
        
        labels1 = {"service": "api", "version": "1.0"}
        labels2 = {"service": "worker", "version": "1.0"}
        
        collector.record_counter("requests", 1.0, labels1)
        collector.record_counter("requests", 2.0, labels2)
        
        summary1 = collector.get_metric_summary("requests", labels1)
        summary2 = collector.get_metric_summary("requests", labels2)
        
        assert summary1["latest"] == 1.0
        assert summary2["latest"] == 2.0
    
    def test_max_points_limit(self):
        """Test max points limit enforcement"""
        collector = MetricsCollector(max_points=3)
        
        for i in range(5):
            collector.record_gauge("test_metric", float(i))
        
        summary = collector.get_metric_summary("test_metric")
        assert summary["count"] == 3  # Limited by max_points
    
    def test_get_all_metrics(self):
        """Test getting all metrics"""
        collector = MetricsCollector(max_points=100)
        
        collector.record_counter("counter1", 1.0)
        collector.record_gauge("gauge1", 10.0)
        
        all_metrics = collector.get_all_metrics()
        
        assert "counter1" in all_metrics
        assert "gauge1" in all_metrics
        assert len(all_metrics) == 2


class TestResourceMonitor:
    """Test resource monitor functionality"""
    
    @patch('src.rag_engine.core.monitoring.psutil')
    def test_system_metrics_collection(self, mock_psutil):
        """Test system metrics collection"""
        # Mock psutil responses
        mock_psutil.cpu_percent.return_value = 50.0
        mock_psutil.virtual_memory.return_value = MagicMock(
            percent=60.0,
            used=1024 * 1024 * 1024,  # 1GB
            available=512 * 1024 * 1024  # 512MB
        )
        mock_psutil.disk_usage.return_value = MagicMock(
            total=100 * 1024 * 1024 * 1024,  # 100GB
            used=50 * 1024 * 1024 * 1024,    # 50GB
            free=50 * 1024 * 1024 * 1024     # 50GB
        )
        mock_process = MagicMock()
        mock_process.cpu_percent.return_value = 25.0
        mock_process.memory_info.return_value = MagicMock(rss=256 * 1024 * 1024)  # 256MB
        mock_process.num_threads.return_value = 10
        mock_psutil.Process.return_value = mock_process
        
        collector = MetricsCollector()
        monitor = ResourceMonitor(collector, interval=0.1)
        
        # Collect metrics once
        monitor._collect_system_metrics()
        
        # Verify metrics were recorded
        cpu_summary = collector.get_metric_summary("system_cpu_percent")
        memory_summary = collector.get_metric_summary("system_memory_percent")
        
        assert cpu_summary["latest"] == 50.0
        assert memory_summary["latest"] == 60.0
    
    def test_monitor_start_stop(self):
        """Test monitor start and stop"""
        collector = MetricsCollector()
        monitor = ResourceMonitor(collector, interval=0.1)
        
        assert not monitor._running
        
        monitor.start()
        assert monitor._running
        assert monitor._thread is not None
        
        time.sleep(0.2)  # Let it run briefly
        
        monitor.stop()
        assert not monitor._running


class TestHealthChecker:
    """Test health checker functionality"""
    
    def test_register_check(self):
        """Test registering health checks"""
        collector = MetricsCollector()
        checker = HealthChecker(collector)
        
        def test_check():
            return True
        
        checker.register_check("test_service", test_check)
        
        assert "test_service" in checker.checks
    
    def test_run_checks_all_healthy(self):
        """Test running checks when all are healthy"""
        collector = MetricsCollector()
        checker = HealthChecker(collector)
        
        checker.register_check("service1", lambda: True)
        checker.register_check("service2", lambda: True)
        
        status = checker.run_checks()
        
        assert status.status == "healthy"
        assert status.checks["service1"] is True
        assert status.checks["service2"] is True
        assert "operational" in status.message.lower()
    
    def test_run_checks_some_unhealthy(self):
        """Test running checks when some are unhealthy"""
        collector = MetricsCollector()
        checker = HealthChecker(collector)
        
        checker.register_check("service1", lambda: True)
        checker.register_check("service2", lambda: False)
        
        status = checker.run_checks()
        
        assert status.status == "degraded"
        assert status.checks["service1"] is True
        assert status.checks["service2"] is False
    
    def test_run_checks_all_unhealthy(self):
        """Test running checks when all are unhealthy"""
        collector = MetricsCollector()
        checker = HealthChecker(collector)
        
        checker.register_check("service1", lambda: False)
        checker.register_check("service2", lambda: False)
        
        status = checker.run_checks()
        
        assert status.status == "unhealthy"
        assert status.checks["service1"] is False
        assert status.checks["service2"] is False
    
    def test_check_exception_handling(self):
        """Test handling of exceptions in health checks"""
        collector = MetricsCollector()
        checker = HealthChecker(collector)
        
        def failing_check():
            raise Exception("Check failed")
        
        checker.register_check("failing_service", failing_check)
        
        status = checker.run_checks()
        
        assert status.status == "unhealthy"
        assert status.checks["failing_service"] is False


class TestPrometheusExporter:
    """Test Prometheus metrics exporter"""
    
    def test_export_metrics(self):
        """Test exporting metrics in Prometheus format"""
        collector = MetricsCollector()
        exporter = PrometheusExporter(collector)
        
        # Add some metrics
        collector.record_counter("http_requests_total", 100.0, {"method": "GET"})
        collector.record_gauge("memory_usage_bytes", 1024.0)
        
        metrics_text = exporter.export_metrics()
        
        assert "rag_engine_info" in metrics_text
        assert "http_requests_total" in metrics_text
        assert "memory_usage_bytes" in metrics_text
        assert 'method="GET"' in metrics_text
    
    def test_save_metrics(self):
        """Test saving metrics to file"""
        import tempfile
        import os
        
        collector = MetricsCollector()
        exporter = PrometheusExporter(collector)
        
        collector.record_gauge("test_metric", 42.0)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "metrics.txt")
            exporter.save_metrics(file_path)
            
            assert os.path.exists(file_path)
            
            with open(file_path, 'r') as f:
                content = f.read()
                assert "test_metric" in content


class TestMonitoringManager:
    """Test monitoring manager functionality"""
    
    def test_manager_initialization(self):
        """Test monitoring manager initialization"""
        config = {"max_metric_points": 500, "resource_monitor_interval": 60.0}
        manager = MonitoringManager(config)
        
        assert manager.config == config
        # Check that the deque has the right maxlen for new entries
        manager.metrics.record_gauge("test", 1.0)
        test_deque = manager.metrics.metrics["test"]
        assert test_deque.maxlen == 500
        assert manager.resource_monitor.interval == 60.0
    
    def test_start_stop(self):
        """Test starting and stopping monitoring"""
        manager = MonitoringManager()
        
        manager.start()
        assert manager.resource_monitor._running
        
        manager.stop()
        assert not manager.resource_monitor._running
    
    def test_get_health_status(self):
        """Test getting health status"""
        manager = MonitoringManager()
        
        status = manager.get_health_status()
        
        assert isinstance(status, HealthStatus)
        assert status.status in ["healthy", "degraded", "unhealthy"]
    
    def test_get_metrics_summary(self):
        """Test getting metrics summary"""
        manager = MonitoringManager()
        
        # Add a metric
        manager.metrics.record_gauge("test_metric", 10.0)
        
        summary = manager.get_metrics_summary()
        
        assert isinstance(summary, dict)
        assert "test_metric" in summary
    
    def test_export_prometheus_metrics(self):
        """Test exporting Prometheus metrics"""
        manager = MonitoringManager()
        
        metrics_text = manager.export_prometheus_metrics()
        
        assert isinstance(metrics_text, str)
        assert "rag_engine_info" in metrics_text


class TestGlobalFunctions:
    """Test global monitoring functions"""
    
    def test_get_monitoring_manager_singleton(self):
        """Test that get_monitoring_manager returns singleton"""
        manager1 = get_monitoring_manager()
        manager2 = get_monitoring_manager()
        
        assert manager1 is manager2
    
    @patch('src.rag_engine.core.monitoring.get_monitoring_manager')
    def test_record_metric_counter(self, mock_get_manager):
        """Test record_metric function with counter"""
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager
        
        record_metric("test_counter", 5.0, "counter", {"label": "value"})
        
        mock_manager.metrics.record_counter.assert_called_once_with(
            "test_counter", 5.0, {"label": "value"}
        )
    
    @patch('src.rag_engine.core.monitoring.get_monitoring_manager')
    def test_record_metric_gauge(self, mock_get_manager):
        """Test record_metric function with gauge"""
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager
        
        record_metric("test_gauge", 10.0, "gauge")
        
        mock_manager.metrics.record_gauge.assert_called_once_with(
            "test_gauge", 10.0, None
        )
    
    def test_record_metric_invalid_type(self):
        """Test record_metric with invalid type"""
        with pytest.raises(ValueError, match="Unknown metric type"):
            record_metric("test_metric", 1.0, "invalid_type")
    
    @patch('src.rag_engine.core.monitoring.record_metric')
    def test_time_operation_decorator_success(self, mock_record_metric):
        """Test time_operation decorator with successful operation"""
        @time_operation("test_op", {"service": "test"})
        def test_function():
            time.sleep(0.1)
            return "success"
        
        result = test_function()
        
        assert result == "success"
        assert mock_record_metric.call_count == 2  # duration and total
        
        # Check calls
        calls = mock_record_metric.call_args_list
        duration_call = calls[0]
        total_call = calls[1]
        
        assert duration_call[0][0] == "test_op_duration_seconds"
        assert duration_call[0][2] == "histogram"
        
        assert total_call[0][0] == "test_op_total"
        assert total_call[0][1] == 1.0
        assert total_call[0][2] == "counter"
        # Check that success status was recorded in the labels
        # labels is the 4th argument (index 3) in record_metric call
        labels_arg = total_call[0][3] if len(total_call[0]) > 3 else {}
        assert labels_arg.get("status") == "success"
    
    @patch('src.rag_engine.core.monitoring.record_metric')
    def test_time_operation_decorator_error(self, mock_record_metric):
        """Test time_operation decorator with error"""
        @time_operation("test_op")
        def test_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            test_function()
        
        assert mock_record_metric.call_count == 2  # duration and total
        
        # Check error status was recorded
        calls = mock_record_metric.call_args_list
        total_call = calls[1]
        # Check that error status was recorded in the labels
        # labels is the 4th argument (index 3) in record_metric call
        labels_arg = total_call[0][3] if len(total_call[0]) > 3 else {}
        assert labels_arg.get("status") == "error"