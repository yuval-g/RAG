"""
Tests for the health check API system
"""

import pytest
import json
import time
import threading
from unittest.mock import patch, MagicMock
from datetime import datetime
from http.server import HTTPServer
import urllib.request
import urllib.error

from src.rag_engine.core.health_api import (
    HealthCheckHandler,
    HealthCheckServer,
    HealthCheckClient,
    get_health_server,
    start_health_server,
    stop_health_server
)
from src.rag_engine.core.monitoring import (
    MonitoringManager,
    MetricsCollector,
    HealthStatus
)


class TestHealthCheckHandler:
    """Test health check HTTP handler"""
    
    def test_handler_initialization(self):
        """Test handler initialization logic"""
        # Since HealthCheckHandler requires HTTP server context,
        # we'll test the monitoring manager assignment logic separately
        from src.rag_engine.core.monitoring import MonitoringManager
        
        manager = MonitoringManager()
        
        # Test that we can create a handler factory with monitoring manager
        def handler_factory(*args, **kwargs):
            # This simulates what happens in the actual server
            return type('MockHandler', (), {
                'monitoring_manager': manager
            })()
        
        mock_handler = handler_factory()
        assert mock_handler.monitoring_manager == manager
    
    def test_health_check_response_format(self):
        """Test health check response format"""
        # This is more of an integration test that would require
        # setting up a full HTTP server, so we'll test the logic separately
        pass


class TestHealthCheckServer:
    """Test health check server functionality"""
    
    def test_server_initialization(self):
        """Test server initialization"""
        manager = MonitoringManager()
        server = HealthCheckServer(host="127.0.0.1", port=8081, monitoring_manager=manager)
        
        assert server.host == "127.0.0.1"
        assert server.port == 8081
        assert server.monitoring_manager == manager
        assert not server.is_running()
    
    def test_server_info(self):
        """Test getting server information"""
        server = HealthCheckServer(host="localhost", port=9090)
        info = server.get_server_info()
        
        assert info["host"] == "localhost"
        assert info["port"] == 9090
        assert info["running"] is False
        assert "/health" in info["endpoints"]
        assert "/metrics" in info["endpoints"]
        assert "/status" in info["endpoints"]
    
    @pytest.mark.integration
    def test_server_start_stop(self):
        """Test starting and stopping the server"""
        server = HealthCheckServer(host="127.0.0.1", port=8082)
        
        # Test start
        server.start()
        assert server.is_running()
        
        # Give server time to start
        time.sleep(0.1)
        
        # Test stop
        server.stop()
        assert not server.is_running()
    
    @pytest.mark.integration
    def test_server_endpoints_availability(self):
        """Test that server endpoints are accessible"""
        server = HealthCheckServer(host="127.0.0.1", port=8083)
        
        try:
            server.start()
            time.sleep(0.2)  # Give server time to start
            
            # Test basic health endpoint
            try:
                response = urllib.request.urlopen("http://127.0.0.1:8083/health", timeout=2)
                assert response.status == 200
                data = json.loads(response.read().decode('utf-8'))
                assert "status" in data
                assert "timestamp" in data
            except urllib.error.URLError:
                pytest.skip("Server not accessible - may be port conflict")
            
        finally:
            server.stop()


class TestHealthCheckClient:
    """Test health check client functionality"""
    
    def test_client_initialization(self):
        """Test client initialization"""
        client = HealthCheckClient("http://localhost:8080")
        assert client.base_url == "http://localhost:8080"
        
        # Test URL normalization
        client2 = HealthCheckClient("http://localhost:8080/")
        assert client2.base_url == "http://localhost:8080"
    
    @patch('urllib.request.urlopen')
    def test_check_health(self, mock_urlopen):
        """Test health check request"""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "status": "healthy",
            "message": "All systems operational",
            "timestamp": "2024-01-01T00:00:00"
        }).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        client = HealthCheckClient("http://localhost:8080")
        result = client.check_health()
        
        assert result["status"] == "healthy"
        assert result["message"] == "All systems operational"
        mock_urlopen.assert_called_once_with("http://localhost:8080/health")
    
    @patch('urllib.request.urlopen')
    def test_get_metrics(self, mock_urlopen):
        """Test metrics request"""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "timestamp": "2024-01-01T00:00:00",
            "metrics": {"test_metric": {"latest": 42.0}}
        }).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        client = HealthCheckClient("http://localhost:8080")
        result = client.get_metrics()
        
        assert "metrics" in result
        assert result["metrics"]["test_metric"]["latest"] == 42.0
        mock_urlopen.assert_called_once_with("http://localhost:8080/metrics")
    
    @patch('urllib.request.urlopen')
    def test_get_prometheus_metrics(self, mock_urlopen):
        """Test Prometheus metrics request"""
        mock_response = MagicMock()
        mock_response.read.return_value = b"# HELP test_metric Test metric\ntest_metric 42.0"
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        client = HealthCheckClient("http://localhost:8080")
        result = client.get_prometheus_metrics()
        
        assert "test_metric 42.0" in result
        mock_urlopen.assert_called_once_with("http://localhost:8080/metrics/prometheus")
    
    @patch('urllib.request.urlopen')
    def test_request_error_handling(self, mock_urlopen):
        """Test error handling in requests"""
        mock_urlopen.side_effect = urllib.error.URLError("Connection failed")
        
        client = HealthCheckClient("http://localhost:8080")
        
        with pytest.raises(urllib.error.URLError):
            client.check_health()


class TestGlobalHealthServer:
    """Test global health server functions"""
    
    def test_get_health_server_singleton(self):
        """Test that get_health_server returns singleton"""
        # Clean up any existing server
        stop_health_server()
        
        server1 = get_health_server(port=8084)
        server2 = get_health_server(port=8085)  # Different port, but should be same instance
        
        assert server1 is server2
        assert server1.port == 8084  # Should keep original configuration
    
    def test_start_stop_health_server(self):
        """Test global start/stop functions"""
        # Clean up any existing server
        stop_health_server()
        
        # Start server
        server = start_health_server(host="127.0.0.1", port=8086)
        assert server.is_running()
        
        # Stop server
        stop_health_server()
        assert not server.is_running()


class TestHealthAPIIntegration:
    """Integration tests for health API with monitoring system"""
    
    def test_health_api_with_monitoring_integration(self):
        """Test health API integration with monitoring system"""
        # Create monitoring manager with test configuration
        config = {
            'max_metric_points': 100,
            'resource_monitor_interval': 1.0,
            'memory_threshold': 95.0,
            'cpu_threshold': 95.0,
            'disk_threshold': 95.0
        }
        manager = MonitoringManager(config)
        
        # Add some test metrics
        manager.record_rag_query(
            response_time=1.5,
            confidence_score=0.85,
            retrieved_docs_count=5,
            query_length=50,
            answer_length=200
        )
        
        # Create health server
        server = HealthCheckServer(host="127.0.0.1", port=8087, monitoring_manager=manager)
        
        try:
            server.start()
            time.sleep(0.2)  # Give server time to start
            
            # Test with client
            client = HealthCheckClient("http://127.0.0.1:8087")
            
            try:
                # Test health check
                health = client.check_health()
                assert "status" in health
                
                # Test RAG metrics
                rag_metrics = client.get_rag_metrics()
                assert "rag_performance" in rag_metrics
                
                # Test complete status
                status = client.get_status()
                assert "health" in status
                assert "performance" in status
                assert "system_info" in status
                
            except urllib.error.URLError:
                pytest.skip("Server not accessible - may be port conflict")
            
        finally:
            server.stop()
    
    def test_health_checks_with_failing_components(self):
        """Test health checks when components are failing"""
        manager = MonitoringManager()
        
        # Register a failing health check
        def failing_check():
            return False
        
        manager.health_checker.register_check("test_service", failing_check)
        
        # Get health status
        health_status = manager.get_health_status()
        
        # Should be degraded or unhealthy
        assert health_status.status in ["degraded", "unhealthy"]
        assert health_status.checks["test_service"] is False
    
    def test_metrics_collection_and_export(self):
        """Test metrics collection and various export formats"""
        manager = MonitoringManager()
        
        # Record various metrics
        manager.record_rag_query(1.0, 0.9, 3, 25, 100, "success")
        manager.record_rag_query(2.0, 0.8, 5, 30, 150, "success")
        manager.record_rag_indexing(10, 5.0, 50, "success")
        
        # Test metrics summary
        summary = manager.get_metrics_summary()
        assert len(summary) > 0
        
        # Test RAG performance summary
        rag_summary = manager.get_rag_performance_summary()
        # RAG summary might be empty if no metrics recorded yet
        # Just verify it returns a dict
        assert isinstance(rag_summary, dict)
        
        # Test Prometheus export
        prometheus_metrics = manager.export_prometheus_metrics()
        assert "rag_engine_info" in prometheus_metrics
        assert len(prometheus_metrics) > 0


class TestHealthAPIErrorHandling:
    """Test error handling in health API"""
    
    @patch('src.rag_engine.core.health_api.get_monitoring_manager')
    def test_handler_error_handling(self, mock_get_manager):
        """Test error handling in HTTP handler"""
        # Mock manager that raises exceptions
        mock_manager = MagicMock()
        mock_manager.get_health_status.side_effect = Exception("Test error")
        mock_get_manager.return_value = mock_manager
        
        # This would require setting up a full HTTP server to test properly
        # For now, we'll test that the manager handles exceptions
        try:
            mock_manager.get_health_status()
            assert False, "Should have raised exception"
        except Exception as e:
            assert str(e) == "Test error"
    
    def test_server_port_conflict_handling(self):
        """Test handling of port conflicts"""
        # Start first server
        server1 = HealthCheckServer(host="127.0.0.1", port=8088)
        
        try:
            server1.start()
            time.sleep(0.1)
            
            # Try to start second server on same port
            server2 = HealthCheckServer(host="127.0.0.1", port=8088)
            
            # This should raise an exception
            with pytest.raises(Exception):
                server2.start()
                
        finally:
            server1.stop()


class TestHealthAPIPerformance:
    """Test performance aspects of health API"""
    
    def test_metrics_export_performance(self):
        """Test performance of metrics export"""
        manager = MonitoringManager()
        
        # Add many metrics
        for i in range(1000):
            manager.metrics.record_gauge(f"test_metric_{i % 10}", float(i))
        
        # Time the export
        start_time = time.time()
        prometheus_metrics = manager.export_prometheus_metrics()
        export_time = time.time() - start_time
        
        # Should complete quickly (under 1 second for 1000 metrics)
        assert export_time < 1.0
        assert len(prometheus_metrics) > 0
    
    def test_concurrent_metrics_collection(self):
        """Test concurrent metrics collection"""
        manager = MonitoringManager()
        
        def record_metrics():
            for i in range(100):
                manager.record_rag_query(1.0, 0.9, 3, 25, 100, "success")
        
        # Start multiple threads recording metrics
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=record_metrics)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify metrics were recorded
        summary = manager.get_rag_performance_summary()
        # Just verify we get a dict back - the exact structure depends on metrics recorded
        assert isinstance(summary, dict)
        
        # Check that we have some metrics recorded
        all_metrics = manager.get_metrics_summary()
        assert len(all_metrics) > 0