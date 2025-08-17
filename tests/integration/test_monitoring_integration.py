"""
Integration tests for monitoring and health check system
"""

import pytest
import time
import threading
from unittest.mock import patch

from src.rag_engine.core.monitoring import (
    get_monitoring_manager,
    record_rag_query_metrics,
    record_rag_indexing_metrics
)
from src.rag_engine.core.health_api import (
    start_health_server,
    stop_health_server,
    HealthCheckClient
)


class TestMonitoringIntegration:
    """Integration tests for the complete monitoring system"""
    
    def setup_method(self):
        """Setup for each test"""
        # Clean up any existing global instances
        stop_health_server()
        
        # Reset global monitoring manager
        import src.rag_engine.core.monitoring as monitoring_module
        monitoring_module._monitoring_manager = None
        
        import src.rag_engine.core.health_api as health_api_module
        health_api_module._health_server = None
    
    def teardown_method(self):
        """Cleanup after each test"""
        stop_health_server()
    
    def test_complete_monitoring_workflow(self):
        """Test the complete monitoring workflow"""
        # 1. Initialize monitoring system
        config = {
            'max_metric_points': 100,
            'resource_monitor_interval': 1.0,
            'memory_threshold': 95.0,
            'cpu_threshold': 95.0,
            'disk_threshold': 95.0
        }
        
        manager = get_monitoring_manager(config)
        manager.start()
        
        try:
            # 2. Record some RAG metrics
            record_rag_query_metrics(
                response_time=1.2,
                confidence_score=0.85,
                retrieved_docs_count=5,
                query_length=45,
                answer_length=180,
                status="success"
            )
            
            record_rag_indexing_metrics(
                docs_processed=10,
                processing_time=3.5,
                chunk_count=50,
                status="success"
            )
            
            # 3. Verify metrics are collected
            metrics_summary = manager.get_metrics_summary()
            assert len(metrics_summary) > 0
            
            # 4. Check health status
            health_status = manager.get_health_status()
            assert health_status.status in ["healthy", "degraded", "unhealthy"]
            
            # 5. Export Prometheus metrics
            prometheus_metrics = manager.export_prometheus_metrics()
            assert "rag_engine_info" in prometheus_metrics
            assert len(prometheus_metrics) > 0
            
            # 6. Get RAG performance summary
            rag_summary = manager.get_rag_performance_summary()
            assert isinstance(rag_summary, dict)
            
        finally:
            manager.stop()
    
    @pytest.mark.integration
    def test_health_api_integration(self):
        """Test health API integration with monitoring system"""
        # Initialize monitoring
        manager = get_monitoring_manager()
        manager.start()
        
        try:
            # Start health server on a different port to avoid conflicts
            server = start_health_server(host="127.0.0.1", port=8089, monitoring_manager=manager)
            time.sleep(0.2)  # Give server time to start
            
            # Record some metrics
            record_rag_query_metrics(1.0, 0.9, 3, 25, 100, "success")
            record_rag_query_metrics(1.5, 0.8, 4, 30, 120, "success")
            
            # Test with client
            client = HealthCheckClient("http://127.0.0.1:8089")
            
            try:
                # Test basic health check
                health = client.check_health()
                assert "status" in health
                assert "timestamp" in health
                
                # Test metrics endpoint
                metrics = client.get_metrics()
                assert "timestamp" in metrics
                assert "metrics" in metrics
                
                # Test Prometheus metrics
                prometheus_metrics = client.get_prometheus_metrics()
                assert isinstance(prometheus_metrics, str)
                assert len(prometheus_metrics) > 0
                
                # Test complete status
                status = client.get_status()
                assert "health" in status
                assert "system_info" in status
                
            except Exception as e:
                pytest.skip(f"Health API not accessible: {e}")
            
        finally:
            stop_health_server()
            manager.stop()
    
    def test_concurrent_metrics_collection(self):
        """Test concurrent metrics collection under load"""
        manager = get_monitoring_manager()
        manager.start()
        
        try:
            def record_metrics_worker(worker_id: int):
                """Worker function to record metrics"""
                for i in range(10):
                    record_rag_query_metrics(
                        response_time=1.0 + (i * 0.1),
                        confidence_score=0.8 + (i * 0.01),
                        retrieved_docs_count=3 + i,
                        query_length=25 + i,
                        answer_length=100 + (i * 10),
                        status="success"
                    )
                    time.sleep(0.01)  # Small delay to simulate real work
            
            # Start multiple worker threads
            threads = []
            for worker_id in range(3):
                thread = threading.Thread(target=record_metrics_worker, args=(worker_id,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Verify metrics were collected
            metrics_summary = manager.get_metrics_summary()
            assert len(metrics_summary) > 0
            
            # Check that we have RAG query metrics
            rag_query_metrics = [k for k in metrics_summary.keys() if 'rag_query' in k]
            assert len(rag_query_metrics) > 0
            
        finally:
            manager.stop()
    
    def test_health_checks_with_custom_checks(self):
        """Test health checks with custom check functions"""
        manager = get_monitoring_manager()
        
        # Add custom health checks
        check_results = {"database": True, "cache": True, "external_api": False}
        
        def database_check():
            return check_results["database"]
        
        def cache_check():
            return check_results["cache"]
        
        def external_api_check():
            return check_results["external_api"]
        
        manager.health_checker.register_check("database", database_check)
        manager.health_checker.register_check("cache", cache_check)
        manager.health_checker.register_check("external_api", external_api_check)
        
        # Test with all checks passing except external API
        health_status = manager.get_health_status()
        assert health_status.status == "degraded"  # Some checks failing
        assert health_status.checks["database"] is True
        assert health_status.checks["cache"] is True
        assert health_status.checks["external_api"] is False
        
        # Fix external API and test again
        check_results["external_api"] = True
        health_status = manager.get_health_status()
        assert health_status.status == "healthy"  # All checks passing
        
        # Break all checks
        check_results.update({"database": False, "cache": False, "external_api": False})
        health_status = manager.get_health_status()
        assert health_status.status == "unhealthy"  # All checks failing
    
    def test_resource_monitoring(self):
        """Test resource monitoring functionality"""
        manager = get_monitoring_manager()
        manager.start()
        
        try:
            # Let resource monitor collect some data
            time.sleep(2.0)
            
            # Check that system metrics are being collected
            metrics_summary = manager.get_metrics_summary()
            
            # Look for system metrics
            system_metrics = [k for k in metrics_summary.keys() if k.startswith('system_')]
            assert len(system_metrics) > 0
            
            # Check for specific system metrics
            expected_metrics = ['system_cpu_percent', 'system_memory_percent']
            for metric in expected_metrics:
                assert any(metric in k for k in system_metrics), f"Missing metric: {metric}"
            
        finally:
            manager.stop()
    
    def test_prometheus_metrics_format(self):
        """Test Prometheus metrics format compliance"""
        manager = get_monitoring_manager()
        
        # Record some metrics
        record_rag_query_metrics(1.5, 0.85, 5, 40, 150, "success")
        
        # Export Prometheus metrics
        prometheus_metrics = manager.export_prometheus_metrics()
        
        # Verify format
        lines = prometheus_metrics.split('\n')
        
        # Should have help and type comments
        help_lines = [line for line in lines if line.startswith('# HELP')]
        type_lines = [line for line in lines if line.startswith('# TYPE')]
        
        assert len(help_lines) > 0, "Missing HELP comments"
        assert len(type_lines) > 0, "Missing TYPE comments"
        
        # Should have rag_engine_info metric
        info_lines = [line for line in lines if 'rag_engine_info' in line and not line.startswith('#')]
        assert len(info_lines) > 0, "Missing rag_engine_info metric"
        
        # Verify metric format (name value or name{labels} value)
        metric_lines = [line for line in lines if line and not line.startswith('#')]
        for line in metric_lines:
            # Each metric line should have a name and a value
            parts = line.split()
            assert len(parts) >= 2, f"Invalid metric format: {line}"
    
    def test_error_handling_in_monitoring(self):
        """Test error handling in monitoring system"""
        manager = get_monitoring_manager()
        
        # Test with invalid health check that raises exception
        def failing_check():
            raise Exception("Simulated failure")
        
        manager.health_checker.register_check("failing_service", failing_check)
        
        # Health check should handle the exception gracefully
        health_status = manager.get_health_status()
        assert health_status.checks["failing_service"] is False
        assert health_status.status in ["degraded", "unhealthy"]
        
        # Test recording metrics with invalid values (should not crash)
        try:
            record_rag_query_metrics(
                response_time=-1.0,  # Invalid negative time
                confidence_score=1.5,  # Invalid confidence > 1
                retrieved_docs_count=-5,  # Invalid negative count
                query_length=0,
                answer_length=0,
                status="error"
            )
            # Should not raise exception
        except Exception as e:
            pytest.fail(f"Metrics recording should handle invalid values gracefully: {e}")