"""
Example demonstrating monitoring and health check functionality
"""

import time
import threading
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_engine.core.monitoring import (
    get_monitoring_manager,
    record_rag_query_metrics,
    record_rag_indexing_metrics,
    get_rag_performance_summary
)
from src.rag_engine.core.health_api import (
    start_health_server,
    stop_health_server,
    HealthCheckClient
)


def simulate_rag_operations():
    """Simulate RAG operations to generate metrics"""
    print("🔄 Simulating RAG operations...")
    
    # Simulate various query operations
    queries = [
        {"response_time": 1.2, "confidence": 0.85, "docs": 5, "query_len": 45, "answer_len": 180},
        {"response_time": 0.8, "confidence": 0.92, "docs": 3, "query_len": 30, "answer_len": 150},
        {"response_time": 2.1, "confidence": 0.78, "docs": 7, "query_len": 60, "answer_len": 220},
        {"response_time": 1.5, "confidence": 0.88, "docs": 4, "query_len": 35, "answer_len": 190},
        {"response_time": 0.9, "confidence": 0.95, "docs": 2, "query_len": 25, "answer_len": 120},
    ]
    
    for i, query in enumerate(queries):
        print(f"  📝 Processing query {i+1}/5...")
        record_rag_query_metrics(
            response_time=query["response_time"],
            confidence_score=query["confidence"],
            retrieved_docs_count=query["docs"],
            query_length=query["query_len"],
            answer_length=query["answer_len"],
            status="success"
        )
        time.sleep(0.5)  # Simulate processing time
    
    # Simulate indexing operations
    print("  📚 Simulating document indexing...")
    record_rag_indexing_metrics(
        docs_processed=25,
        processing_time=8.5,
        chunk_count=150,
        status="success"
    )
    
    # Simulate some failed operations
    print("  ❌ Simulating some failed operations...")
    record_rag_query_metrics(
        response_time=5.0,
        confidence_score=0.3,
        retrieved_docs_count=0,
        query_length=40,
        answer_length=50,
        status="error"
    )


def demonstrate_monitoring_system():
    """Demonstrate the monitoring system functionality"""
    print("🚀 Starting Monitoring and Health Check Demo")
    print("=" * 50)
    
    # Get monitoring manager
    config = {
        'max_metric_points': 1000,
        'resource_monitor_interval': 5.0,
        'memory_threshold': 85.0,
        'cpu_threshold': 80.0,
        'disk_threshold': 90.0
    }
    
    manager = get_monitoring_manager(config)
    
    # Start monitoring
    print("📊 Starting monitoring system...")
    manager.start()
    
    # Start health check server
    print("🏥 Starting health check server on http://localhost:8080...")
    try:
        server = start_health_server(host="localhost", port=8080, monitoring_manager=manager)
        print("✅ Health check server started successfully!")
        print("\n🌐 Available endpoints:")
        print("  • http://localhost:8080/health - Basic health check")
        print("  • http://localhost:8080/health/detailed - Detailed health check")
        print("  • http://localhost:8080/metrics - All metrics")
        print("  • http://localhost:8080/metrics/prometheus - Prometheus format")
        print("  • http://localhost:8080/metrics/rag - RAG-specific metrics")
        print("  • http://localhost:8080/status - Complete system status")
        
        # Give server time to start
        time.sleep(1)
        
        # Simulate RAG operations
        simulate_rag_operations()
        
        # Demonstrate health check client
        print("\n🔍 Testing health check endpoints...")
        client = HealthCheckClient("http://localhost:8080")
        
        try:
            # Test basic health check
            print("  📋 Basic health check:")
            health = client.check_health()
            print(f"    Status: {health['status']}")
            print(f"    Message: {health['message']}")
            
            # Test detailed health check
            print("\n  📋 Detailed health check:")
            detailed_health = client.check_detailed_health()
            print(f"    Status: {detailed_health['status']}")
            print(f"    Checks: {detailed_health['checks']}")
            
            # Test RAG metrics
            print("\n  📊 RAG Performance Metrics:")
            rag_metrics = client.get_rag_metrics()
            if 'rag_performance' in rag_metrics:
                perf = rag_metrics['rag_performance']
                if 'response_time' in perf:
                    rt = perf['response_time']
                    print(f"    Average Response Time: {rt.get('avg_seconds', 0):.2f}s")
                    print(f"    Min Response Time: {rt.get('min_seconds', 0):.2f}s")
                    print(f"    Max Response Time: {rt.get('max_seconds', 0):.2f}s")
                
                if 'accuracy' in perf:
                    acc = perf['accuracy']
                    print(f"    Average Confidence: {acc.get('avg_confidence', 0):.2f}")
                    print(f"    Min Confidence: {acc.get('min_confidence', 0):.2f}")
                    print(f"    Max Confidence: {acc.get('max_confidence', 0):.2f}")
                
                if 'queries' in perf:
                    queries = perf['queries']
                    print(f"    Total Queries: {queries.get('total_queries', 0)}")
            
            # Test Prometheus metrics
            print("\n  📈 Prometheus Metrics Sample:")
            prometheus_metrics = client.get_prometheus_metrics()
            lines = prometheus_metrics.split('\n')[:10]  # Show first 10 lines
            for line in lines:
                if line.strip():
                    print(f"    {line}")
            print("    ... (truncated)")
            
            # Test complete status
            print("\n  🔍 Complete System Status:")
            status = client.get_status()
            print(f"    Health Status: {status['health']['status']}")
            print(f"    System Info: {status.get('system_info', {}).get('platform', 'Unknown')}")
            
        except Exception as e:
            print(f"  ❌ Error testing endpoints: {e}")
            print("  💡 Make sure the server is running and accessible")
        
        # Demonstrate direct monitoring manager usage
        print("\n📊 Direct Monitoring Manager Usage:")
        
        # Get metrics summary
        metrics_summary = manager.get_metrics_summary()
        print(f"  Total metrics tracked: {len(metrics_summary)}")
        
        # Get RAG performance summary
        rag_summary = manager.get_rag_performance_summary()
        print(f"  RAG performance categories: {list(rag_summary.keys())}")
        
        # Get health status
        health_status = manager.get_health_status()
        print(f"  System health: {health_status.status}")
        print(f"  Health checks: {list(health_status.checks.keys())}")
        
        # Export Prometheus metrics
        prometheus_export = manager.export_prometheus_metrics()
        metric_lines = len([line for line in prometheus_export.split('\n') if line.strip() and not line.startswith('#')])
        print(f"  Prometheus metrics exported: {metric_lines} metrics")
        
        print("\n⏱️  Letting system run for 10 seconds to collect resource metrics...")
        time.sleep(10)
        
        # Show updated metrics
        print("\n📈 Updated Metrics After Resource Collection:")
        updated_summary = manager.get_metrics_summary()
        
        # Look for system metrics
        system_metrics = {k: v for k, v in updated_summary.items() if k.startswith('system_')}
        if system_metrics:
            print("  System Resource Metrics:")
            for metric_name, summary in system_metrics.items():
                if 'latest' in summary:
                    value = summary['latest']
                    if 'cpu' in metric_name:
                        print(f"    {metric_name}: {value:.1f}%")
                    elif 'memory' in metric_name and 'mb' in metric_name:
                        print(f"    {metric_name}: {value:.1f} MB")
                    elif 'memory' in metric_name and 'percent' in metric_name:
                        print(f"    {metric_name}: {value:.1f}%")
                    else:
                        print(f"    {metric_name}: {value}")
        
        print("\n✅ Demo completed successfully!")
        print("\n💡 Tips:")
        print("  • Use the health endpoints for monitoring dashboards")
        print("  • Integrate Prometheus metrics with your monitoring stack")
        print("  • Set up alerts based on health check status")
        print("  • Monitor RAG performance metrics for optimization")
        
    except Exception as e:
        print(f"❌ Error starting health server: {e}")
        print("💡 Try using a different port if 8080 is already in use")
    
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        stop_health_server()
        manager.stop()
        print("✅ Cleanup completed")


def demonstrate_custom_health_checks():
    """Demonstrate adding custom health checks"""
    print("\n🔧 Demonstrating Custom Health Checks")
    print("-" * 40)
    
    manager = get_monitoring_manager()
    
    # Add custom health checks
    def check_database_connection():
        """Simulate database connection check"""
        # In real implementation, this would check actual database
        import random
        return random.choice([True, True, True, False])  # 75% success rate
    
    def check_external_api():
        """Simulate external API check"""
        # In real implementation, this would check actual API
        return True
    
    def check_disk_space():
        """Check if we have enough disk space"""
        import psutil
        disk = psutil.disk_usage('/')
        usage_percent = (disk.used / disk.total) * 100
        return usage_percent < 95.0  # Fail if over 95% full
    
    # Register custom checks
    manager.health_checker.register_check("database", check_database_connection)
    manager.health_checker.register_check("external_api", check_external_api)
    manager.health_checker.register_check("disk_space", check_disk_space)
    
    print("✅ Registered custom health checks:")
    print("  • database - Database connection check")
    print("  • external_api - External API availability")
    print("  • disk_space - Disk space availability")
    
    # Run health checks multiple times to show different results
    for i in range(3):
        print(f"\n🔍 Health Check Run {i+1}:")
        health_status = manager.get_health_status()
        print(f"  Overall Status: {health_status.status}")
        print(f"  Message: {health_status.message}")
        print("  Individual Checks:")
        for check_name, result in health_status.checks.items():
            status_icon = "✅" if result else "❌"
            print(f"    {status_icon} {check_name}: {'PASS' if result else 'FAIL'}")
        
        time.sleep(2)


if __name__ == "__main__":
    try:
        # Run main demonstration
        demonstrate_monitoring_system()
        
        # Run custom health checks demonstration
        demonstrate_custom_health_checks()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
        stop_health_server()
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        stop_health_server()
    
    print("\n👋 Demo finished!")