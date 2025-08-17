"""
Performance tests and benchmarks for the RAG system.
Tests requirement 8.4 for scaling and performance optimization.
"""

import pytest
import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import statistics
from functools import wraps
import logging

logger = logging.getLogger(__name__)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from rag_engine.core.performance import (
    PerformanceTracker,
    ConnectionPool,
    ConnectionPoolConfig,
    CacheManager,
    AsyncExecutor,
    BatchProcessor,
    track_performance,
    track_async_performance,
    get_performance_tracker,
    get_cache_manager,
    get_async_executor
)
from rag_engine.core.async_interfaces import AsyncConnectionPool
from rag_engine.core.async_engine import AsyncRAGEngine, process_queries_concurrently
from rag_engine.core.config import PipelineConfig
from rag_engine.core.models import Document


class TestPerformanceTracker:
    """Test performance tracking functionality"""
    
    def test_performance_tracker_basic(self):
        """Test basic performance tracking"""
        tracker = PerformanceTracker()
        
        @track_performance("test_operation")
        def test_function():
            time.sleep(0.1)
            return "result"
        
        result = test_function()
        assert result == "result"
        
        metrics = tracker.get_metrics("test_operation")
        assert len(metrics) == 1
        assert metrics[0].operation_name == "test_operation"
        assert metrics[0].duration >= 0.1
        assert metrics[0].success is True
    
    def test_performance_tracker_error_handling(self):
        """Test performance tracking with errors"""
        tracker = PerformanceTracker()
        
        @track_performance("test_error_operation")
        def test_error_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            test_error_function()
        
        metrics = tracker.get_metrics("test_error_operation")
        assert len(metrics) == 1
        assert metrics[0].success is False
        assert metrics[0].error_message == "Test error"
    
    def test_performance_stats(self):
        """Test performance statistics calculation"""
        tracker = PerformanceTracker()
        
        @track_performance("stats_test")
        def test_function(duration):
            time.sleep(duration)
            return "result"
        
        # Run multiple times with different durations
        durations = [0.1, 0.2, 0.15, 0.25, 0.12]
        for duration in durations:
            test_function(duration)
        
        stats = tracker.get_operation_stats("stats_test")
        assert stats["count"] == 5
        assert abs(stats["avg_duration"] - statistics.mean(durations)) < 0.05
        assert stats["min_duration"] >= min(durations) - 0.01
        assert stats["max_duration"] <= max(durations) + 0.01


class TestConnectionPool:
    """Test connection pooling functionality"""
    
    def test_connection_pool_basic(self):
        """Test basic connection pool operations"""
        connection_count = 0
        
        def create_connection():
            nonlocal connection_count
            connection_count += 1
            return f"connection_{connection_count}"
        
        config = ConnectionPoolConfig(max_connections=3, min_connections=1)
        pool = ConnectionPool(create_connection, config)
        
        # Test getting connections
        conn1 = pool.get_connection()
        conn2 = pool.get_connection()
        
        assert conn1 != conn2
        assert connection_count >= 1
        
        # Test returning connections
        pool.return_connection(conn1)
        pool.return_connection(conn2)
        
        # Test reusing connections
        conn3 = pool.get_connection()
        assert conn3 in [conn1, conn2]
        
        pool.close_all()
    
    def test_connection_pool_max_connections(self):
        """Test connection pool maximum connections limit"""
        def create_connection():
            return {"id": time.time()}
        
        config = ConnectionPoolConfig(max_connections=2, min_connections=0)
        pool = ConnectionPool(create_connection, config)
        
        # Get maximum connections
        conn1 = pool.get_connection()
        conn2 = pool.get_connection()
        
        # Try to get one more (should timeout)
        start_time = time.time()
        with pytest.raises(TimeoutError):
            pool.get_connection(timeout=0.5)
        
        elapsed = time.time() - start_time
        assert elapsed >= 0.5
        
        pool.close_all()
    
    def test_connection_pool_validation(self):
        """Test connection validation"""
        def create_connection():
            return {"valid": True, "id": time.time()}
        
        def validate_connection(conn):
            return conn.get("valid", False)
        
        config = ConnectionPoolConfig(max_connections=3, min_connections=1)
        pool = ConnectionPool(create_connection, config, validate_connection)
        
        conn = pool.get_connection()
        assert conn["valid"] is True
        
        # Invalidate connection
        conn["valid"] = False
        pool.return_connection(conn)
        
        # Should create new connection since old one is invalid
        new_conn = pool.get_connection()
        assert new_conn["valid"] is True
        assert new_conn["id"] != conn["id"]
        
        pool.close_all()


class TestCacheManager:
    """Test caching functionality"""
    
    def test_cache_basic_operations(self):
        """Test basic cache operations"""
        cache = CacheManager(max_size=100, default_ttl=1.0)
        
        # Test set and get
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Test non-existent key
        assert cache.get("nonexistent") is None
        
        # Test delete
        assert cache.delete("key1") is True
        assert cache.get("key1") is None
        assert cache.delete("key1") is False
    
    def test_cache_ttl(self):
        """Test cache TTL functionality"""
        cache = CacheManager(max_size=100, default_ttl=0.2)
        
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(0.3)
        assert cache.get("key1") is None
    
    def test_cache_size_limit(self):
        """Test cache size limits"""
        cache = CacheManager(max_size=3, default_ttl=10.0)
        
        # Fill cache to capacity
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        # Add one more (should evict oldest)
        cache.set("key4", "value4")
        
        # Check that oldest was evicted
        assert cache.get("key1") is None
        assert cache.get("key4") == "value4"
    
    def test_cached_decorator(self):
        """Test cached decorator"""
        from rag_engine.core.performance import cached
        
        call_count = 0
        
        @cached(ttl=1.0)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call (should use cache)
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1
        
        # Different argument (should call function)
        result3 = expensive_function(6)
        assert result3 == 12
        assert call_count == 2


class TestAsyncPerformance:
    """Test async performance features"""
    
    @pytest.mark.asyncio
    async def test_async_performance_tracking(self):
        """Test async performance tracking"""
        tracker = PerformanceTracker()
        
        @track_async_performance("async_test_operation")
        async def async_test_function():
            await asyncio.sleep(0.1)
            return "async_result"
        
        result = await async_test_function()
        assert result == "async_result"
        
        metrics = tracker.get_metrics("async_test_operation")
        assert len(metrics) == 1
        assert metrics[0].operation_name == "async_test_operation"
        assert metrics[0].duration >= 0.1
        assert metrics[0].success is True
    
    @pytest.mark.asyncio
    async def test_async_connection_pool(self):
        """Test async connection pool"""
        connection_count = 0
        
        async def create_connection():
            nonlocal connection_count
            connection_count += 1
            await asyncio.sleep(0.01)  # Simulate async connection creation
            return f"async_connection_{connection_count}"
        
        pool = AsyncConnectionPool(create_connection, max_connections=3)
        
        # Test getting connections
        conn1 = await pool.get_connection()
        conn2 = await pool.get_connection()
        
        assert conn1 != conn2
        assert connection_count >= 1
        
        # Test returning connections
        await pool.return_connection(conn1)
        await pool.return_connection(conn2)
        
        await pool.close_all()
    
    @pytest.mark.asyncio
    async def test_async_rag_engine_basic(self):
        """Test basic async RAG engine functionality"""
        config = PipelineConfig(
            llm_provider="google",
            embedding_provider="openai",
            vector_store="chroma",
            enable_caching=True
        )
        
        async with AsyncRAGEngine(config) as engine:
            # Test system info
            info = await engine.get_system_info()
            assert info["async_enabled"] is True
            assert "performance" in info
            
            # Test query (will return placeholder response)
            response = await engine.query("What is the capital of France?")
            assert response.answer is not None
            assert response.processing_time >= 0
    
    @pytest.mark.asyncio
    async def test_concurrent_query_processing(self):
        """Test concurrent query processing"""
        config = PipelineConfig(
            llm_provider="google",
            embedding_provider="openai",
            vector_store="chroma"
        )
        
        async with AsyncRAGEngine(config) as engine:
            queries = [
                "What is machine learning?",
                "How does neural network work?",
                "What is deep learning?",
                "Explain artificial intelligence",
                "What is natural language processing?"
            ]
            
            start_time = time.time()
            responses = await process_queries_concurrently(engine, queries, max_concurrent=3)
            elapsed_time = time.time() - start_time
            
            assert len(responses) == len(queries)
            # Concurrent processing should be faster than sequential
            assert elapsed_time < len(queries) * 0.5  # Assuming each query takes less than 0.5s


class TestBatchProcessor:
    """Test batch processing functionality"""
    
    def test_batch_processor_basic(self):
        """Test basic batch processing"""
        processed_batches = []
        
        def process_batch(items):
            processed_batches.append(items)
            return [f"processed_{item}" for item in items]
        
        processor = BatchProcessor(
            batch_size=3,
            max_wait_time=0.1,
            processor_func=process_batch
        )
        
        # Submit items
        results = []
        for i in range(5):
            result = processor.submit(f"item_{i}", f"id_{i}")
            results.append(result)
        
        # Wait a bit for processing
        time.sleep(0.2)
        
        assert len(results) == 5
        assert len(processed_batches) >= 1
        
        # Check that items were processed in batches
        total_processed = sum(len(batch) for batch in processed_batches)
        assert total_processed == 5


class TestPerformanceBenchmarks:
    """Performance benchmarks and stress tests"""
    
    def test_concurrent_cache_access(self):
        """Test cache performance under concurrent access"""
        cache = CacheManager(max_size=1000, default_ttl=10.0)
        
        def cache_worker(worker_id):
            for i in range(100):
                key = f"worker_{worker_id}_key_{i}"
                value = f"worker_{worker_id}_value_{i}"
                cache.set(key, value)
                retrieved = cache.get(key)
                assert retrieved == value
        
        # Run multiple workers concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(cache_worker, i) for i in range(10)]
            for future in as_completed(futures):
                future.result()  # Will raise exception if worker failed
    
    def test_connection_pool_stress(self):
        """Test connection pool under stress"""
        connection_count = 0
        
        def create_connection():
            nonlocal connection_count
            connection_count += 1
            time.sleep(0.01)  # Simulate connection creation time
            return f"connection_{connection_count}"
        
        config = ConnectionPoolConfig(max_connections=5, min_connections=2)
        pool = ConnectionPool(create_connection, config)
        
        def pool_worker(worker_id):
            connections = []
            for i in range(10):
                conn = pool.get_connection()
                connections.append(conn)
                time.sleep(0.001)  # Simulate work
            
            for conn in connections:
                pool.return_connection(conn)
        
        # Run multiple workers concurrently
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(pool_worker, i) for i in range(20)]
            for future in as_completed(futures):
                future.result()
        
        stats = pool.get_stats()
        assert stats["total_created"] <= 20  # Should reuse connections
        
        pool.close_all()
    
    @pytest.mark.asyncio
    async def test_async_performance_benchmark(self):
        """Benchmark async vs sync performance"""
        # Simulate work function
        def sync_work(duration=0.01):
            time.sleep(duration)
            return "sync_result"
        
        async def async_work(duration=0.01):
            await asyncio.sleep(duration)
            return "async_result"
        
        # Benchmark sync execution
        start_time = time.time()
        sync_results = []
        for _ in range(10):
            result = sync_work()
            sync_results.append(result)
        sync_time = time.time() - start_time
        
        # Benchmark async execution
        start_time = time.time()
        async_tasks = [async_work() for _ in range(10)]
        async_results = await asyncio.gather(*async_tasks)
        async_time = time.time() - start_time
        
        assert len(sync_results) == 10
        assert len(async_results) == 10
        
        # Async should be significantly faster for concurrent operations
        assert async_time < sync_time * 0.5
    
    def test_performance_metrics_collection(self):
        """Test comprehensive performance metrics collection"""
        tracker = get_performance_tracker()
        tracker.clear_metrics()
        
        @track_performance("benchmark_operation")
        def benchmark_function(work_amount):
            # Simulate variable work
            time.sleep(work_amount * 0.01)
            return f"completed_{work_amount}"
        
        # Run benchmark with different work amounts
        work_amounts = [1, 2, 3, 4, 5, 3, 2, 1, 4, 5]
        for amount in work_amounts:
            benchmark_function(amount)
        
        # Analyze performance metrics
        stats = tracker.get_operation_stats("benchmark_operation")
        
        assert stats["count"] == len(work_amounts)
        assert stats["avg_duration"] > 0
        assert stats["min_duration"] > 0
        assert stats["max_duration"] >= stats["min_duration"]
        assert stats["total_duration"] > 0
        
        # Check that metrics correlate with work amount
        metrics = tracker.get_metrics("benchmark_operation")
        durations = [m.duration for m in metrics]
        
        # Higher work amounts should generally take longer
        # (allowing for some variance due to system scheduling)
        max_duration_idx = durations.index(max(durations))
        min_duration_idx = durations.index(min(durations))
        
        assert work_amounts[max_duration_idx] >= work_amounts[min_duration_idx]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])