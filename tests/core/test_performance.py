"""
Tests for the performance optimization system
"""

import pytest
import time
import threading
import asyncio
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from src.rag_engine.core.performance import (
    CacheConfig,
    ConnectionPoolConfig,
    TTLCache,
    ConnectionPool,
    AsyncExecutor,
    PerformanceOptimizer,
    get_performance_optimizer,
    cached,
    with_connection_pool,
    run_concurrent_tasks,
    batch_process
)


class TestCacheConfig:
    """Test cache configuration"""
    
    def test_default_config(self):
        """Test default cache configuration"""
        config = CacheConfig()
        
        assert config.max_size == 1000
        assert config.ttl_seconds == 3600
        assert config.enable_persistence is False
        assert config.persistence_file is None
    
    def test_custom_config(self):
        """Test custom cache configuration"""
        config = CacheConfig(
            max_size=500,
            ttl_seconds=1800,
            enable_persistence=True,
            persistence_file="/tmp/cache.pkl"
        )
        
        assert config.max_size == 500
        assert config.ttl_seconds == 1800
        assert config.enable_persistence is True
        assert config.persistence_file == "/tmp/cache.pkl"


class TestConnectionPoolConfig:
    """Test connection pool configuration"""
    
    def test_default_config(self):
        """Test default connection pool configuration"""
        config = ConnectionPoolConfig()
        
        assert config.max_connections == 10
        assert config.min_connections == 2
        assert config.connection_timeout == 30.0
        assert config.idle_timeout == 300.0
        assert config.max_retries == 3
    
    def test_custom_config(self):
        """Test custom connection pool configuration"""
        config = ConnectionPoolConfig(
            max_connections=20,
            min_connections=5,
            connection_timeout=60.0,
            idle_timeout=600.0,
            max_retries=5
        )
        
        assert config.max_connections == 20
        assert config.min_connections == 5
        assert config.connection_timeout == 60.0
        assert config.idle_timeout == 600.0
        assert config.max_retries == 5


class TestTTLCache:
    """Test TTL cache functionality"""
    
    def test_cache_put_get(self):
        """Test basic cache put and get operations"""
        cache = TTLCache(max_size=10, ttl_seconds=60)
        
        cache.put("key1", "value1")
        result = cache.get("key1")
        
        assert result == "value1"
    
    def test_cache_miss(self):
        """Test cache miss"""
        cache = TTLCache(max_size=10, ttl_seconds=60)
        
        result = cache.get("nonexistent_key")
        
        assert result is None
    
    def test_cache_expiration(self):
        """Test cache expiration"""
        cache = TTLCache(max_size=10, ttl_seconds=0.1)
        
        cache.put("key1", "value1")
        
        # Should be available immediately
        assert cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Should be expired
        assert cache.get("key1") is None
    
    def test_cache_size_limit(self):
        """Test cache size limit enforcement"""
        cache = TTLCache(max_size=2, ttl_seconds=60)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")  # Should evict key1
        
        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.size() == 2
    
    def test_cache_lru_behavior(self):
        """Test LRU behavior"""
        cache = TTLCache(max_size=2, ttl_seconds=60)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Access key1 to make it recently used
        cache.get("key1")
        
        # Add key3, should evict key2 (least recently used)
        cache.put("key3", "value3")
        
        assert cache.get("key1") == "value1"  # Still there
        assert cache.get("key2") is None     # Evicted
        assert cache.get("key3") == "value3"
    
    def test_cache_invalidate(self):
        """Test cache invalidation"""
        cache = TTLCache(max_size=10, ttl_seconds=60)
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        result = cache.invalidate("key1")
        assert result is True
        assert cache.get("key1") is None
        
        # Invalidating non-existent key
        result = cache.invalidate("nonexistent")
        assert result is False
    
    def test_cache_clear(self):
        """Test cache clear"""
        cache = TTLCache(max_size=10, ttl_seconds=60)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        assert cache.size() == 2
        
        cache.clear()
        
        assert cache.size() == 0
        assert cache.get("key1") is None
        assert cache.get("key2") is None
    
    def test_cleanup_expired(self):
        """Test cleanup of expired entries"""
        cache = TTLCache(max_size=10, ttl_seconds=0.1)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Wait for expiration
        time.sleep(0.2)
        
        expired_count = cache.cleanup_expired()
        
        assert expired_count == 2
        assert cache.size() == 0


class TestConnectionPool:
    """Test connection pool functionality"""
    
    def test_pool_initialization(self):
        """Test connection pool initialization"""
        def create_connection():
            return MagicMock()
        
        config = ConnectionPoolConfig(min_connections=2, max_connections=5)
        pool = ConnectionPool(create_connection, config)
        
        stats = pool.get_stats()
        assert stats["pool_size"] == 2  # min_connections
        assert stats["max_connections"] == 5
        assert stats["created_count"] == 2
    
    def test_get_connection(self):
        """Test getting connection from pool"""
        def create_connection():
            return MagicMock()
        
        config = ConnectionPoolConfig(min_connections=1, max_connections=3)
        pool = ConnectionPool(create_connection, config)
        
        conn = pool.get_connection()
        
        assert conn is not None
        
        stats = pool.get_stats()
        assert stats["in_use"] == 1
    
    def test_return_connection(self):
        """Test returning connection to pool"""
        def create_connection():
            return MagicMock()
        
        config = ConnectionPoolConfig(min_connections=1, max_connections=3)
        pool = ConnectionPool(create_connection, config)
        
        conn = pool.get_connection()
        pool.return_connection(conn)
        
        stats = pool.get_stats()
        assert stats["in_use"] == 0
        assert stats["pool_size"] >= 1
    
    def test_pool_exhaustion(self):
        """Test connection pool exhaustion"""
        def create_connection():
            return MagicMock()
        
        config = ConnectionPoolConfig(min_connections=0, max_connections=1)
        pool = ConnectionPool(create_connection, config)
        
        # Get the only connection
        conn1 = pool.get_connection()
        
        # Try to get another - should fail
        with pytest.raises(RuntimeError, match="Connection pool exhausted"):
            pool.get_connection()
    
    def test_connection_validation(self):
        """Test connection validation"""
        def create_connection():
            conn = MagicMock()
            conn.is_valid = True
            return conn
        
        def validate_connection(conn):
            return getattr(conn, 'is_valid', False)
        
        config = ConnectionPoolConfig(min_connections=1, max_connections=3)
        pool = ConnectionPool(create_connection, config, validate_connection)
        
        conn = pool.get_connection()
        
        # Invalidate connection
        conn.is_valid = False
        
        pool.return_connection(conn)
        
        # Pool should not reuse invalid connection
        stats = pool.get_stats()
        assert stats["pool_size"] == 0  # Invalid connection discarded
    
    def test_close_all(self):
        """Test closing all connections"""
        def create_connection():
            conn = MagicMock()
            conn.close = MagicMock()
            return conn
        
        config = ConnectionPoolConfig(min_connections=2, max_connections=5)
        pool = ConnectionPool(create_connection, config)
        
        pool.close_all()
        
        stats = pool.get_stats()
        assert stats["pool_size"] == 0
        assert stats["created_count"] == 0


class TestAsyncExecutor:
    """Test async executor functionality"""
    
    def test_executor_initialization(self):
        """Test async executor initialization"""
        executor = AsyncExecutor(max_workers=5)
        
        assert executor.max_workers == 5
        assert executor.executor is not None
    
    @pytest.mark.asyncio
    async def test_run_in_executor(self):
        """Test running function in executor"""
        executor = AsyncExecutor(max_workers=2)
        
        def cpu_bound_task(n):
            return n * 2
        
        result = await executor.run_in_executor(cpu_bound_task, 5)
        
        assert result == 10
    
    @pytest.mark.asyncio
    async def test_run_concurrent(self):
        """Test running concurrent tasks"""
        executor = AsyncExecutor(max_workers=3)
        
        def task1():
            time.sleep(0.1)
            return "task1_result"
        
        def task2():
            time.sleep(0.1)
            return "task2_result"
        
        def task3():
            time.sleep(0.1)
            return "task3_result"
        
        tasks = [
            (task1, (), {}),
            (task2, (), {}),
            (task3, (), {})
        ]
        
        start_time = time.time()
        results = await executor.run_concurrent(tasks)
        duration = time.time() - start_time
        
        assert len(results) == 3
        assert "task1_result" in results
        assert "task2_result" in results
        assert "task3_result" in results
        
        # Should be faster than sequential execution
        assert duration < 0.25  # Much less than 0.3 seconds sequential
    
    def test_shutdown(self):
        """Test executor shutdown"""
        executor = AsyncExecutor(max_workers=2)
        
        executor.shutdown(wait=True)
        
        # Executor should be shut down
        assert executor.executor._shutdown


class TestPerformanceOptimizer:
    """Test performance optimizer functionality"""
    
    def test_optimizer_initialization(self):
        """Test performance optimizer initialization"""
        config = {"max_workers": 5}
        optimizer = PerformanceOptimizer(config)
        
        assert optimizer.config == config
        assert optimizer.async_executor.max_workers == 5
    
    def test_create_cache(self):
        """Test creating cache"""
        optimizer = PerformanceOptimizer()
        config = CacheConfig(max_size=100, ttl_seconds=300)
        
        cache = optimizer.create_cache("test_cache", config)
        
        assert isinstance(cache, TTLCache)
        assert "test_cache" in optimizer.caches
        assert cache.max_size == 100
        assert cache.ttl_seconds == 300
    
    def test_get_cache(self):
        """Test getting cache"""
        optimizer = PerformanceOptimizer()
        config = CacheConfig()
        
        created_cache = optimizer.create_cache("test_cache", config)
        retrieved_cache = optimizer.get_cache("test_cache")
        
        assert created_cache is retrieved_cache
        assert optimizer.get_cache("nonexistent") is None
    
    def test_create_connection_pool(self):
        """Test creating connection pool"""
        optimizer = PerformanceOptimizer()
        
        def create_conn():
            return MagicMock()
        
        config = ConnectionPoolConfig(max_connections=5)
        
        pool = optimizer.create_connection_pool("test_pool", create_conn, config)
        
        assert isinstance(pool, ConnectionPool)
        assert "test_pool" in optimizer.connection_pools
    
    def test_get_connection_pool(self):
        """Test getting connection pool"""
        optimizer = PerformanceOptimizer()
        
        def create_conn():
            return MagicMock()
        
        config = ConnectionPoolConfig()
        
        created_pool = optimizer.create_connection_pool("test_pool", create_conn, config)
        retrieved_pool = optimizer.get_connection_pool("test_pool")
        
        assert created_pool is retrieved_pool
        assert optimizer.get_connection_pool("nonexistent") is None
    
    def test_cleanup_caches(self):
        """Test cache cleanup"""
        optimizer = PerformanceOptimizer()
        
        # Create cache with short TTL
        config = CacheConfig(ttl_seconds=0.1)
        cache = optimizer.create_cache("test_cache", config)
        
        # Add some data
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Wait for expiration
        time.sleep(0.2)
        
        cleanup_stats = optimizer.cleanup_caches()
        
        assert "test_cache" in cleanup_stats
        assert cleanup_stats["test_cache"] == 2  # 2 expired entries
    
    def test_get_performance_stats(self):
        """Test getting performance statistics"""
        optimizer = PerformanceOptimizer()
        
        # Create some components
        cache_config = CacheConfig()
        optimizer.create_cache("test_cache", cache_config)
        
        def create_conn():
            return MagicMock()
        
        pool_config = ConnectionPoolConfig()
        optimizer.create_connection_pool("test_pool", create_conn, pool_config)
        
        stats = optimizer.get_performance_stats()
        
        assert "caches" in stats
        assert "connection_pools" in stats
        assert "async_executor" in stats
        
        assert "test_cache" in stats["caches"]
        assert "test_pool" in stats["connection_pools"]
    
    def test_shutdown(self):
        """Test optimizer shutdown"""
        optimizer = PerformanceOptimizer()
        
        # Create some components
        cache_config = CacheConfig()
        cache = optimizer.create_cache("test_cache", cache_config)
        cache.put("key1", "value1")
        
        def create_conn():
            conn = MagicMock()
            conn.close = MagicMock()
            return conn
        
        pool_config = ConnectionPoolConfig()
        optimizer.create_connection_pool("test_pool", create_conn, pool_config)
        
        optimizer.shutdown()
        
        # Cache should be cleared
        assert cache.size() == 0
        
        # Executor should be shut down
        assert optimizer.async_executor.executor._shutdown


class TestGlobalFunctions:
    """Test global performance functions"""
    
    def test_get_performance_optimizer_singleton(self):
        """Test that get_performance_optimizer returns singleton"""
        optimizer1 = get_performance_optimizer()
        optimizer2 = get_performance_optimizer()
        
        assert optimizer1 is optimizer2
    
    def test_cached_decorator(self):
        """Test cached decorator"""
        call_count = 0
        
        @cached("test_cache", ttl_seconds=60)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call with same args - should use cache
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Not incremented
        
        # Different args - should call function
        result3 = expensive_function(10)
        assert result3 == 20
        assert call_count == 2
    
    def test_with_connection_pool_decorator(self):
        """Test connection pool decorator"""
        optimizer = get_performance_optimizer()
        
        def create_conn():
            conn = MagicMock()
            conn.name = "test_connection"
            return conn
        
        config = ConnectionPoolConfig()
        optimizer.create_connection_pool("test_pool", create_conn, config)
        
        @with_connection_pool("test_pool")
        def database_operation(connection, data):
            return f"processed_{data}_with_{connection.name}"
        
        result = database_operation("test_data")
        
        assert "processed_test_data_with_test_connection" in result
    
    def test_with_connection_pool_no_pool(self):
        """Test connection pool decorator without configured pool"""
        @with_connection_pool("nonexistent_pool")
        def simple_operation(data):
            return f"processed_{data}"
        
        result = simple_operation("test_data")
        
        assert result == "processed_test_data"
    
    @pytest.mark.asyncio
    async def test_run_concurrent_tasks(self):
        """Test running concurrent tasks globally"""
        def task1():
            return "result1"
        
        def task2():
            return "result2"
        
        tasks = [
            (task1, (), {}),
            (task2, (), {})
        ]
        
        results = await run_concurrent_tasks(tasks)
        
        assert len(results) == 2
        assert "result1" in results
        assert "result2" in results
    
    def test_batch_process(self):
        """Test batch processing"""
        items = list(range(10))  # [0, 1, 2, ..., 9]
        
        def processor(batch):
            return [x * 2 for x in batch]
        
        results = batch_process(items, processor, batch_size=3)
        
        expected = [x * 2 for x in items]
        assert results == expected
    
    def test_batch_process_exact_batches(self):
        """Test batch processing with exact batch sizes"""
        items = list(range(6))  # [0, 1, 2, 3, 4, 5]
        
        def processor(batch):
            return [x + 10 for x in batch]
        
        results = batch_process(items, processor, batch_size=2)
        
        expected = [x + 10 for x in items]
        assert results == expected