"""
Performance optimization and scaling features for the RAG system.
Implements requirement 8.4 for scaling and performance optimization.
"""

import asyncio
import time
import threading
from typing import Dict, Any, List, Optional, Callable, Union, Awaitable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from functools import wraps, lru_cache
from collections import defaultdict
import weakref
import json
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking"""
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheConfig:
    """Configuration for the caching system"""
    max_size: int = 1000
    ttl_seconds: float = 3600.0
    enable_persistence: bool = False
    persistence_file: Optional[str] = None

@dataclass
class ConnectionPoolConfig:
    """Configuration for connection pooling"""
    max_connections: int = 10
    min_connections: int = 2
    connection_timeout: float = 30.0
    idle_timeout: float = 300.0
    max_retries: int = 3
    retry_delay: float = 1.0


class PerformanceTracker:
    """Tracks performance metrics across the system"""
    
    def __init__(self):
        self._metrics: List[PerformanceMetrics] = []
        self._lock = threading.Lock()
        self._operation_stats = defaultdict(list)
    
    def record_metric(self, metric: PerformanceMetrics) -> None:
        """Record a performance metric"""
        with self._lock:
            self._metrics.append(metric)
            self._operation_stats[metric.operation_name].append(metric.duration)
    
    def get_metrics(self, operation_name: Optional[str] = None) -> List[PerformanceMetrics]:
        """Get recorded metrics"""
        with self._lock:
            if operation_name:
                return [m for m in self._metrics if m.operation_name == operation_name]
            return self._metrics.copy()
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, float]:
        """Get statistics for a specific operation"""
        with self._lock:
            durations = self._operation_stats.get(operation_name, [])
            if not durations:
                return {}
            
            return {
                "count": len(durations),
                "avg_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "total_duration": sum(durations)
            }
    
    def clear_metrics(self) -> None:
        """Clear all recorded metrics"""
        with self._lock:
            self._metrics.clear()
            self._operation_stats.clear()


# Global performance tracker instance
_performance_tracker = PerformanceTracker()


def get_performance_tracker() -> PerformanceTracker:
    """Get the global performance tracker instance"""
    return _performance_tracker


def track_performance(operation_name: str):
    """Decorator to track performance of functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            error_message = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_message = str(e)
                raise
            finally:
                end_time = time.time()
                duration = end_time - start_time
                
                metric = PerformanceMetrics(
                    operation_name=operation_name,
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    success=success,
                    error_message=error_message
                )
                
                _performance_tracker.record_metric(metric)
        
        return wrapper
    return decorator


def track_async_performance(operation_name: str):
    """Decorator to track performance of async functions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            error_message = None
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_message = str(e)
                raise
            finally:
                end_time = time.time()
                duration = end_time - start_time
                
                metric = PerformanceMetrics(
                    operation_name=operation_name,
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    success=success,
                    error_message=error_message
                )
                
                _performance_tracker.record_metric(metric)
        
        return wrapper
    return decorator

class ConnectionPool:
    """Generic connection pool for managing connections to external services"""
    
    def __init__(self, 
                 connection_factory: Callable[[], Any],
                 config: ConnectionPoolConfig,
                 connection_validator: Optional[Callable[[Any], bool]] = None):
        """
        Initialize connection pool
        
        Args:
            connection_factory: Function to create new connections
            config: Pool configuration
            connection_validator: Function to validate connections (optional)
        """
        self.connection_factory = connection_factory
        self.config = config
        self.connection_validator = connection_validator or (lambda x: True)
        
        self._pool: List[Any] = []
        self._active_connections: Dict[int, Any] = {}
        self._lock = threading.Lock()
        self._created_count = 0
        
        # Pre-create minimum connections
        self._initialize_pool()
    
    def _initialize_pool(self) -> None:
        """Initialize the pool with minimum connections"""
        for _ in range(self.config.min_connections):
            try:
                conn = self.connection_factory()
                self._pool.append(conn)
                self._created_count += 1
            except Exception as e:
                logger.warning(f"Failed to create initial connection: {e}")
    
    def get_connection(self, timeout: Optional[float] = None) -> Any:
        """
        Get a connection from the pool
        
        Args:
            timeout: Timeout for getting connection
            
        Returns:
            Connection object
        """
        timeout = timeout or self.config.connection_timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self._lock:
                # Try to get from pool
                if self._pool:
                    conn = self._pool.pop()
                    if self.connection_validator(conn):
                        conn_id = id(conn)
                        self._active_connections[conn_id] = conn
                        return conn
                    else:
                        # Connection is invalid, try to create a new one
                        continue
                
                # Create new connection if under limit
                if len(self._active_connections) < self.config.max_connections:
                    try:
                        conn = self.connection_factory()
                        conn_id = id(conn)
                        self._active_connections[conn_id] = conn
                        self._created_count += 1
                        return conn
                    except Exception as e:
                        logger.error(f"Failed to create new connection: {e}")
            
            # Wait a bit before retrying
            time.sleep(0.1)
        
        raise TimeoutError(f"Could not get connection within {timeout} seconds")
    
    def return_connection(self, connection: Any) -> None:
        """
        Return a connection to the pool
        
        Args:
            connection: Connection to return
        """
        with self._lock:
            conn_id = id(connection)
            if conn_id in self._active_connections:
                del self._active_connections[conn_id]
                
                # Validate connection before returning to pool
                if self.connection_validator(connection):
                    self._pool.append(connection)
                else:
                    logger.debug("Connection failed validation, not returning to pool")
    
    def close_all(self) -> None:
        """Close all connections in the pool"""
        with self._lock:
            # Close pooled connections
            for conn in self._pool:
                try:
                    if hasattr(conn, 'close'):
                        conn.close()
                except Exception as e:
                    logger.warning(f"Error closing pooled connection: {e}")
            
            # Close active connections
            for conn in self._active_connections.values():
                try:
                    if hasattr(conn, 'close'):
                        conn.close()
                except Exception as e:
                    logger.warning(f"Error closing active connection: {e}")
            
            self._pool.clear()
            self._active_connections.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        with self._lock:
            return {
                "pool_size": len(self._pool),
                "active_connections": len(self._active_connections),
                "total_created": self._created_count,
                "max_connections": self.config.max_connections,
                "min_connections": self.config.min_connections
            }


class CacheManager:
    """Advanced caching system with TTL and size limits"""
    
    def __init__(self, 
                 max_size: int = 1000,
                 default_ttl: float = 3600.0,
                 cleanup_interval: float = 300.0):
        """
        Initialize cache manager
        
        Args:
            max_size: Maximum number of cached items
            default_ttl: Default time-to-live in seconds
            cleanup_interval: Interval for cleanup in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = threading.Lock()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            
            # Check if expired
            if time.time() > entry['expires_at']:
                del self._cache[key]
                if key in self._access_times:
                    del self._access_times[key]
                return None
            
            # Update access time
            self._access_times[key] = time.time()
            return entry['value']
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set item in cache"""
        ttl = ttl or self.default_ttl
        expires_at = time.time() + ttl
        
        with self._lock:
            # Remove oldest items if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_oldest()
            
            self._cache[key] = {
                'value': value,
                'expires_at': expires_at,
                'created_at': time.time()
            }
            self._access_times[key] = time.time()
    
    def delete(self, key: str) -> bool:
        """Delete item from cache"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                if key in self._access_times:
                    del self._access_times[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
    
    def _evict_oldest(self) -> None:
        """Evict the oldest accessed item"""
        if not self._access_times:
            return
        
        oldest_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        if oldest_key in self._cache:
            del self._cache[oldest_key]
        del self._access_times[oldest_key]
    
    def _cleanup_loop(self) -> None:
        """Background cleanup of expired entries"""
        while True:
            try:
                time.sleep(self.cleanup_interval)
                self._cleanup_expired()
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = []
        
        with self._lock:
            for key, entry in self._cache.items():
                if current_time > entry['expires_at']:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
                if key in self._access_times:
                    del self._access_times[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hit_rate": getattr(self, '_hit_count', 0) / max(getattr(self, '_total_requests', 1), 1),
                "default_ttl": self.default_ttl
            }


# Global cache manager instance
_cache_manager = CacheManager()


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance"""
    return _cache_manager


def cached(ttl: Optional[float] = None, key_func: Optional[Callable] = None):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache_manager()
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{cache._generate_key(*args, **kwargs)}"
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator


class AsyncExecutor:
    """Async execution manager for concurrent operations"""
    
    def __init__(self, max_workers: int = 10):
        """
        Initialize async executor
        
        Args:
            max_workers: Maximum number of worker threads
        """
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def run_in_executor(self, func: Callable, *args, **kwargs) -> Any:
        """
        Run a synchronous function in the executor
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, lambda: func(*args, **kwargs))
    
    async def run_concurrent(self, tasks: List[Callable], *args, **kwargs) -> List[Any]:
        """
        Run multiple tasks concurrently
        
        Args:
            tasks: List of functions to execute
            *args: Arguments for all functions
            **kwargs: Keyword arguments for all functions
            
        Returns:
            List of results in the same order as tasks
        """
        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(self._executor, lambda t=task: t(*args, **kwargs))
            for task in tasks
        ]
        return await asyncio.gather(*futures)
    
    async def run_with_timeout(self, func: Callable, timeout: float, *args, **kwargs) -> Any:
        """
        Run function with timeout
        
        Args:
            func: Function to execute
            timeout: Timeout in seconds
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            asyncio.TimeoutError: If function times out
        """
        return await asyncio.wait_for(
            self.run_in_executor(func, *args, **kwargs),
            timeout=timeout
        )
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor"""
        self._executor.shutdown(wait=wait)


# Global async executor instance
_async_executor = AsyncExecutor()


def get_async_executor() -> AsyncExecutor:
    """Get the global async executor instance"""
    return _async_executor


def make_async(func: Callable) -> Callable:
    """Convert a synchronous function to async"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        executor = get_async_executor()
        return await executor.run_in_executor(func, *args, **kwargs)
    
    return async_wrapper


class BatchProcessor:
    """Batch processing for improved performance"""
    
    def __init__(self, 
                 batch_size: int = 10,
                 max_wait_time: float = 1.0,
                 processor_func: Optional[Callable] = None):
        """
        Initialize batch processor
        
        Args:
            batch_size: Maximum batch size
            max_wait_time: Maximum time to wait for batch to fill
            processor_func: Function to process batches
        """
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.processor_func = processor_func
        
        self._queue: List[Any] = []
        self._results: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        
        # Start processing thread
        self._processing_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._processing_thread.start()
    
    def submit(self, item: Any, item_id: str) -> Any:
        """
        Submit item for batch processing
        
        Args:
            item: Item to process
            item_id: Unique identifier for the item
            
        Returns:
            Processing result
        """
        with self._condition:
            self._queue.append((item, item_id))
            self._condition.notify()
        
        # Wait for result
        while item_id not in self._results:
            time.sleep(0.01)
        
        result = self._results.pop(item_id)
        return result
    
    def _process_loop(self) -> None:
        """Main processing loop"""
        while True:
            try:
                batch = self._get_batch()
                if batch and self.processor_func:
                    self._process_batch(batch)
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
    
    def _get_batch(self) -> List[tuple]:
        """Get a batch of items to process"""
        with self._condition:
            # Wait for items or timeout
            if not self._queue:
                self._condition.wait(timeout=self.max_wait_time)
            
            if not self._queue:
                return []
            
            # Get batch
            batch_size = min(len(self._queue), self.batch_size)
            batch = self._queue[:batch_size]
            self._queue = self._queue[batch_size:]
            
            return batch
    
    def _process_batch(self, batch: List[tuple]) -> None:
        """Process a batch of items"""
        try:
            items = [item for item, _ in batch]
            item_ids = [item_id for _, item_id in batch]
            
            # Process the batch
            results = self.processor_func(items)
            
            # Store results
            for item_id, result in zip(item_ids, results):
                self._results[item_id] = result
                
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Store error for all items in batch
            for _, item_id in batch:
                self._results[item_id] = Exception(f"Batch processing failed: {e}")