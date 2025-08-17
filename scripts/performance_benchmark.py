#!/usr/bin/env python3
"""
Performance benchmarking script for the RAG system.
Provides comprehensive performance analysis and optimization recommendations.
"""

import asyncio
import time
import statistics
import json
import argparse
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.rag_engine.core.performance import (
    PerformanceTracker,
    ConnectionPool,
    ConnectionPoolConfig,
    CacheManager,
    get_performance_tracker,
    track_performance,
    track_async_performance
)
from src.rag_engine.core.async_engine import AsyncRAGEngine, process_queries_concurrently
from src.rag_engine.core.config import PipelineConfig
from src.rag_engine.core.models import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite"""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize benchmark suite"""
        self.config = config or PipelineConfig(
            llm_provider="google",
            embedding_provider="openai",
            vector_store="chroma",
            enable_caching=True,
            enable_metrics=True
        )
        self.results: Dict[str, Any] = {}
        self.tracker = get_performance_tracker()
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all performance benchmarks"""
        logger.info("Starting comprehensive performance benchmarks...")
        
        # Clear previous metrics
        self.tracker.clear_metrics()
        
        # Run individual benchmarks
        self.results["cache_performance"] = self.benchmark_cache_performance()
        self.results["connection_pool_performance"] = self.benchmark_connection_pool()
        self.results["concurrent_operations"] = self.benchmark_concurrent_operations()
        self.results["async_vs_sync"] = asyncio.run(self.benchmark_async_vs_sync())
        self.results["memory_usage"] = self.benchmark_memory_usage()
        self.results["scalability"] = self.benchmark_scalability()
        
        # Generate recommendations
        self.results["recommendations"] = self.generate_recommendations()
        
        logger.info("All benchmarks completed")
        return self.results
    
    def benchmark_cache_performance(self) -> Dict[str, Any]:
        """Benchmark cache performance"""
        logger.info("Benchmarking cache performance...")
        
        cache = CacheManager(max_size=10000, default_ttl=3600.0)
        
        # Test cache write performance
        start_time = time.time()
        for i in range(1000):
            cache.set(f"key_{i}", f"value_{i}")
        write_time = time.time() - start_time
        
        # Test cache read performance
        start_time = time.time()
        for i in range(1000):
            cache.get(f"key_{i}")
        read_time = time.time() - start_time
        
        # Test cache hit rate
        hits = 0
        for i in range(1000):
            if cache.get(f"key_{i}") is not None:
                hits += 1
        hit_rate = hits / 1000
        
        # Test concurrent access
        def cache_worker():
            for i in range(100):
                cache.set(f"concurrent_key_{i}", f"concurrent_value_{i}")
                cache.get(f"concurrent_key_{i}")
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(cache_worker) for _ in range(10)]
            for future in as_completed(futures):
                future.result()
        concurrent_time = time.time() - start_time
        
        return {
            "write_ops_per_second": 1000 / write_time,
            "read_ops_per_second": 1000 / read_time,
            "hit_rate": hit_rate,
            "concurrent_ops_per_second": 1000 / concurrent_time,
            "write_latency_ms": write_time * 1000 / 1000,
            "read_latency_ms": read_time * 1000 / 1000
        }
    
    def benchmark_connection_pool(self) -> Dict[str, Any]:
        """Benchmark connection pool performance"""
        logger.info("Benchmarking connection pool performance...")
        
        connection_count = 0
        
        def create_connection():
            nonlocal connection_count
            connection_count += 1
            time.sleep(0.001)  # Simulate connection creation time
            return f"connection_{connection_count}"
        
        config = ConnectionPoolConfig(max_connections=10, min_connections=2)
        pool = ConnectionPool(create_connection, config)
        
        # Test connection acquisition performance
        acquisition_times = []
        for _ in range(100):
            start_time = time.time()
            conn = pool.get_connection()
            acquisition_time = time.time() - start_time
            acquisition_times.append(acquisition_time)
            pool.return_connection(conn)
        
        # Test concurrent access
        def pool_worker():
            times = []
            for _ in range(10):
                start_time = time.time()
                conn = pool.get_connection()
                acquisition_time = time.time() - start_time
                times.append(acquisition_time)
                time.sleep(0.001)  # Simulate work
                pool.return_connection(conn)
            return times
        
        concurrent_times = []
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(pool_worker) for _ in range(20)]
            for future in as_completed(futures):
                concurrent_times.extend(future.result())
        
        stats = pool.get_stats()
        pool.close_all()
        
        return {
            "avg_acquisition_time_ms": statistics.mean(acquisition_times) * 1000,
            "p95_acquisition_time_ms": statistics.quantiles(acquisition_times, n=20)[18] * 1000,
            "p99_acquisition_time_ms": statistics.quantiles(acquisition_times, n=100)[98] * 1000,
            "concurrent_avg_acquisition_time_ms": statistics.mean(concurrent_times) * 1000,
            "total_connections_created": stats["total_created"],
            "connection_reuse_rate": 1 - (stats["total_created"] / 200)  # 200 total acquisitions
        }
    
    def benchmark_concurrent_operations(self) -> Dict[str, Any]:
        """Benchmark concurrent operations performance"""
        logger.info("Benchmarking concurrent operations...")
        
        @track_performance("concurrent_operation")
        def cpu_intensive_task(n):
            # Simulate CPU-intensive work
            result = 0
            for i in range(n * 1000):
                result += i ** 0.5
            return result
        
        # Sequential execution
        start_time = time.time()
        sequential_results = []
        for i in range(10):
            result = cpu_intensive_task(100)
            sequential_results.append(result)
        sequential_time = time.time() - start_time
        
        # Concurrent execution
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(cpu_intensive_task, 100) for _ in range(10)]
            concurrent_results = [future.result() for future in as_completed(futures)]
        concurrent_time = time.time() - start_time
        
        # Analyze performance metrics
        metrics = self.tracker.get_metrics("concurrent_operation")
        operation_times = [m.duration for m in metrics]
        
        return {
            "sequential_time_seconds": sequential_time,
            "concurrent_time_seconds": concurrent_time,
            "speedup_factor": sequential_time / concurrent_time,
            "avg_operation_time_ms": statistics.mean(operation_times) * 1000,
            "operation_throughput": len(metrics) / sum(operation_times)
        }
    
    async def benchmark_async_vs_sync(self) -> Dict[str, Any]:
        """Benchmark async vs sync performance"""
        logger.info("Benchmarking async vs sync performance...")
        
        # Sync I/O simulation
        def sync_io_task(duration=0.01):
            time.sleep(duration)
            return "sync_result"
        
        # Async I/O simulation
        @track_async_performance("async_io_task")
        async def async_io_task(duration=0.01):
            await asyncio.sleep(duration)
            return "async_result"
        
        # Benchmark sync execution
        start_time = time.time()
        sync_results = []
        for _ in range(20):
            result = sync_io_task(0.01)
            sync_results.append(result)
        sync_time = time.time() - start_time
        
        # Benchmark async execution
        start_time = time.time()
        async_tasks = [async_io_task(0.01) for _ in range(20)]
        async_results = await asyncio.gather(*async_tasks)
        async_time = time.time() - start_time
        
        # Test async RAG engine performance
        async with AsyncRAGEngine(self.config) as engine:
            queries = [f"Test query {i}" for i in range(10)]
            
            start_time = time.time()
            responses = await process_queries_concurrently(engine, queries, max_concurrent=5)
            concurrent_query_time = time.time() - start_time
        
        return {
            "sync_execution_time_seconds": sync_time,
            "async_execution_time_seconds": async_time,
            "async_speedup_factor": sync_time / async_time,
            "concurrent_query_time_seconds": concurrent_query_time,
            "queries_per_second": len(queries) / concurrent_query_time
        }
    
    def benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage patterns"""
        logger.info("Benchmarking memory usage...")
        
        try:
            import psutil
            process = psutil.Process()
            
            # Baseline memory
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Test cache memory usage
            cache = CacheManager(max_size=10000, default_ttl=3600.0)
            for i in range(5000):
                cache.set(f"memory_test_key_{i}", f"memory_test_value_{i}" * 100)
            
            cache_memory = process.memory_info().rss / 1024 / 1024  # MB
            cache_overhead = cache_memory - baseline_memory
            
            # Test connection pool memory usage
            def create_connection():
                return {"data": "x" * 1000, "id": time.time()}
            
            config = ConnectionPoolConfig(max_connections=100, min_connections=10)
            pool = ConnectionPool(create_connection, config)
            
            # Create connections to fill pool
            connections = []
            for _ in range(50):
                conn = pool.get_connection()
                connections.append(conn)
            
            pool_memory = process.memory_info().rss / 1024 / 1024  # MB
            pool_overhead = pool_memory - cache_memory
            
            # Clean up
            for conn in connections:
                pool.return_connection(conn)
            pool.close_all()
            cache.clear()
            
            return {
                "baseline_memory_mb": baseline_memory,
                "cache_memory_overhead_mb": cache_overhead,
                "connection_pool_overhead_mb": pool_overhead,
                "memory_per_cached_item_kb": (cache_overhead * 1024) / 5000,
                "memory_per_connection_kb": (pool_overhead * 1024) / 50
            }
            
        except ImportError:
            logger.warning("psutil not available, skipping memory benchmarks")
            return {"error": "psutil not available for memory monitoring"}
    
    def benchmark_scalability(self) -> Dict[str, Any]:
        """Benchmark system scalability"""
        logger.info("Benchmarking scalability...")
        
        scalability_results = {}
        
        # Test cache scalability
        cache_sizes = [100, 1000, 5000, 10000]
        for size in cache_sizes:
            cache = CacheManager(max_size=size, default_ttl=3600.0)
            
            start_time = time.time()
            for i in range(size):
                cache.set(f"scale_key_{i}", f"scale_value_{i}")
            write_time = time.time() - start_time
            
            start_time = time.time()
            for i in range(size):
                cache.get(f"scale_key_{i}")
            read_time = time.time() - start_time
            
            scalability_results[f"cache_size_{size}"] = {
                "write_ops_per_second": size / write_time,
                "read_ops_per_second": size / read_time
            }
        
        # Test concurrent user scalability
        concurrent_users = [1, 5, 10, 20, 50]
        for users in concurrent_users:
            @track_performance(f"concurrent_users_{users}")
            def user_simulation():
                # Simulate user operations
                cache = CacheManager(max_size=1000, default_ttl=60.0)
                for i in range(10):
                    cache.set(f"user_key_{i}", f"user_value_{i}")
                    cache.get(f"user_key_{i}")
                return "completed"
            
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=users) as executor:
                futures = [executor.submit(user_simulation) for _ in range(users)]
                results = [future.result() for future in as_completed(futures)]
            total_time = time.time() - start_time
            
            scalability_results[f"concurrent_users_{users}"] = {
                "total_time_seconds": total_time,
                "operations_per_second": (users * 20) / total_time,  # 20 ops per user
                "avg_response_time_ms": (total_time / users) * 1000
            }
        
        return scalability_results
    
    def generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Analyze cache performance
        cache_perf = self.results.get("cache_performance", {})
        if cache_perf.get("read_ops_per_second", 0) < 10000:
            recommendations.append("Consider increasing cache size or optimizing cache key generation")
        
        if cache_perf.get("hit_rate", 0) < 0.8:
            recommendations.append("Cache hit rate is low, consider adjusting TTL or cache size")
        
        # Analyze connection pool performance
        pool_perf = self.results.get("connection_pool_performance", {})
        if pool_perf.get("connection_reuse_rate", 0) < 0.7:
            recommendations.append("Low connection reuse rate, consider increasing pool size")
        
        if pool_perf.get("p95_acquisition_time_ms", 0) > 10:
            recommendations.append("High connection acquisition latency, consider optimizing pool configuration")
        
        # Analyze async performance
        async_perf = self.results.get("async_vs_sync", {})
        if async_perf.get("async_speedup_factor", 1) < 2:
            recommendations.append("Limited async performance benefit, check for blocking operations")
        
        # Analyze memory usage
        memory_perf = self.results.get("memory_usage", {})
        if isinstance(memory_perf, dict) and memory_perf.get("cache_memory_overhead_mb", 0) > 100:
            recommendations.append("High cache memory usage, consider implementing cache eviction policies")
        
        # Analyze scalability
        scalability = self.results.get("scalability", {})
        user_50_perf = scalability.get("concurrent_users_50", {})
        if user_50_perf.get("avg_response_time_ms", 0) > 1000:
            recommendations.append("High response times under load, consider horizontal scaling")
        
        if not recommendations:
            recommendations.append("Performance is within acceptable ranges")
        
        return recommendations
    
    def save_results(self, output_file: str) -> None:
        """Save benchmark results to file"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Benchmark results saved to {output_file}")
    
    def print_summary(self) -> None:
        """Print benchmark summary"""
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("="*60)
        
        # Cache Performance
        cache_perf = self.results.get("cache_performance", {})
        print(f"\nCache Performance:")
        print(f"  Read ops/sec: {cache_perf.get('read_ops_per_second', 0):.0f}")
        print(f"  Write ops/sec: {cache_perf.get('write_ops_per_second', 0):.0f}")
        print(f"  Hit rate: {cache_perf.get('hit_rate', 0):.2%}")
        
        # Connection Pool Performance
        pool_perf = self.results.get("connection_pool_performance", {})
        print(f"\nConnection Pool Performance:")
        print(f"  Avg acquisition time: {pool_perf.get('avg_acquisition_time_ms', 0):.2f}ms")
        print(f"  P95 acquisition time: {pool_perf.get('p95_acquisition_time_ms', 0):.2f}ms")
        print(f"  Connection reuse rate: {pool_perf.get('connection_reuse_rate', 0):.2%}")
        
        # Async Performance
        async_perf = self.results.get("async_vs_sync", {})
        print(f"\nAsync Performance:")
        print(f"  Async speedup factor: {async_perf.get('async_speedup_factor', 1):.2f}x")
        print(f"  Queries per second: {async_perf.get('queries_per_second', 0):.1f}")
        
        # Memory Usage
        memory_perf = self.results.get("memory_usage", {})
        if isinstance(memory_perf, dict) and "error" not in memory_perf:
            print(f"\nMemory Usage:")
            print(f"  Baseline memory: {memory_perf.get('baseline_memory_mb', 0):.1f}MB")
            print(f"  Cache overhead: {memory_perf.get('cache_memory_overhead_mb', 0):.1f}MB")
        
        # Recommendations
        recommendations = self.results.get("recommendations", [])
        print(f"\nRecommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "="*60)


def main():
    """Main benchmark execution"""
    parser = argparse.ArgumentParser(description="RAG System Performance Benchmark")
    parser.add_argument("--output", "-o", default="benchmark_results.json",
                       help="Output file for benchmark results")
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration if provided
    config = None
    if args.config:
        from src.rag_engine.core.config import ConfigurationManager
        config_manager = ConfigurationManager(args.config)
        config = config_manager.load_config()
    
    # Run benchmarks
    benchmark = PerformanceBenchmark(config)
    results = benchmark.run_all_benchmarks()
    
    # Save and display results
    benchmark.save_results(args.output)
    benchmark.print_summary()


if __name__ == "__main__":
    main()