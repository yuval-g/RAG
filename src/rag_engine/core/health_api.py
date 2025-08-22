"""
Health check API endpoints for monitoring system status
"""

import json
import time
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import urllib.parse

from .monitoring import get_monitoring_manager, MonitoringManager
from .logging import get_logger


class HealthCheckHandler(BaseHTTPRequestHandler):
    """HTTP handler for health check endpoints"""
    
    def __init__(self, *args, monitoring_manager: MonitoringManager = None, **kwargs):
        self.monitoring_manager = monitoring_manager or get_monitoring_manager()
        self.logger = get_logger("health_api")
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        try:
            parsed_path = urllib.parse.urlparse(self.path)
            path = parsed_path.path
            
            if path == "/health":
                self._handle_health_check()
            elif path == "/health/detailed":
                self._handle_detailed_health_check()
            elif path == "/metrics":
                self._handle_metrics()
            elif path == "/metrics/prometheus":
                self._handle_prometheus_metrics()
            elif path == "/metrics/rag":
                self._handle_rag_metrics()
            elif path == "/status":
                self._handle_status()
            else:
                self._send_error(404, "Not Found")
        except Exception as e:
            self.logger.exception(f"Error handling request: {e}")
            self._send_error(500, "Internal Server Error")
    
    def _handle_health_check(self):
        """Handle basic health check"""
        health_status = self.monitoring_manager.get_health_status()
        
        status_code = 200 if health_status.status == "healthy" else 503
        
        response = {
            "status": health_status.status,
            "message": health_status.message,
            "timestamp": health_status.timestamp.isoformat()
        }
        
        self._send_json_response(response, status_code)
    
    def _handle_detailed_health_check(self):
        """Handle detailed health check with all check results"""
        health_status = self.monitoring_manager.get_health_status()
        
        status_code = 200 if health_status.status == "healthy" else 503
        
        response = {
            "status": health_status.status,
            "message": health_status.message,
            "timestamp": health_status.timestamp.isoformat(),
            "checks": health_status.checks,
            "system_info": self._get_system_info()
        }
        
        self._send_json_response(response, status_code)
    
    def _handle_metrics(self):
        """Handle metrics endpoint"""
        metrics_summary = self.monitoring_manager.get_metrics_summary()
        
        response = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics_summary
        }
        
        self._send_json_response(response)
    
    def _handle_prometheus_metrics(self):
        """Handle Prometheus metrics endpoint"""
        metrics_text = self.monitoring_manager.export_prometheus_metrics()
        
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.end_headers()
        self.wfile.write(metrics_text.encode('utf-8'))
    
    def _handle_rag_metrics(self):
        """Handle RAG-specific metrics endpoint"""
        rag_summary = self.monitoring_manager.get_rag_performance_summary()
        
        response = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "rag_performance": rag_summary
        }
        
        self._send_json_response(response)
    
    def _handle_status(self):
        """Handle system status endpoint"""
        health_status = self.monitoring_manager.get_health_status()
        rag_summary = self.monitoring_manager.get_rag_performance_summary()
        
        response = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "health": {
                "status": health_status.status,
                "message": health_status.message,
                "checks": health_status.checks
            },
            "performance": rag_summary,
            "system_info": self._get_system_info()
        }
        
        self._send_json_response(response)
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get basic system information"""
        import psutil
        import platform
        
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "disk_total_gb": round(psutil.disk_usage('/').total / (1024**3), 2)
        }
    
    def _send_json_response(self, data: Dict[str, Any], status_code: int = 200):
        """Send JSON response"""
        json_data = json.dumps(data, indent=2, default=str)
        
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(json_data)))
        self.end_headers()
        self.wfile.write(json_data.encode('utf-8'))
    
    def _send_error(self, status_code: int, message: str):
        """Send error response"""
        error_response = {
            "error": message,
            "status_code": status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self._send_json_response(error_response, status_code)
    
    def log_message(self, format, *args):
        """Override to use our logger instead of stderr"""
        self.logger.info(f"{self.address_string()} - {format % args}")


class HealthCheckServer:
    """HTTP server for health check endpoints"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8080, 
                 monitoring_manager: Optional[MonitoringManager] = None):
        self.host = host
        self.port = port
        self.monitoring_manager = monitoring_manager or get_monitoring_manager()
        self.logger = get_logger("health_server")
        self.server = None
        self.server_thread = None
        self._running = False
    
    def start(self):
        """Start the health check server"""
        if self._running:
            self.logger.warning("Health check server is already running")
            return
        
        try:
            # Create handler class with monitoring manager
            def handler_factory(*args, **kwargs):
                return HealthCheckHandler(*args, monitoring_manager=self.monitoring_manager, **kwargs)
            
            self.server = HTTPServer((self.host, self.port), handler_factory)
            self._running = True
            
            # Start server in separate thread
            self.server_thread = threading.Thread(target=self._run_server, daemon=True)
            self.server_thread.start()
            
            self.logger.info(f"Health check server started on {self.host}:{self.port}")
            self.logger.info("Available endpoints:")
            self.logger.info("  GET /health - Basic health check")
            self.logger.info("  GET /health/detailed - Detailed health check")
            self.logger.info("  GET /metrics - All metrics summary")
            self.logger.info("  GET /metrics/prometheus - Prometheus format metrics")
            self.logger.info("  GET /metrics/rag - RAG-specific metrics")
            self.logger.info("  GET /status - Complete system status")
            
        except Exception as e:
            self.logger.exception(f"Failed to start health check server: {e}")
            self._running = False
            raise
    
    def stop(self):
        """Stop the health check server"""
        if not self._running:
            return
        
        self._running = False
        
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        
        if self.server_thread:
            self.server_thread.join(timeout=5.0)
        
        self.logger.info("Health check server stopped")
    
    def _run_server(self):
        """Run the server loop"""
        try:
            self.server.serve_forever()
        except Exception as e:
            if self._running:  # Only log if we're supposed to be running
                self.logger.exception(f"Health check server error: {e}")
    
    def is_running(self) -> bool:
        """Check if server is running"""
        return self._running
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server information"""
        return {
            "host": self.host,
            "port": self.port,
            "running": self._running,
            "endpoints": [
                "/health",
                "/health/detailed", 
                "/metrics",
                "/metrics/prometheus",
                "/metrics/rag",
                "/status"
            ]
        }


class HealthCheckClient:
    """Client for making health check requests"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip('/')
        self.logger = get_logger("health_client")
    
    def check_health(self) -> Dict[str, Any]:
        """Check basic health status"""
        return self._make_request("/health")
    
    def check_detailed_health(self) -> Dict[str, Any]:
        """Check detailed health status"""
        return self._make_request("/health/detailed")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics"""
        return self._make_request("/metrics")
    
    def get_rag_metrics(self) -> Dict[str, Any]:
        """Get RAG-specific metrics"""
        return self._make_request("/metrics/rag")
    
    def get_status(self) -> Dict[str, Any]:
        """Get complete system status"""
        return self._make_request("/status")
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus format metrics"""
        import urllib.request
        
        try:
            url = f"{self.base_url}/metrics/prometheus"
            with urllib.request.urlopen(url) as response:
                return response.read().decode('utf-8')
        except Exception as e:
            self.logger.exception(f"Failed to get Prometheus metrics: {e}")
            raise
    
    def _make_request(self, endpoint: str) -> Dict[str, Any]:
        """Make HTTP request to health check endpoint"""
        import urllib.request
        import json
        
        try:
            url = f"{self.base_url}{endpoint}"
            with urllib.request.urlopen(url) as response:
                return json.loads(response.read().decode('utf-8'))
        except Exception as e:
            self.logger.exception(f"Failed to make request to {endpoint}: {e}")
            raise


# Global health check server instance
_health_server = None


def get_health_server(host: str = "0.0.0.0", port: int = 8080, 
                     monitoring_manager: Optional[MonitoringManager] = None) -> HealthCheckServer:
    """Get the global health check server instance"""
    global _health_server
    if _health_server is None:
        _health_server = HealthCheckServer(host, port, monitoring_manager)
    return _health_server


def start_health_server(host: str = "0.0.0.0", port: int = 8080, 
                       monitoring_manager: Optional[MonitoringManager] = None):
    """Start the global health check server"""
    server = get_health_server(host, port, monitoring_manager)
    server.start()
    return server


def stop_health_server():
    """Stop the global health check server"""
    global _health_server
    if _health_server:
        _health_server.stop()
        _health_server = None