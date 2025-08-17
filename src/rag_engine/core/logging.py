"""
Comprehensive logging system for the RAG engine
"""

import logging
import logging.config
import json
import time
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, Union
from functools import wraps
from pathlib import Path
import sys
import os


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info and record.exc_info != (None, None, None):
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_entry, default=str)


class RAGLogger:
    """Enhanced logger for RAG engine components"""
    
    def __init__(self, name: str, extra_fields: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(name)
        self.extra_fields = extra_fields or {}
    
    def _log_with_extra(self, level: int, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log message with extra fields"""
        combined_extra = {**self.extra_fields}
        if extra:
            combined_extra.update(extra)
        
        # Create a custom LogRecord with extra fields
        record = self.logger.makeRecord(
            self.logger.name, level, "", 0, message, (), None, **kwargs
        )
        record.extra_fields = combined_extra
        self.logger.handle(record)
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log debug message"""
        self._log_with_extra(logging.DEBUG, message, extra, **kwargs)
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log info message"""
        self._log_with_extra(logging.INFO, message, extra, **kwargs)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log warning message"""
        self._log_with_extra(logging.WARNING, message, extra, **kwargs)
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log error message"""
        self._log_with_extra(logging.ERROR, message, extra, **kwargs)
    
    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log critical message"""
        self._log_with_extra(logging.CRITICAL, message, extra, **kwargs)
    
    def exception(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log exception with traceback"""
        combined_extra = {**self.extra_fields}
        if extra:
            combined_extra.update(extra)
        
        # Create a custom LogRecord with extra fields and exception info
        record = self.logger.makeRecord(
            self.logger.name, logging.ERROR, "", 0, message, (), sys.exc_info()
        )
        record.extra_fields = combined_extra
        self.logger.handle(record)


class PerformanceLogger:
    """Logger for performance metrics"""
    
    def __init__(self, logger: RAGLogger):
        self.logger = logger
    
    def log_operation_time(self, operation: str, duration: float, extra: Optional[Dict[str, Any]] = None):
        """Log operation timing"""
        timing_extra = {
            'operation': operation,
            'duration_seconds': duration,
            'performance_metric': True
        }
        if extra:
            timing_extra.update(extra)
        
        self.logger.info(f"Operation '{operation}' completed in {duration:.4f}s", extra=timing_extra)
    
    def log_resource_usage(self, memory_mb: float, cpu_percent: float, extra: Optional[Dict[str, Any]] = None):
        """Log resource usage"""
        resource_extra = {
            'memory_mb': memory_mb,
            'cpu_percent': cpu_percent,
            'resource_metric': True
        }
        if extra:
            resource_extra.update(extra)
        
        self.logger.info(f"Resource usage - Memory: {memory_mb:.2f}MB, CPU: {cpu_percent:.2f}%", extra=resource_extra)


def setup_logging(
    environment: str = "development",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_structured: bool = False
) -> None:
    """
    Setup logging configuration for the RAG engine
    
    Args:
        environment: Environment name (development, production, testing)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        enable_console: Whether to enable console logging
        enable_structured: Whether to use structured JSON logging
    """
    
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'structured': {
                '()': StructuredFormatter,
            }
        },
        'handlers': {},
        'loggers': {
            'rag_engine': {
                'level': log_level,
                'handlers': [],
                'propagate': False
            }
        },
        'root': {
            'level': log_level,
            'handlers': []
        }
    }
    
    # Add console handler if enabled
    if enable_console:
        formatter = 'structured' if enable_structured else ('detailed' if environment == 'development' else 'standard')
        config['handlers']['console'] = {
            'class': 'logging.StreamHandler',
            'level': log_level,
            'formatter': formatter,
            'stream': 'ext://sys.stdout'
        }
        config['loggers']['rag_engine']['handlers'].append('console')
        config['root']['handlers'].append('console')
    
    # Add file handler if log file specified
    if log_file:
        formatter = 'structured' if enable_structured else 'detailed'
        config['handlers']['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': log_level,
            'formatter': formatter,
            'filename': log_file,
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'encoding': 'utf8'
        }
        config['loggers']['rag_engine']['handlers'].append('file')
        config['root']['handlers'].append('file')
    
    # Apply configuration
    logging.config.dictConfig(config)


def get_logger(name: str, extra_fields: Optional[Dict[str, Any]] = None) -> RAGLogger:
    """
    Get a RAG logger instance
    
    Args:
        name: Logger name (typically module name)
        extra_fields: Additional fields to include in all log messages
    
    Returns:
        RAGLogger instance
    """
    return RAGLogger(f"rag_engine.{name}", extra_fields)


def log_performance(operation_name: str = None):
    """
    Decorator to log function performance
    
    Args:
        operation_name: Optional custom operation name
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            perf_logger = PerformanceLogger(logger)
            
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            
            try:
                logger.debug(f"Starting operation: {op_name}")
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                perf_logger.log_operation_time(op_name, duration, {'status': 'success'})
                return result
            except Exception as e:
                duration = time.time() - start_time
                perf_logger.log_operation_time(op_name, duration, {'status': 'error', 'error': str(e)})
                logger.exception(f"Operation failed: {op_name}")
                raise
        
        return wrapper
    return decorator


def log_method_calls(cls):
    """
    Class decorator to log all method calls
    """
    for attr_name in dir(cls):
        attr = getattr(cls, attr_name)
        if callable(attr) and not attr_name.startswith('_'):
            setattr(cls, attr_name, log_performance(f"{cls.__name__}.{attr_name}")(attr))
    return cls


# Environment-specific configurations
LOGGING_CONFIGS = {
    'development': {
        'log_level': 'DEBUG',
        'enable_console': True,
        'enable_structured': False,
        'log_file': 'logs/rag_engine_dev.log'
    },
    'production': {
        'log_level': 'INFO',
        'enable_console': True,
        'enable_structured': True,
        'log_file': 'logs/rag_engine_prod.log'
    },
    'testing': {
        'log_level': 'WARNING',
        'enable_console': False,
        'enable_structured': False,
        'log_file': 'logs/rag_engine_test.log'
    }
}


def configure_logging_from_environment():
    """Configure logging based on environment variables"""
    environment = os.getenv('RAG_ENGINE_ENV', 'development').lower()
    config = LOGGING_CONFIGS.get(environment, LOGGING_CONFIGS['development'])
    
    # Override with environment variables if present
    config['log_level'] = os.getenv('RAG_ENGINE_LOG_LEVEL', config['log_level'])
    config['log_file'] = os.getenv('RAG_ENGINE_LOG_FILE', config['log_file'])
    config['enable_structured'] = os.getenv('RAG_ENGINE_STRUCTURED_LOGS', str(config['enable_structured'])).lower() == 'true'
    
    setup_logging(**config)