"""
Tests for the logging system
"""

import pytest
import logging
import json
import tempfile
import os
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.rag_engine.core.logging import (
    StructuredFormatter,
    RAGLogger,
    PerformanceLogger,
    setup_logging,
    get_logger,
    log_performance,
    configure_logging_from_environment
)


class TestStructuredFormatter:
    """Test structured JSON formatter"""
    
    def test_basic_formatting(self):
        """Test basic log record formatting"""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.module = "test_module"
        record.funcName = "test_function"
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test_logger"
        assert log_data["message"] == "Test message"
        assert log_data["module"] == "test_module"
        assert log_data["function"] == "test_function"
        assert log_data["line"] == 42
        assert "timestamp" in log_data
    
    def test_formatting_with_extra_fields(self):
        """Test formatting with extra fields"""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.module = "test_module"
        record.funcName = "test_function"
        record.extra_fields = {"user_id": "123", "operation": "test"}
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["user_id"] == "123"
        assert log_data["operation"] == "test"
    
    def test_formatting_with_exception(self):
        """Test formatting with exception info"""
        formatter = StructuredFormatter()
        
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            import sys
            record = logging.LogRecord(
                name="test_logger",
                level=logging.ERROR,
                pathname="",
                lineno=42,
                msg="Error occurred",
                args=(),
                exc_info=sys.exc_info()
            )
            record.module = "test_module"
            record.funcName = "test_function"
            
            formatted = formatter.format(record)
            log_data = json.loads(formatted)
            
            assert "exception" in log_data
            assert log_data["exception"]["type"] == "ValueError"
            assert log_data["exception"]["message"] == "Test exception"
            assert "traceback" in log_data["exception"]


class TestRAGLogger:
    """Test RAG logger functionality"""
    
    def test_logger_creation(self):
        """Test logger creation with extra fields"""
        extra_fields = {"component": "test", "version": "1.0"}
        logger = RAGLogger("test_logger", extra_fields)
        
        assert logger.logger.name == "test_logger"
        assert logger.extra_fields == extra_fields
    
    @patch('src.rag_engine.core.logging.logging.getLogger')
    def test_logging_methods(self, mock_get_logger):
        """Test all logging methods"""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        logger = RAGLogger("test_logger")
        
        # Test each logging method
        logger.debug("Debug message", {"key": "value"})
        logger.info("Info message", {"key": "value"})
        logger.warning("Warning message", {"key": "value"})
        logger.error("Error message", {"key": "value"})
        logger.critical("Critical message", {"key": "value"})
        
        # Verify logger was called
        assert mock_logger.handle.call_count == 5
    
    @patch('src.rag_engine.core.logging.logging.getLogger')
    def test_exception_logging(self, mock_get_logger):
        """Test exception logging"""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        logger = RAGLogger("test_logger")
        logger.exception("Exception occurred", {"context": "test"})
        
        # Verify exception logging was called
        mock_logger.handle.assert_called_once()
        call_args = mock_logger.handle.call_args[0][0]
        assert hasattr(call_args, 'extra_fields')


class TestPerformanceLogger:
    """Test performance logger functionality"""
    
    def test_operation_timing(self):
        """Test operation timing logging"""
        mock_rag_logger = MagicMock()
        perf_logger = PerformanceLogger(mock_rag_logger)
        
        perf_logger.log_operation_time("test_operation", 1.5, {"status": "success"})
        
        mock_rag_logger.info.assert_called_once()
        call_args = mock_rag_logger.info.call_args
        assert "test_operation" in call_args[0][0]
        assert "1.5000s" in call_args[0][0]
        
        extra_fields = call_args[1]["extra"]
        assert extra_fields["operation"] == "test_operation"
        assert extra_fields["duration_seconds"] == 1.5
        assert extra_fields["performance_metric"] is True
    
    def test_resource_usage_logging(self):
        """Test resource usage logging"""
        mock_rag_logger = MagicMock()
        perf_logger = PerformanceLogger(mock_rag_logger)
        
        perf_logger.log_resource_usage(512.5, 75.2, {"component": "indexer"})
        
        mock_rag_logger.info.assert_called_once()
        call_args = mock_rag_logger.info.call_args
        
        extra_fields = call_args[1]["extra"]
        assert extra_fields["memory_mb"] == 512.5
        assert extra_fields["cpu_percent"] == 75.2
        assert extra_fields["resource_metric"] is True


class TestSetupLogging:
    """Test logging setup functionality"""
    
    def test_development_setup(self):
        """Test development environment setup"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")
            
            setup_logging(
                environment="development",
                log_level="DEBUG",
                log_file=log_file,
                enable_console=True,
                enable_structured=False
            )
            
            # Test that logging works
            logger = logging.getLogger("rag_engine.test")
            logger.info("Test message")
            
            # Check log file was created
            assert os.path.exists(log_file)
    
    def test_production_setup(self):
        """Test production environment setup"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")
            
            setup_logging(
                environment="production",
                log_level="INFO",
                log_file=log_file,
                enable_console=True,
                enable_structured=True
            )
            
            # Test that logging works
            logger = logging.getLogger("rag_engine.test")
            logger.info("Test message")
            
            # Check log file was created
            assert os.path.exists(log_file)
    
    def test_console_only_setup(self):
        """Test console-only logging setup"""
        setup_logging(
            environment="testing",
            log_level="WARNING",
            log_file=None,
            enable_console=True,
            enable_structured=False
        )
        
        # Test that logging works
        logger = logging.getLogger("rag_engine.test")
        logger.warning("Test warning")


class TestLogPerformanceDecorator:
    """Test performance logging decorator"""
    
    @patch('src.rag_engine.core.logging.get_logger')
    @patch('src.rag_engine.core.logging.PerformanceLogger')
    def test_successful_operation(self, mock_perf_logger_class, mock_get_logger):
        """Test decorator with successful operation"""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_perf_logger = MagicMock()
        mock_perf_logger_class.return_value = mock_perf_logger
        
        @log_performance("test_operation")
        def test_function():
            return "success"
        
        result = test_function()
        
        assert result == "success"
        mock_logger.debug.assert_called_once()
        mock_perf_logger.log_operation_time.assert_called_once()
        
        # Check that success status was logged
        call_args = mock_perf_logger.log_operation_time.call_args
        assert call_args[0][2]["status"] == "success"
    
    @patch('src.rag_engine.core.logging.get_logger')
    @patch('src.rag_engine.core.logging.PerformanceLogger')
    def test_failed_operation(self, mock_perf_logger_class, mock_get_logger):
        """Test decorator with failed operation"""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_perf_logger = MagicMock()
        mock_perf_logger_class.return_value = mock_perf_logger
        
        @log_performance("test_operation")
        def test_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            test_function()
        
        mock_logger.debug.assert_called_once()
        mock_logger.exception.assert_called_once()
        mock_perf_logger.log_operation_time.assert_called_once()
        
        # Check that error status was logged
        call_args = mock_perf_logger.log_operation_time.call_args
        assert call_args[0][2]["status"] == "error"


class TestEnvironmentConfiguration:
    """Test environment-based configuration"""
    
    @patch.dict(os.environ, {
        'RAG_ENGINE_ENV': 'production',
        'RAG_ENGINE_LOG_LEVEL': 'WARNING',
        'RAG_ENGINE_STRUCTURED_LOGS': 'true'
    })
    @patch('src.rag_engine.core.logging.setup_logging')
    def test_environment_override(self, mock_setup_logging):
        """Test environment variable override"""
        configure_logging_from_environment()
        
        mock_setup_logging.assert_called_once()
        call_kwargs = mock_setup_logging.call_args[1]
        
        assert call_kwargs['log_level'] == 'WARNING'
        assert call_kwargs['enable_structured'] is True
    
    @patch.dict(os.environ, {}, clear=True)
    @patch('src.rag_engine.core.logging.setup_logging')
    def test_default_configuration(self, mock_setup_logging):
        """Test default configuration when no env vars set"""
        configure_logging_from_environment()
        
        mock_setup_logging.assert_called_once()
        call_kwargs = mock_setup_logging.call_args[1]
        
        # Should use development defaults
        assert call_kwargs['log_level'] == 'DEBUG'
        assert call_kwargs['enable_structured'] is False


class TestGetLogger:
    """Test logger factory function"""
    
    def test_get_logger_basic(self):
        """Test basic logger creation"""
        logger = get_logger("test_component")
        
        assert isinstance(logger, RAGLogger)
        assert logger.logger.name == "rag_engine.test_component"
    
    def test_get_logger_with_extra_fields(self):
        """Test logger creation with extra fields"""
        extra_fields = {"service": "indexer", "version": "2.0"}
        logger = get_logger("test_component", extra_fields)
        
        assert isinstance(logger, RAGLogger)
        assert logger.extra_fields == extra_fields