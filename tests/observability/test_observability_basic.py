"""
Basic tests for observability system
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from src.rag_engine.observability import (
    ObservabilityConfig,
    ObservabilityProvider,
    ObservabilityManager,
    TraceLevel
)
from src.rag_engine.observability.interfaces import NoOpObservabilityProvider


class TestObservabilityConfig:
    """Test observability configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = ObservabilityConfig()
        
        assert config.provider == ObservabilityProvider.DISABLED
        assert config.enabled is False
        assert config.sample_rate == 1.0
        assert config.trace_llm_calls is True
        assert config.trace_retrieval is True
        assert config.capture_inputs is True
        assert config.capture_outputs is True
    
    def test_langfuse_config(self):
        """Test Langfuse configuration"""
        config = ObservabilityConfig(
            provider=ObservabilityProvider.LANGFUSE,
            enabled=True,
            langfuse_secret_key="sk-test",
            langfuse_public_key="pk-test",
            langfuse_host="https://test.langfuse.com"
        )
        
        assert config.provider == ObservabilityProvider.LANGFUSE
        assert config.enabled is True
        assert config.langfuse_secret_key == "sk-test"
        assert config.langfuse_public_key == "pk-test"
        assert config.langfuse_host == "https://test.langfuse.com"


class TestNoOpProvider:
    """Test the no-op observability provider"""
    
    def test_noop_provider_initialization(self):
        """Test no-op provider initializes successfully"""
        config = ObservabilityConfig()
        provider = NoOpObservabilityProvider(config)
        
        assert provider.initialize() is True
        assert provider.is_initialized is True
    
    def test_noop_provider_operations(self):
        """Test no-op provider operations don't raise errors"""
        config = ObservabilityConfig()
        provider = NoOpObservabilityProvider(config)
        provider.initialize()
        
        # Create trace
        context = provider.create_trace("test_trace")
        assert context.trace_id is not None
        assert context.span_id is not None
        
        # Start/end span
        from src.rag_engine.observability.interfaces import SpanData
        span_data = SpanData(
            name="test_span",
            span_type="test",
            start_time=datetime.now()
        )
        span_id = provider.start_span(context, span_data)
        assert span_id is not None
        
        provider.end_span(context, span_id)
        
        # Log operations
        from src.rag_engine.observability.interfaces import LLMSpanData, RetrievalSpanData, EmbeddingSpanData
        
        llm_data = LLMSpanData(
            name="test_llm",
            span_type="llm",  # This will be overridden in __post_init__
            start_time=datetime.now(),
            model="test-model",
            provider="test"
        )
        provider.log_llm_call(context, llm_data)
        
        retrieval_data = RetrievalSpanData(
            name="test_retrieval",
            span_type="retrieval",  # This will be overridden in __post_init__
            start_time=datetime.now(),
            query="test query",
            retrieved_count=5,
            vector_store="test"
        )
        provider.log_retrieval(context, retrieval_data)
        
        embedding_data = EmbeddingSpanData(
            name="test_embedding",
            span_type="embedding",  # This will be overridden in __post_init__
            start_time=datetime.now(),
            model="test-model",
            provider="test",
            input_count=1
        )
        provider.log_embedding(context, embedding_data)
        
        # Log event
        provider.log_event(context, "test_event", {"key": "value"})
        
        # Flush and shutdown
        provider.flush()
        provider.shutdown()


class TestObservabilityManager:
    """Test the observability manager"""
    
    def test_manager_disabled_by_default(self):
        """Test manager uses no-op provider when disabled"""
        config = ObservabilityConfig(enabled=False)
        manager = ObservabilityManager(config)
        
        assert manager.initialize() is True
        assert isinstance(manager._provider, NoOpObservabilityProvider)
        assert manager.provider_name == "disabled"
    
    def test_manager_with_invalid_provider(self):
        """Test manager falls back to no-op with invalid provider"""
        config = ObservabilityConfig(
            enabled=True,
            provider="invalid_provider"  # This will be converted to enum
        )
        manager = ObservabilityManager(config)
        
        # Should fall back to no-op provider
        assert manager.initialize() is True
        assert isinstance(manager._provider, NoOpObservabilityProvider)
    
    @patch('src.rag_engine.observability.langfuse_provider.LANGFUSE_AVAILABLE', False)
    def test_manager_langfuse_unavailable(self):
        """Test manager falls back when Langfuse is unavailable"""
        config = ObservabilityConfig(
            enabled=True,
            provider=ObservabilityProvider.LANGFUSE,
            langfuse_secret_key="sk-test",
            langfuse_public_key="pk-test"
        )
        manager = ObservabilityManager(config)
        
        # Should fall back to no-op provider
        assert manager.initialize() is True
        assert isinstance(manager._provider, NoOpObservabilityProvider)
    
    def test_manager_trace_context_manager(self):
        """Test manager trace context manager"""
        config = ObservabilityConfig(enabled=False)
        manager = ObservabilityManager(config)
        manager.initialize()
        
        with manager.trace("test_trace") as context:
            assert context.trace_id is not None
            
            with manager.span(context, "test_span", "test") as span_id:
                assert span_id is not None
    
    def test_manager_logging_methods(self):
        """Test manager logging methods"""
        config = ObservabilityConfig(enabled=False)
        manager = ObservabilityManager(config)
        manager.initialize()
        
        context = manager.create_trace("test")
        
        # Test LLM logging
        llm_id = manager.log_llm_call(
            context=context,
            name="test_llm",
            model="test-model",
            provider="test",
            prompt_tokens=10,
            completion_tokens=20
        )
        assert llm_id is not None
        
        # Test retrieval logging
        retrieval_id = manager.log_retrieval(
            context=context,
            name="test_retrieval",
            query="test query",
            retrieved_count=5,
            vector_store="test"
        )
        assert retrieval_id is not None
        
        # Test embedding logging
        embedding_id = manager.log_embedding(
            context=context,
            name="test_embedding",
            model="test-model",
            provider="test",
            input_count=1
        )
        assert embedding_id is not None
        
        # Test event logging
        manager.log_event(context, "test_event", {"key": "value"})
    
    def test_manager_info_methods(self):
        """Test manager info methods"""
        config = ObservabilityConfig(
            enabled=True,
            provider=ObservabilityProvider.LANGFUSE,
            sample_rate=0.5
        )
        manager = ObservabilityManager(config)
        manager.initialize()
        
        info = manager.get_provider_info()
        assert "provider" in info
        assert "enabled" in info
        assert "config" in info
        assert info["config"]["sample_rate"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__])