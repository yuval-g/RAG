"""
Observability manager for coordinating different providers
"""

import logging
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from datetime import datetime

from .interfaces import (
    BaseObservabilityProvider, 
    ObservabilityConfig, 
    ObservabilityProvider,
    TraceContext, 
    SpanData, 
    LLMSpanData, 
    RetrievalSpanData, 
    EmbeddingSpanData,
    TraceLevel,
    NoOpObservabilityProvider
)
from .langfuse_provider import LangfuseProvider
from .phoenix_provider import PhoenixProvider


class ObservabilityManager:
    """Manager for observability providers with support for multiple backends"""
    
    def __init__(self, config: ObservabilityConfig):
        self.config = config
        self._provider: BaseObservabilityProvider = NoOpObservabilityProvider(config)
        self._logger = logging.getLogger(__name__)
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize the observability manager with the configured provider"""
        if not self.config.enabled:
            self._logger.info("Observability is disabled")
            self._provider = NoOpObservabilityProvider(self.config)
            self._initialized = True
            return True
        
        # Create provider based on configuration
        if self.config.provider == ObservabilityProvider.LANGFUSE:
            self._provider = LangfuseProvider(self.config)
        elif self.config.provider == ObservabilityProvider.PHOENIX:
            self._provider = PhoenixProvider(self.config)
        else:
            self._logger.warning(f"Unknown provider {self.config.provider}, using no-op provider")
            self._provider = NoOpObservabilityProvider(self.config)
        
        # Initialize the provider
        success = self._provider.initialize()
        if success:
            self._initialized = True
            self._logger.info(f"Observability manager initialized with {self.config.provider} provider")
        else:
            self._logger.error(f"Failed to initialize {self.config.provider} provider, falling back to no-op")
            self._provider = NoOpObservabilityProvider(self.config)
            self._provider.initialize()
            self._initialized = True
        
        return self._initialized
    
    def create_trace(self, name: str, session_id: Optional[str] = None, 
                    user_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> TraceContext:
        """Create a new trace"""
        if not self._initialized:
            self.initialize()
        
        return self._provider.create_trace(name, session_id, user_id, metadata)
    
    @contextmanager
    def trace(self, name: str, session_id: Optional[str] = None, 
             user_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for creating and managing a trace"""
        context = self.create_trace(name, session_id, user_id, metadata)
        try:
            yield context
        finally:
            self.flush()
    
    @contextmanager
    def span(self, context: TraceContext, name: str, span_type: str = "generic",
             input_data: Optional[Dict[str, Any]] = None, 
             metadata: Optional[Dict[str, Any]] = None):
        """Context manager for creating and managing a span"""
        span_data = SpanData(
            name=name,
            span_type=span_type,
            start_time=datetime.now(),
            input_data=input_data,
            metadata=metadata or {}
        )
        
        span_id = self._provider.start_span(context, span_data)
        try:
            yield span_id
        except Exception as e:
            self._provider.end_span(context, span_id, status="error", error_message=str(e))
            raise
        else:
            self._provider.end_span(context, span_id, status="success")
    
    def log_llm_call(self, context: TraceContext, name: str, model: str, provider: str,
                    input_data: Optional[Dict[str, Any]] = None,
                    output_data: Optional[Dict[str, Any]] = None,
                    prompt_tokens: Optional[int] = None,
                    completion_tokens: Optional[int] = None,
                    total_tokens: Optional[int] = None,
                    cost: Optional[float] = None,
                    temperature: Optional[float] = None,
                    max_tokens: Optional[int] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """Log an LLM call"""
        span_data = LLMSpanData(
            name=name,
            span_type="llm",  # Will be overridden in __post_init__
            start_time=datetime.now(),
            end_time=datetime.now(),
            input_data=input_data,
            output_data=output_data,
            metadata=metadata or {},
            model=model,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=cost,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return self._provider.log_llm_call(context, span_data)
    
    def log_retrieval(self, context: TraceContext, name: str, query: str,
                     retrieved_count: int, vector_store: str,
                     similarity_threshold: Optional[float] = None,
                     reranked: bool = False,
                     output_data: Optional[Dict[str, Any]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """Log a retrieval operation"""
        span_data = RetrievalSpanData(
            name=name,
            span_type="retrieval",  # Will be overridden in __post_init__
            start_time=datetime.now(),
            end_time=datetime.now(),
            output_data=output_data,
            metadata=metadata or {},
            query=query,
            retrieved_count=retrieved_count,
            vector_store=vector_store,
            similarity_threshold=similarity_threshold,
            reranked=reranked
        )
        
        return self._provider.log_retrieval(context, span_data)
    
    def log_embedding(self, context: TraceContext, name: str, model: str, provider: str,
                     input_count: int, embedding_dimension: Optional[int] = None,
                     input_data: Optional[Dict[str, Any]] = None,
                     output_data: Optional[Dict[str, Any]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """Log an embedding operation"""
        span_data = EmbeddingSpanData(
            name=name,
            span_type="embedding",  # Will be overridden in __post_init__
            start_time=datetime.now(),
            end_time=datetime.now(),
            input_data=input_data,
            output_data=output_data,
            metadata=metadata or {},
            model=model,
            provider=provider,
            input_count=input_count,
            embedding_dimension=embedding_dimension
        )
        
        return self._provider.log_embedding(context, span_data)
    
    def log_event(self, context: TraceContext, event_name: str, 
                 data: Optional[Dict[str, Any]] = None, level: TraceLevel = TraceLevel.INFO) -> None:
        """Log a custom event"""
        self._provider.log_event(context, event_name, data, level)
    
    def flush(self) -> None:
        """Flush any pending traces/spans"""
        if self._initialized:
            self._provider.flush()
    
    def shutdown(self) -> None:
        """Shutdown the observability manager"""
        if self._initialized:
            self._provider.shutdown()
            self._initialized = False
    
    @property
    def is_enabled(self) -> bool:
        """Check if observability is enabled"""
        return self.config.enabled and self._initialized
    
    @property
    def provider_name(self) -> str:
        """Get the name of the current provider"""
        return self.config.provider.value if self.config.provider else "disabled"
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider"""
        return {
            "provider": self.provider_name,
            "enabled": self.is_enabled,
            "initialized": self._initialized,
            "config": {
                "trace_level": self.config.trace_level.value,
                "sample_rate": self.config.sample_rate,
                "trace_llm_calls": self.config.trace_llm_calls,
                "trace_retrieval": self.config.trace_retrieval,
                "trace_embeddings": self.config.trace_embeddings,
                "trace_evaluation": self.config.trace_evaluation,
                "capture_inputs": self.config.capture_inputs,
                "capture_outputs": self.config.capture_outputs
            }
        }