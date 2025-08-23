"""
Langfuse observability provider implementation
"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional
import logging

from .interfaces import (
    BaseObservabilityProvider, 
    ObservabilityConfig, 
    TraceContext, 
    SpanData, 
    LLMSpanData, 
    RetrievalSpanData, 
    EmbeddingSpanData,
    TraceLevel
)

try:
    from langfuse import Langfuse
    from langfuse.model import CreateTrace, CreateSpan, CreateGeneration
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    Langfuse = None


class LangfuseProvider(BaseObservabilityProvider):
    """Langfuse implementation of observability provider"""
    
    def __init__(self, config: ObservabilityConfig):
        super().__init__(config)
        self.client: Optional[Langfuse] = None
        self._logger = logging.getLogger(__name__)
        self._active_traces: Dict[str, Any] = {}
        self._active_spans: Dict[str, Any] = {}
    
    def initialize(self) -> bool:
        """Initialize Langfuse client"""
        if not LANGFUSE_AVAILABLE:
            self._logger.error("Langfuse is not installed. Install with: uv add langfuse")
            return False
        
        if not self.config.langfuse_secret_key or not self.config.langfuse_public_key:
            self._logger.error("Langfuse credentials not provided")
            return False
        
        try:
            self.client = Langfuse(
                secret_key=self.config.langfuse_secret_key,
                public_key=self.config.langfuse_public_key,
                host=self.config.langfuse_host or "https://cloud.langfuse.com"
            )
            
            # Test connection
            self.client.auth_check()
            self._initialized = True
            self._logger.info("Langfuse provider initialized successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize Langfuse: {e}")
            return False
    
    def create_trace(self, name: str, session_id: Optional[str] = None, 
                    user_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> TraceContext:
        """Create a new trace in Langfuse"""
        if not self._initialized or not self.should_trace():
            return TraceContext(
                trace_id=str(uuid.uuid4()),
                span_id=str(uuid.uuid4()),
                session_id=session_id,
                user_id=user_id,
                metadata=metadata or {}
            )
        
        trace_id = str(uuid.uuid4())
        
        try:
            trace = self.client.trace(
                id=trace_id,
                name=name,
                session_id=session_id,
                user_id=user_id,
                metadata=metadata or {}
            )
            
            self._active_traces[trace_id] = trace
            
            context = TraceContext(
                trace_id=trace_id,
                span_id=str(uuid.uuid4()),
                session_id=session_id,
                user_id=user_id,
                metadata=metadata or {}
            )
            
            self._logger.debug(f"Created trace: {trace_id}")
            return context
            
        except Exception as e:
            self._logger.error(f"Failed to create trace: {e}")
            return TraceContext(
                trace_id=trace_id,
                span_id=str(uuid.uuid4()),
                session_id=session_id,
                user_id=user_id,
                metadata=metadata or {}
            )
    
    def start_span(self, context: TraceContext, span_data: SpanData) -> str:
        """Start a new span within a trace"""
        if not self._initialized or not self.should_trace():
            return str(uuid.uuid4())
        
        span_id = str(uuid.uuid4())
        
        try:
            trace = self._active_traces.get(context.trace_id)
            if not trace:
                self._logger.warning(f"Trace {context.trace_id} not found for span")
                return span_id
            
            span = trace.span(
                id=span_id,
                name=span_data.name,
                start_time=span_data.start_time,
                metadata={
                    **span_data.metadata,
                    "span_type": span_data.span_type,
                    "level": span_data.level.value
                },
                input=span_data.input_data if self.config.capture_inputs else None
            )
            
            self._active_spans[span_id] = span
            self._logger.debug(f"Started span: {span_id}")
            return span_id
            
        except Exception as e:
            self._logger.error(f"Failed to start span: {e}")
            return span_id
    
    def end_span(self, context: TraceContext, span_id: str, 
                output_data: Optional[Dict[str, Any]] = None,
                status: str = "success", error_message: Optional[str] = None) -> None:
        """End a span"""
        if not self._initialized or span_id not in self._active_spans:
            return
        
        try:
            span = self._active_spans[span_id]
            
            update_data = {
                "end_time": datetime.now(),
                "level": "ERROR" if status == "error" else "DEFAULT"
            }
            
            if self.config.capture_outputs and output_data:
                update_data["output"] = output_data
            
            if error_message:
                update_data["status_message"] = error_message
            
            span.update(**update_data)
            
            del self._active_spans[span_id]
            self._logger.debug(f"Ended span: {span_id}")
            
        except Exception as e:
            self._logger.error(f"Failed to end span: {e}")
    
    def log_llm_call(self, context: TraceContext, span_data: LLMSpanData) -> str:
        """Log an LLM call as a generation in Langfuse"""
        if not self._initialized or not self.should_trace():
            return str(uuid.uuid4())
        
        generation_id = str(uuid.uuid4())
        
        try:
            trace = self._active_traces.get(context.trace_id)
            if not trace:
                self._logger.warning(f"Trace {context.trace_id} not found for LLM call")
                return generation_id
            
            generation_data = {
                "id": generation_id,
                "name": span_data.name,
                "start_time": span_data.start_time,
                "end_time": span_data.end_time or datetime.now(),
                "model": span_data.model,
                "metadata": {
                    **span_data.metadata,
                    "provider": span_data.provider,
                    "temperature": span_data.temperature,
                    "max_tokens": span_data.max_tokens
                }
            }
            
            if self.config.capture_inputs and span_data.input_data:
                generation_data["input"] = span_data.input_data
            
            if self.config.capture_outputs and span_data.output_data:
                generation_data["output"] = span_data.output_data
            
            # Add usage information if available
            if span_data.prompt_tokens or span_data.completion_tokens:
                generation_data["usage"] = {
                    "input": span_data.prompt_tokens,
                    "output": span_data.completion_tokens,
                    "total": span_data.total_tokens
                }
            
            if span_data.cost:
                generation_data["usage"]["total_cost"] = span_data.cost
            
            generation = trace.generation(**generation_data)
            self._logger.debug(f"Logged LLM call: {generation_id}")
            return generation_id
            
        except Exception as e:
            self._logger.error(f"Failed to log LLM call: {e}")
            return generation_id
    
    def log_retrieval(self, context: TraceContext, span_data: RetrievalSpanData) -> str:
        """Log a retrieval operation"""
        if not self._initialized or not self.should_trace():
            return str(uuid.uuid4())
        
        span_id = str(uuid.uuid4())
        
        try:
            trace = self._active_traces.get(context.trace_id)
            if not trace:
                self._logger.warning(f"Trace {context.trace_id} not found for retrieval")
                return span_id
            
            span = trace.span(
                id=span_id,
                name=span_data.name,
                start_time=span_data.start_time,
                end_time=span_data.end_time or datetime.now(),
                metadata={
                    **span_data.metadata,
                    "span_type": "retrieval",
                    "vector_store": span_data.vector_store,
                    "retrieved_count": span_data.retrieved_count,
                    "similarity_threshold": span_data.similarity_threshold,
                    "reranked": span_data.reranked
                },
                input={"query": span_data.query} if self.config.capture_inputs and span_data.query else None,
                output=span_data.output_data if self.config.capture_outputs else None
            )
            
            self._logger.debug(f"Logged retrieval: {span_id}")
            return span_id
            
        except Exception as e:
            self._logger.error(f"Failed to log retrieval: {e}")
            return span_id
    
    def log_embedding(self, context: TraceContext, span_data: EmbeddingSpanData) -> str:
        """Log an embedding operation"""
        if not self._initialized or not self.should_trace():
            return str(uuid.uuid4())
        
        span_id = str(uuid.uuid4())
        
        try:
            trace = self._active_traces.get(context.trace_id)
            if not trace:
                self._logger.warning(f"Trace {context.trace_id} not found for embedding")
                return span_id
            
            span = trace.span(
                id=span_id,
                name=span_data.name,
                start_time=span_data.start_time,
                end_time=span_data.end_time or datetime.now(),
                metadata={
                    **span_data.metadata,
                    "span_type": "embedding",
                    "model": span_data.model,
                    "provider": span_data.provider,
                    "input_count": span_data.input_count,
                    "embedding_dimension": span_data.embedding_dimension
                },
                input=span_data.input_data if self.config.capture_inputs else None,
                output=span_data.output_data if self.config.capture_outputs else None
            )
            
            self._logger.debug(f"Logged embedding: {span_id}")
            return span_id
            
        except Exception as e:
            self._logger.error(f"Failed to log embedding: {e}")
            return span_id
    
    def log_event(self, context: TraceContext, event_name: str, 
                 data: Optional[Dict[str, Any]] = None, level: TraceLevel = TraceLevel.INFO) -> None:
        """Log a custom event"""
        if not self._initialized or not self.should_trace():
            return
        
        try:
            trace = self._active_traces.get(context.trace_id)
            if not trace:
                self._logger.warning(f"Trace {context.trace_id} not found for event")
                return
            
            event_id = str(uuid.uuid4())
            trace.event(
                id=event_id,
                name=event_name,
                metadata={
                    "level": level.value,
                    **(data or {})
                }
            )
            
            self._logger.debug(f"Logged event: {event_name}")
            
        except Exception as e:
            self._logger.error(f"Failed to log event: {e}")
    
    def flush(self) -> None:
        """Flush any pending traces/spans"""
        if not self._initialized:
            return
        
        try:
            if self.client:
                self.client.flush()
                self._logger.debug("Flushed Langfuse data")
        except Exception as e:
            self._logger.error(f"Failed to flush Langfuse data: {e}")
    
    def shutdown(self) -> None:
        """Shutdown the provider and cleanup resources"""
        if not self._initialized:
            return
        
        try:
            # End any remaining spans
            for span_id in list(self._active_spans.keys()):
                self.end_span(TraceContext("", ""), span_id, status="interrupted")
            
            # Flush remaining data
            self.flush()
            
            self._active_traces.clear()
            self._active_spans.clear()
            self._initialized = False
            
            self._logger.info("Langfuse provider shutdown complete")
            
        except Exception as e:
            self._logger.error(f"Error during Langfuse shutdown: {e}")