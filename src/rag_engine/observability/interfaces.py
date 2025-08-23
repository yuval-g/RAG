"""
Base interfaces for observability providers
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import uuid
from datetime import datetime


class ObservabilityProvider(str, Enum):
    """Supported observability providers"""
    LANGFUSE = "langfuse"
    PHOENIX = "phoenix"
    DISABLED = "disabled"


class TraceLevel(str, Enum):
    """Trace levels for different operations"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class ObservabilityConfig:
    """Configuration for observability providers"""
    provider: ObservabilityProvider = ObservabilityProvider.DISABLED
    enabled: bool = False
    
    # Langfuse specific
    langfuse_secret_key: Optional[str] = None
    langfuse_public_key: Optional[str] = None
    langfuse_host: Optional[str] = None
    
    # Phoenix specific
    phoenix_endpoint: Optional[str] = None
    phoenix_api_key: Optional[str] = None
    
    # General settings
    trace_level: TraceLevel = TraceLevel.INFO
    sample_rate: float = 1.0  # 0.0 to 1.0
    flush_interval: int = 10  # seconds
    max_batch_size: int = 100
    
    # Feature flags
    trace_llm_calls: bool = True
    trace_retrieval: bool = True
    trace_embeddings: bool = True
    trace_evaluation: bool = True
    capture_inputs: bool = True
    capture_outputs: bool = True
    capture_metadata: bool = True


@dataclass
class TraceContext:
    """Context information for a trace"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    tags: Dict[str, str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SpanData:
    """Data structure for span information"""
    name: str
    span_type: str  # "llm", "retrieval", "embedding", "evaluation", "generation"
    start_time: datetime
    end_time: Optional[datetime] = None
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[Dict[str, str]] = None
    level: TraceLevel = TraceLevel.INFO
    status: str = "success"  # "success", "error", "warning"
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.tags is None:
            self.tags = {}


@dataclass
class LLMSpanData(SpanData):
    """Specialized span data for LLM calls"""
    model: Optional[str] = None
    provider: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost: Optional[float] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.span_type = "llm"


@dataclass
class RetrievalSpanData(SpanData):
    """Specialized span data for retrieval operations"""
    query: Optional[str] = None
    retrieved_count: Optional[int] = None
    vector_store: Optional[str] = None
    similarity_threshold: Optional[float] = None
    reranked: bool = False
    
    def __post_init__(self):
        super().__post_init__()
        self.span_type = "retrieval"


@dataclass
class EmbeddingSpanData(SpanData):
    """Specialized span data for embedding operations"""
    model: Optional[str] = None
    provider: Optional[str] = None
    input_count: Optional[int] = None
    embedding_dimension: Optional[int] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.span_type = "embedding"


class BaseObservabilityProvider(ABC):
    """Abstract base class for observability providers"""
    
    def __init__(self, config: ObservabilityConfig):
        self.config = config
        self._initialized = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the observability provider"""
        pass
    
    @abstractmethod
    def create_trace(self, name: str, session_id: Optional[str] = None, 
                    user_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> TraceContext:
        """Create a new trace"""
        pass
    
    @abstractmethod
    def start_span(self, context: TraceContext, span_data: SpanData) -> str:
        """Start a new span within a trace"""
        pass
    
    @abstractmethod
    def end_span(self, context: TraceContext, span_id: str, 
                output_data: Optional[Dict[str, Any]] = None,
                status: str = "success", error_message: Optional[str] = None) -> None:
        """End a span"""
        pass
    
    @abstractmethod
    def log_llm_call(self, context: TraceContext, span_data: LLMSpanData) -> str:
        """Log an LLM call"""
        pass
    
    @abstractmethod
    def log_retrieval(self, context: TraceContext, span_data: RetrievalSpanData) -> str:
        """Log a retrieval operation"""
        pass
    
    @abstractmethod
    def log_embedding(self, context: TraceContext, span_data: EmbeddingSpanData) -> str:
        """Log an embedding operation"""
        pass
    
    @abstractmethod
    def log_event(self, context: TraceContext, event_name: str, 
                 data: Optional[Dict[str, Any]] = None, level: TraceLevel = TraceLevel.INFO) -> None:
        """Log a custom event"""
        pass
    
    @abstractmethod
    def flush(self) -> None:
        """Flush any pending traces/spans"""
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the provider and cleanup resources"""
        pass
    
    @property
    def is_initialized(self) -> bool:
        """Check if provider is initialized"""
        return self._initialized
    
    def should_trace(self) -> bool:
        """Check if tracing should be performed based on sample rate"""
        import random
        return random.random() < self.config.sample_rate


class NoOpObservabilityProvider(BaseObservabilityProvider):
    """No-op implementation for when observability is disabled"""
    
    def initialize(self) -> bool:
        self._initialized = True
        return True
    
    def create_trace(self, name: str, session_id: Optional[str] = None, 
                    user_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> TraceContext:
        return TraceContext(
            trace_id=str(uuid.uuid4()),
            span_id=str(uuid.uuid4()),
            session_id=session_id,
            user_id=user_id,
            metadata=metadata or {}
        )
    
    def start_span(self, context: TraceContext, span_data: SpanData) -> str:
        return str(uuid.uuid4())
    
    def end_span(self, context: TraceContext, span_id: str, 
                output_data: Optional[Dict[str, Any]] = None,
                status: str = "success", error_message: Optional[str] = None) -> None:
        pass
    
    def log_llm_call(self, context: TraceContext, span_data: LLMSpanData) -> str:
        return str(uuid.uuid4())
    
    def log_retrieval(self, context: TraceContext, span_data: RetrievalSpanData) -> str:
        return str(uuid.uuid4())
    
    def log_embedding(self, context: TraceContext, span_data: EmbeddingSpanData) -> str:
        return str(uuid.uuid4())
    
    def log_event(self, context: TraceContext, event_name: str, 
                 data: Optional[Dict[str, Any]] = None, level: TraceLevel = TraceLevel.INFO) -> None:
        pass
    
    def flush(self) -> None:
        pass
    
    def shutdown(self) -> None:
        pass