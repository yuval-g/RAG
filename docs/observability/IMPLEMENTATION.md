# Observability Implementation

This document describes the observability system implementation for the RAG engine.

## Architecture Overview

The observability system follows a pluggable architecture that allows for multiple observability providers:

```
ObservabilityManager
├── BaseObservabilityProvider (Interface)
├── LangfuseProvider (Implementation)
├── PhoenixProvider (Placeholder)
└── NoOpObservabilityProvider (Fallback)
```

## Key Components

### 1. Interfaces (`interfaces.py`)
- `BaseObservabilityProvider`: Abstract base class for all providers
- `ObservabilityConfig`: Configuration data class
- `TraceContext`: Context information for traces
- `SpanData`: Base data structure for spans
- `LLMSpanData`, `RetrievalSpanData`, `EmbeddingSpanData`: Specialized span types

### 2. Providers
- **LangfuseProvider** (`langfuse_provider.py`): Full implementation for Langfuse
- **PhoenixProvider** (`phoenix_provider.py`): Placeholder for future Arize Phoenix support
- **NoOpObservabilityProvider**: Fallback when observability is disabled

### 3. Manager (`manager.py`)
- `ObservabilityManager`: Main interface for the observability system
- Handles provider initialization and fallback logic
- Provides high-level methods for tracing operations

### 4. Configuration Integration
- Extended `PipelineConfig` with observability settings
- Environment variable support
- Utility functions for config conversion

### 5. Decorators (`decorators.py`)
- `@trace_llm_call`: Automatic LLM call tracing
- `@trace_retrieval`: Automatic retrieval operation tracing
- `@trace_embedding`: Automatic embedding operation tracing
- `@trace_function`: Generic function tracing

## Usage Patterns

### Manual Instrumentation
```python
# Create trace
with obs_manager.trace("operation_name") as context:
    # Log LLM call
    obs_manager.log_llm_call(
        context=context,
        name="generate_answer",
        model="gemini-2.0-flash-lite",
        provider="google",
        prompt_tokens=100,
        completion_tokens=50
    )
    
    # Log retrieval
    obs_manager.log_retrieval(
        context=context,
        name="document_search",
        query="user query",
        retrieved_count=5,
        vector_store="chroma"
    )
```

### Context Managers
```python
# Automatic span management
with obs_manager.span(context, "processing", "data") as span_id:
    # Your processing logic
    result = process_data()
```

### Decorator-based (Future Enhancement)
```python
@trace_llm_call(obs_manager, model="gpt-4", provider="openai")
def generate_response(prompt: str) -> str:
    # Your LLM logic
    return response
```

## Configuration

### Environment Variables
```bash
# Core settings
export RAG_OBSERVABILITY_ENABLED=true
export RAG_OBSERVABILITY_PROVIDER=langfuse
export RAG_OBSERVABILITY_SAMPLE_RATE=1.0

# Langfuse
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_HOST=https://cloud.langfuse.com

# Feature flags
export RAG_TRACE_LLM_CALLS=true
export RAG_TRACE_RETRIEVAL=true
export RAG_TRACE_EMBEDDINGS=true
export RAG_CAPTURE_INPUTS=true
export RAG_CAPTURE_OUTPUTS=true
```

### YAML Configuration
```yaml
observability_enabled: true
observability_provider: "langfuse"
observability_sample_rate: 1.0

langfuse_secret_key: "sk-lf-..."
langfuse_public_key: "pk-lf-..."
langfuse_host: "https://cloud.langfuse.com"

trace_llm_calls: true
trace_retrieval: true
trace_embeddings: true
capture_inputs: true
capture_outputs: true
```

## Provider Implementation Guide

To add a new observability provider:

1. **Create Provider Class**
   ```python
   class NewProvider(BaseObservabilityProvider):
       def initialize(self) -> bool:
           # Initialize your provider
           pass
       
       def create_trace(self, name: str, ...) -> TraceContext:
           # Create trace in your system
           pass
       
       # Implement all abstract methods
   ```

2. **Update Configuration**
   ```python
   class ObservabilityProvider(str, Enum):
       LANGFUSE = "langfuse"
       PHOENIX = "phoenix"
       NEW_PROVIDER = "new_provider"  # Add here
   ```

3. **Update Manager**
   ```python
   # In ObservabilityManager.initialize()
   elif self.config.provider == ObservabilityProvider.NEW_PROVIDER:
       self._provider = NewProvider(self.config)
   ```

4. **Add Configuration Fields**
   ```python
   # In ObservabilityConfig
   new_provider_api_key: Optional[str] = None
   new_provider_endpoint: Optional[str] = None
   ```

## Testing

The system includes comprehensive tests:

- **Unit Tests**: `tests/observability/test_observability_basic.py`
- **Integration Example**: `examples/observability_integration.py`

Run tests:
```bash
uv run python -m pytest tests/observability/ -v
```

## Future Enhancements

1. **Arize Phoenix Integration**
   - Complete the `PhoenixProvider` implementation
   - Add Phoenix-specific configuration options
   - Test integration with Phoenix server

2. **Enhanced Decorators**
   - Improve decorator robustness
   - Add more extraction functions for token counting
   - Support for async functions

3. **Metrics and Analytics**
   - Add built-in metrics collection
   - Performance monitoring
   - Cost tracking and optimization

4. **Batch Processing**
   - Implement batched trace uploads
   - Configurable flush intervals
   - Memory-efficient trace buffering

5. **Error Handling**
   - Retry logic for failed uploads
   - Circuit breaker pattern
   - Graceful degradation

## Dependencies

- **Langfuse**: `langfuse>=2.0.0` (optional, for Langfuse provider)
- **Core**: No additional dependencies for base functionality

## Performance Considerations

- **Sampling**: Use `observability_sample_rate` to reduce overhead
- **Data Capture**: Disable `capture_inputs`/`capture_outputs` for large payloads
- **Async**: Future versions will support async operations
- **Memory**: Traces are flushed regularly to prevent memory buildup