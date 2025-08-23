# Observability in RAG Engine

The RAG engine includes comprehensive observability support to help you monitor, debug, and optimize your RAG applications. The system is designed with a pluggable architecture that currently supports Langfuse with planned support for Arize Phoenix.

## Supported Providers

### Langfuse
[Langfuse](https://langfuse.com/) is an open-source LLM engineering platform that provides tracing, evaluation, and analytics for LLM applications.

**Features:**
- Detailed trace visualization
- Token usage and cost tracking
- Performance analytics
- Evaluation metrics
- Session management

### Arize Phoenix (Coming Soon)
[Arize Phoenix](https://phoenix.arize.com/) is an open-source observability platform for ML and LLM applications.

**Planned Features:**
- Real-time monitoring
- Drift detection
- Performance analysis
- Data quality monitoring

## Quick Start

### 1. Install Dependencies

Langfuse is included in the project dependencies. If you need to install it separately:

```bash
uv add langfuse
```

### 2. Configure Environment Variables

Set up your Langfuse credentials:

```bash
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_HOST="https://cloud.langfuse.com"  # Optional
export RAG_OBSERVABILITY_ENABLED=true
export RAG_OBSERVABILITY_PROVIDER=langfuse
```

### 3. Basic Usage

```python
from rag_engine.core.config import ConfigurationManager
from rag_engine.observability import ObservabilityManager, create_observability_config

# Load configuration
config_manager = ConfigurationManager()
config = config_manager.load_config()

# Create observability manager
obs_config = create_observability_config(config)
obs_manager = ObservabilityManager(obs_config)
obs_manager.initialize()

# Create a trace
with obs_manager.trace("rag_query", session_id="user_123") as context:
    # Your RAG operations here
    obs_manager.log_llm_call(
        context=context,
        name="generate_answer",
        model="gemini-2.0-flash-lite",
        provider="google",
        input_data={"prompt": "What is RAG?"},
        output_data={"response": "RAG stands for..."},
        prompt_tokens=50,
        completion_tokens=100
    )
    
    obs_manager.log_retrieval(
        context=context,
        name="document_search",
        query="What is RAG?",
        retrieved_count=5,
        vector_store="chroma"
    )
```

## Configuration Options

### Core Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `observability_enabled` | bool | false | Enable/disable observability |
| `observability_provider` | str | "disabled" | Provider to use ("langfuse", "phoenix", "disabled") |
| `observability_sample_rate` | float | 1.0 | Sampling rate (0.0-1.0) |

### Langfuse Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `langfuse_secret_key` | str | None | Langfuse secret key |
| `langfuse_public_key` | str | None | Langfuse public key |
| `langfuse_host` | str | None | Langfuse host URL |

### Tracing Features

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `trace_llm_calls` | bool | true | Trace LLM generation calls |
| `trace_retrieval` | bool | true | Trace document retrieval |
| `trace_embeddings` | bool | true | Trace embedding generation |
| `trace_evaluation` | bool | true | Trace evaluation runs |

### Data Capture

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `capture_inputs` | bool | true | Capture input data |
| `capture_outputs` | bool | true | Capture output data |
| `capture_metadata` | bool | true | Capture metadata |

## Using Decorators

The observability system provides decorators for automatic instrumentation:

```python
from rag_engine.observability.decorators import trace_llm_call, trace_retrieval

@trace_llm_call(obs_manager, model="gemini-2.0-flash-lite", provider="google")
def generate_response(prompt: str) -> str:
    # Your LLM call logic
    return response

@trace_retrieval(obs_manager, vector_store="chroma")
def search_documents(query: str) -> List[Document]:
    # Your retrieval logic
    return documents
```

## Manual Instrumentation

For more control, use manual instrumentation:

```python
# Create trace context
context = obs_manager.create_trace("custom_operation")

# Start a span
with obs_manager.span(context, "processing", "data_processing") as span_id:
    # Your processing logic
    result = process_data()
    
    # Log custom events
    obs_manager.log_event(
        context=context,
        event_name="processing_complete",
        data={"processed_items": len(result)}
    )
```

## Best Practices

### 1. Use Meaningful Names
```python
# Good
context = obs_manager.create_trace("user_query_processing")
obs_manager.log_llm_call(context, "answer_generation", ...)

# Avoid
context = obs_manager.create_trace("trace1")
obs_manager.log_llm_call(context, "llm", ...)
```

### 2. Include Session and User Context
```python
context = obs_manager.create_trace(
    "rag_query",
    session_id=user_session_id,
    user_id=user_id,
    metadata={"query_type": "factual", "domain": "science"}
)
```

### 3. Handle Errors Gracefully
```python
try:
    result = risky_operation()
    obs_manager.log_event(context, "operation_success")
except Exception as e:
    obs_manager.log_event(
        context, 
        "operation_error", 
        data={"error": str(e)},
        level=TraceLevel.ERROR
    )
    raise
```

### 4. Use Sampling for High-Volume Applications
```python
# Set sample rate to 10% for production
observability_sample_rate: 0.1
```

### 5. Flush Regularly
```python
# Ensure data is sent to the provider
obs_manager.flush()

# Clean shutdown
obs_manager.shutdown()
```

## Environment Variables

All configuration options can be set via environment variables:

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

## Troubleshooting

### Common Issues

1. **Langfuse Connection Failed**
   - Verify your API keys are correct
   - Check network connectivity to Langfuse host
   - Ensure the host URL is correct

2. **No Traces Appearing**
   - Check if observability is enabled
   - Verify sample rate is > 0
   - Call `obs_manager.flush()` to force upload

3. **High Memory Usage**
   - Reduce sample rate
   - Disable input/output capture for large payloads
   - Increase flush interval

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.getLogger('rag_engine.observability').setLevel(logging.DEBUG)
```

## Extending with New Providers

To add support for a new observability provider:

1. Create a new provider class inheriting from `BaseObservabilityProvider`
2. Implement all abstract methods
3. Add the provider to the `ObservabilityManager`
4. Update configuration enums and mappings

See `phoenix_provider.py` for a template implementation.