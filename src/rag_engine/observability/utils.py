"""
Utility functions for observability
"""

from ..core.config import PipelineConfig
from .interfaces import ObservabilityConfig, ObservabilityProvider, TraceLevel


def create_observability_config(pipeline_config: PipelineConfig) -> ObservabilityConfig:
    """Create ObservabilityConfig from PipelineConfig"""
    
    # Map provider string to enum
    provider_map = {
        "langfuse": ObservabilityProvider.LANGFUSE,
        "phoenix": ObservabilityProvider.PHOENIX,
        "disabled": ObservabilityProvider.DISABLED
    }
    
    provider = provider_map.get(
        pipeline_config.observability_provider.lower(), 
        ObservabilityProvider.DISABLED
    )
    
    # Map log level to trace level
    trace_level_map = {
        "DEBUG": TraceLevel.DEBUG,
        "INFO": TraceLevel.INFO,
        "WARNING": TraceLevel.WARNING,
        "ERROR": TraceLevel.ERROR
    }
    
    trace_level = trace_level_map.get(
        pipeline_config.log_level.upper(),
        TraceLevel.INFO
    )
    
    return ObservabilityConfig(
        provider=provider,
        enabled=pipeline_config.observability_enabled,
        langfuse_secret_key=pipeline_config.langfuse_secret_key,
        langfuse_public_key=pipeline_config.langfuse_public_key,
        langfuse_host=pipeline_config.langfuse_host,
        phoenix_endpoint=pipeline_config.phoenix_endpoint,
        phoenix_api_key=pipeline_config.phoenix_api_key,
        trace_level=trace_level,
        sample_rate=pipeline_config.observability_sample_rate,
        trace_llm_calls=pipeline_config.trace_llm_calls,
        trace_retrieval=pipeline_config.trace_retrieval,
        trace_embeddings=pipeline_config.trace_embeddings,
        trace_evaluation=pipeline_config.trace_evaluation,
        capture_inputs=pipeline_config.capture_inputs,
        capture_outputs=pipeline_config.capture_outputs
    )