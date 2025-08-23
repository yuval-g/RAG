"""
Decorators for automatic observability instrumentation
"""

import functools
from typing import Callable, Any, Optional, Dict
from datetime import datetime
import inspect

from .interfaces import TraceContext, SpanData, LLMSpanData, RetrievalSpanData, EmbeddingSpanData
from .manager import ObservabilityManager


def trace_llm_call(
    observability_manager: ObservabilityManager,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    extract_tokens: Optional[Callable] = None,
    extract_cost: Optional[Callable] = None
):
    """Decorator to automatically trace LLM calls"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not observability_manager.is_enabled or not observability_manager.config.trace_llm_calls:
                return func(*args, **kwargs)
            
            # Get trace context from kwargs or create a new one
            context = kwargs.get('trace_context')
            if not context:
                context = observability_manager.create_trace(f"llm_call_{func.__name__}")
            
            # Extract input data
            input_data = None
            if observability_manager.config.capture_inputs:
                input_data = {
                    "function": func.__name__,
                    "args": str(args)[:1000],  # Truncate long args
                    "kwargs": {k: str(v)[:1000] for k, v in kwargs.items() if k != 'trace_context'}
                }
            
            start_time = datetime.now()
            
            try:
                result = func(*args, **kwargs)
                end_time = datetime.now()
                
                # Extract output data
                output_data = None
                if observability_manager.config.capture_outputs:
                    output_data = {"result": str(result)[:1000]}
                
                # Extract token usage if function provided
                prompt_tokens = None
                completion_tokens = None
                total_tokens = None
                cost = None
                
                if extract_tokens:
                    try:
                        token_info = extract_tokens(result)
                        prompt_tokens = token_info.get('prompt_tokens')
                        completion_tokens = token_info.get('completion_tokens')
                        total_tokens = token_info.get('total_tokens')
                    except Exception:
                        pass
                
                if extract_cost:
                    try:
                        cost = extract_cost(result)
                    except Exception:
                        pass
                
                # Log the LLM call
                observability_manager.log_llm_call(
                    context=context,
                    name=f"llm_call_{func.__name__}",
                    model=model or "unknown",
                    provider=provider or "unknown",
                    input_data=input_data,
                    output_data=output_data,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cost=cost
                )
                
                return result
                
            except Exception as e:
                # Log error
                observability_manager.log_event(
                    context=context,
                    event_name=f"llm_call_error_{func.__name__}",
                    data={"error": str(e), "function": func.__name__}
                )
                raise
        
        return wrapper
    return decorator


def trace_retrieval(
    observability_manager: ObservabilityManager,
    vector_store: Optional[str] = None,
    extract_query: Optional[Callable] = None,
    extract_results: Optional[Callable] = None
):
    """Decorator to automatically trace retrieval operations"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not observability_manager.is_enabled or not observability_manager.config.trace_retrieval:
                return func(*args, **kwargs)
            
            # Get trace context from kwargs or create a new one
            context = kwargs.get('trace_context')
            if not context:
                context = observability_manager.create_trace(f"retrieval_{func.__name__}")
            
            # Extract query
            query = "unknown"
            if extract_query:
                try:
                    query = extract_query(*args, **kwargs)
                except Exception:
                    pass
            elif args:
                query = str(args[0])[:200]  # Assume first arg is query
            
            start_time = datetime.now()
            
            try:
                result = func(*args, **kwargs)
                
                # Extract result info
                retrieved_count = 0
                output_data = None
                
                if extract_results:
                    try:
                        result_info = extract_results(result)
                        retrieved_count = result_info.get('count', 0)
                        if observability_manager.config.capture_outputs:
                            output_data = result_info.get('data')
                    except Exception:
                        pass
                elif hasattr(result, '__len__'):
                    retrieved_count = len(result)
                    if observability_manager.config.capture_outputs:
                        output_data = {"results": str(result)[:1000]}
                
                # Log the retrieval
                observability_manager.log_retrieval(
                    context=context,
                    name=f"retrieval_{func.__name__}",
                    query=query,
                    retrieved_count=retrieved_count,
                    vector_store=vector_store or "unknown",
                    output_data=output_data
                )
                
                return result
                
            except Exception as e:
                # Log error
                observability_manager.log_event(
                    context=context,
                    event_name=f"retrieval_error_{func.__name__}",
                    data={"error": str(e), "function": func.__name__, "query": query}
                )
                raise
        
        return wrapper
    return decorator


def trace_embedding(
    observability_manager: ObservabilityManager,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    extract_input_count: Optional[Callable] = None
):
    """Decorator to automatically trace embedding operations"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not observability_manager.is_enabled or not observability_manager.config.trace_embeddings:
                return func(*args, **kwargs)
            
            # Get trace context from kwargs or create a new one
            context = kwargs.get('trace_context')
            if not context:
                context = observability_manager.create_trace(f"embedding_{func.__name__}")
            
            # Extract input count
            input_count = 0
            if extract_input_count:
                try:
                    input_count = extract_input_count(*args, **kwargs)
                except Exception:
                    pass
            elif args and hasattr(args[0], '__len__'):
                input_count = len(args[0])
            
            input_data = None
            if observability_manager.config.capture_inputs and args:
                input_data = {"input": str(args[0])[:500]}
            
            try:
                result = func(*args, **kwargs)
                
                # Extract embedding dimension
                embedding_dimension = None
                output_data = None
                
                if result and hasattr(result, '__len__') and len(result) > 0:
                    if hasattr(result[0], '__len__'):
                        embedding_dimension = len(result[0])
                    
                    if observability_manager.config.capture_outputs:
                        output_data = {"embedding_shape": f"{len(result)}x{embedding_dimension}"}
                
                # Log the embedding
                observability_manager.log_embedding(
                    context=context,
                    name=f"embedding_{func.__name__}",
                    model=model or "unknown",
                    provider=provider or "unknown",
                    input_count=input_count,
                    embedding_dimension=embedding_dimension,
                    input_data=input_data,
                    output_data=output_data
                )
                
                return result
                
            except Exception as e:
                # Log error
                observability_manager.log_event(
                    context=context,
                    event_name=f"embedding_error_{func.__name__}",
                    data={"error": str(e), "function": func.__name__}
                )
                raise
        
        return wrapper
    return decorator


def trace_function(
    observability_manager: ObservabilityManager,
    span_type: str = "generic",
    capture_args: bool = True,
    capture_result: bool = True
):
    """Generic decorator to trace any function"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not observability_manager.is_enabled:
                return func(*args, **kwargs)
            
            # Get trace context from kwargs or create a new one
            context = kwargs.get('trace_context')
            if not context:
                context = observability_manager.create_trace(f"{span_type}_{func.__name__}")
            
            # Prepare input data
            input_data = None
            if capture_args and observability_manager.config.capture_inputs:
                input_data = {
                    "function": func.__name__,
                    "args": str(args)[:1000],
                    "kwargs": {k: str(v)[:1000] for k, v in kwargs.items() if k != 'trace_context'}
                }
            
            with observability_manager.span(
                context=context,
                name=f"{span_type}_{func.__name__}",
                span_type=span_type,
                input_data=input_data
            ) as span_id:
                try:
                    result = func(*args, **kwargs)
                    
                    # Log result if requested
                    if capture_result and observability_manager.config.capture_outputs:
                        observability_manager.log_event(
                            context=context,
                            event_name=f"{func.__name__}_result",
                            data={"result": str(result)[:1000]}
                        )
                    
                    return result
                    
                except Exception as e:
                    observability_manager.log_event(
                        context=context,
                        event_name=f"{func.__name__}_error",
                        data={"error": str(e)}
                    )
                    raise
        
        return wrapper
    return decorator