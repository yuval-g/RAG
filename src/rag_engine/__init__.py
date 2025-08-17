"""
Production RAG System - A modular, extensible RAG platform
"""

from .core import (
    RAGEngine, PipelineConfig, ConfigurationManager,
    Document, ProcessedQuery, RAGResponse, EvaluationResult,
    TestCase, RoutingDecision, RouteConfig
)

__version__ = "0.1.0"

__all__ = [
    'RAGEngine',
    'PipelineConfig', 
    'ConfigurationManager',
    'Document',
    'ProcessedQuery',
    'RAGResponse',
    'EvaluationResult',
    'TestCase',
    'RoutingDecision',
    'RouteConfig',
]