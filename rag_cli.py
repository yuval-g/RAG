#!/usr/bin/env python3
"""
Main entry point for the RAG CLI
"""
# Ensure environment variables are loaded early
from src.rag_engine.core.env_loader import ensure_env_loaded
ensure_env_loaded()

from src.rag_engine.cli.main import cli

if __name__ == '__main__':
    cli()