#!/usr/bin/env python3
"""
Main entry point for the RAG CLI
"""
from dotenv import load_dotenv

load_dotenv()

from src.rag_engine.cli.main import cli

if __name__ == '__main__':
    cli()