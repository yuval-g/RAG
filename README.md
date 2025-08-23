# RAG Engine

A production-ready, modular, and extensible Retrieval-Augmented Generation (RAG) platform built with Python.

## Overview

The RAG Engine is a comprehensive platform for implementing RAG workflows with support for multiple LLM providers, vector stores, and retrieval strategies. It provides a complete solution for building AI applications that combine the power of large language models with domain-specific knowledge retrieval.

## Key Features

- **Multi-Provider Support**: Integrations with Google, OpenAI, and local LLM providers
- **Flexible Vector Stores**: Support for Chroma, Pinecone, and Weaviate
- **Advanced Retrieval**: Multiple retrieval strategies including vector, keyword, and hybrid approaches
- **Query Enhancement**: Techniques like Multi-Query, RAG-Fusion, and HyDE
- **Intelligent Routing**: Logical and semantic query routing
- **Evaluation Framework**: Built-in evaluation with RAGAS and custom metrics
- **Production Ready**: Docker support, monitoring, and health checks
- **Extensible Architecture**: Modular design for easy customization

## Documentation

Comprehensive documentation is available in the [docs](./docs) directory:

- [Getting Started Guide](./docs/getting-started/)
- [Project Map](./docs/PROJECT_MAP.md) - Visual overview of the project structure
- [API Reference](./docs/api-reference/)
- [Configuration Guide](./docs/configuration/)
- [Deployment Instructions](./docs/deployment/)

## Quick Start

1. Install dependencies:
   ```bash
   pip install -e .
   ```

2. Configure your environment:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. Run the CLI:
   ```bash
   python rag_cli.py
   ```

4. Or start the API server:
   ```bash
   python -m src.rag_engine.api.main
   ```

## Development

To run in development mode with all services:

```bash
docker-compose up
```

This will start the RAG Engine API, Chroma vector database, Redis, Prometheus, and Grafana.

## Project Structure

See the [Project Map](./docs/PROJECT_MAP.md) for a detailed overview of the codebase structure.