# Environment Setup Guide

This guide details how to set up the environment for the RAG System, focusing on environment variables and their role in configuring the application.

## Overview

The RAG System utilizes environment variables for sensitive information (like API keys) and for overriding configuration file settings. This provides flexibility and security, especially across different deployment environments (development, testing, production).

## Configuration Precedence

It's important to understand the order of precedence for configuration values:

1.  **Runtime parameters** (e.g., CLI arguments, API call parameters) - Highest precedence.
2.  **Environment variables** - Override settings from configuration files.
3.  **Configuration files** (e.g., `config.yaml`, `config.production.yaml`).
4.  **Default values** - Built-in defaults in the application code - Lowest precedence.

## Using Environment Variables

Environment variables can be set directly in your shell, in a `.env` file, or through your deployment platform's mechanisms (e.g., Docker Compose, Kubernetes Secrets).

### `.env` File (Recommended for Local Development)

For local development, create a `.env` file in the root directory of your project. This file is automatically loaded by the application (if configured to do so) and allows you to manage your environment variables easily without setting them globally.

1.  **Create `.env` from example:**
    ```bash
cp .env.example .env
    ```

2.  **Edit `.env`:** Open the `.env` file and fill in your actual values.

    ```ini
# Example environment variables for the RAG Engine

# LangChain API Key (if using LangChain services)
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your-langchain-api-key-here

# Google Gemini API Key (required for Google LLM/Embedding providers)
GOOGLE_API_KEY=your-google-api-key-here

# Optional: OpenAI API Key (if using OpenAI LLM/Embedding providers)
OPENAI_API_KEY=your-openai-api-key-here

# Optional: Anthropic API Key (if using Anthropic LLM providers)
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Optional: Cohere API Key (if using Cohere reranking)
COHERE_API_KEY=your-cohere-api-key-here

# Optional: Pinecone API Key and Environment (if using Pinecone vector store)
PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_ENVIRONMENT=us-west1-gcp # e.g., us-west1-gcp

# Optional: Weaviate URL and API Key (if using Weaviate vector store)
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=your-weaviate-api-key-here

# Core Configuration
ENVIRONMENT=development # Set to production, testing, etc.
LOG_LEVEL=INFO          # DEBUG, INFO, WARNING, ERROR
API_HOST=0.0.0.0
API_PORT=8000
HEALTH_PORT=8089
WORKERS=4

# Security
API_KEY=your_api_key_here # Main API authentication key
JWT_SECRET=your_jwt_secret # JWT signing secret
CORS_ORIGINS=* # CORS allowed origins (e.g., http://localhost:3000, https://yourdomain.com)

# Database and Cache
CHROMA_HOST=localhost
CHROMA_PORT=8001
CHROMA_PERSIST_DIRECTORY=./chroma_db
REDIS_URL=redis://localhost:6379
CACHE_TTL=3600
ENABLE_CACHING=true

# Performance
MAX_CONCURRENT_REQUESTS=100
REQUEST_TIMEOUT=300
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=5
MAX_MEMORY_USAGE=4GB
ENABLE_GPU=false
BATCH_SIZE=32

# Monitoring
ENABLE_MONITORING=true
METRICS_PORT=8089
PROMETHEUS_ENABLED=true
HEALTH_CHECK_INTERVAL=30
LOG_FORMAT=json
LOG_FILE=/app/logs/rag-engine.log
ENABLE_REQUEST_LOGGING=true

# Optional: Set user agent for requests
USER_AGENT=rag-engine/1.0

# Optional: Default LLM model and temperature
DEFAULT_LLM_MODEL=gemini-1.5-flash
DEFAULT_TEMPERATURE=0.1
    ```

### Setting in Shell (Temporary)

You can set environment variables directly in your shell. These will only persist for the current session or terminal window.

```bash
export GOOGLE_API_KEY="your-google-api-key-here"
export ENVIRONMENT="production"
rag serve
```

### Docker and Kubernetes

When deploying with Docker or Kubernetes, environment variables are typically managed through their respective configuration mechanisms:

*   **Docker Compose**: Variables defined in your `.env` file are automatically picked up, or you can specify them directly in the `environment` section of your `docker-compose.yml`.
*   **Kubernetes**: Use `ConfigMaps` for non-sensitive configuration and `Secrets` for sensitive data like API keys. These can then be injected into your pod definitions.

Refer to the [Docker Deployment Guide](../deployment/docker.md) and [Kubernetes Deployment Guide](../deployment/kubernetes.md) for more details on managing environment variables in containerized environments.

## Key Environment Variables

Here's a breakdown of important environment variables and their purpose:

### Core Settings

*   `ENVIRONMENT`: Sets the application environment (`development`, `testing`, `production`). Influences logging, debugging, and other environment-specific behaviors.
*   `LOG_LEVEL`: Controls the verbosity of application logs (`DEBUG`, `INFO`, `WARNING`, `ERROR`).
*   `API_HOST`, `API_PORT`: Define the host and port for the RAG API server.
*   `WORKERS`: Number of worker processes for the API server (e.g., Gunicorn workers).

### API Keys and Credentials

*   `GOOGLE_API_KEY`: Your API key for Google AI services (Gemini models, Google Embeddings).
*   `OPENAI_API_KEY`: Your API key for OpenAI services (GPT models, OpenAI Embeddings).
*   `ANTHROPIC_API_KEY`: Your API key for Anthropic models.
*   `COHERE_API_KEY`: Your API key for Cohere services (e.g., reranking).
*   `PINECONE_API_KEY`, `PINECONE_ENVIRONMENT`: Credentials for Pinecone vector database.
*   `WEAVIATE_URL`, `WEAVIATE_API_KEY`: Connection details for Weaviate vector database.
*   `API_KEY`: A general API key for authenticating requests to your RAG Engine API.
*   `JWT_SECRET`: Secret key used for signing JSON Web Tokens (JWT) if JWT authentication is enabled.

### Database and Cache Configuration

*   `CHROMA_HOST`, `CHROMA_PORT`, `CHROMA_PERSIST_DIRECTORY`: Configuration for the ChromaDB vector store.
*   `REDIS_URL`: Connection string for the Redis cache.
*   `CACHE_TTL`, `ENABLE_CACHING`: Control caching behavior.

### Performance Settings

*   `MAX_CONCURRENT_REQUESTS`, `REQUEST_TIMEOUT`: Control concurrency and request timeouts.
*   `CHUNK_SIZE`, `CHUNK_OVERLAP`: Parameters for document chunking during indexing.
*   `RETRIEVAL_K`: Default number of documents to retrieve during a query.
*   `MAX_MEMORY_USAGE`, `ENABLE_GPU`, `BATCH_SIZE`: Resource and processing optimization settings.

### Monitoring and Observability

*   `ENABLE_MONITORING`, `METRICS_PORT`, `PROMETHEUS_ENABLED`: Control metrics collection and exposure.
*   `HEALTH_CHECK_INTERVAL`: Frequency of internal health checks.
*   `LOG_FORMAT`, `LOG_FILE`, `ENABLE_REQUEST_LOGGING`: Logging configuration.

## Best Practices

*   **Never commit secrets**: Do not commit your `.env` file or any files containing sensitive API keys directly to version control. Use `.gitignore`.
*   **Use environment-specific variables**: Leverage the `ENVIRONMENT` variable to load different configurations for different deployment stages.
*   **Prioritize secrets**: Always use environment variables or dedicated secret management solutions for API keys and sensitive data, as they override values in configuration files.

For more detailed configuration options, refer to the [Configuration Reference](../configuration-reference.md).
