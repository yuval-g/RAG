# Installation Guide

This guide will help you install and set up the RAG System on your local machine or server.

## Prerequisites

- Python 3.13 or higher
- pip or uv package manager
- Git (for cloning the repository)
- Docker (optional, for containerized deployment)

## Installation Methods

### Method 1: Using pip (Recommended for development)

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd rag-system
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

### Method 2: Using uv (Faster installation)

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd rag-system
   ```

2. Install using uv:
   ```bash
   uv sync
   ```

### Method 3: Docker (Recommended for production)

See the [Docker Deployment Guide](../deployment/docker.md) for instructions on running the RAG System in Docker containers.

## Environment Variables

Create a `.env` file in the project root based on the `.env.example` file:

```bash
cp .env.example .env
```

Configure the necessary environment variables:
- `OPENAI_API_KEY` - Your OpenAI API key (if using OpenAI)
- `GOOGLE_API_KEY` - Your Google API key (if using Google models)
- `VECTOR_STORE_TYPE` - The vector store to use (chroma, pinecone, weaviate)

## Verify Installation

After installation, verify that the RAG CLI is working:

```bash
rag --help
```

You should see the available commands and options.

## Next Steps

- [Quick Start Guide](quick-start.md) - Learn how to use the basic features
- [Configuration Guide](../configuration/configuration-reference.md) - Configure the system for your needs