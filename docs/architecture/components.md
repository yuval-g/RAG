# System Components

This document details the individual components that make up the RAG System.

## Core Engine

The `RAGEngine` is the central orchestrator of the system:

### Responsibilities
- Managing the complete RAG pipeline
- Coordinating between different components
- Handling system configuration
- Maintaining system state

### Key Methods
- `add_documents()` - Index new documents
- `query()` - Process user queries
- `evaluate()` - Run system evaluations
- `get_system_info()` - Retrieve system status

## Providers

Providers are abstract interfaces for external services with multiple implementations.

### LLM Providers
- **OpenAIProvider**: Connects to OpenAI models
- **GoogleProvider**: Connects to Google Gemini models
- **AnthropicProvider**: (Planned) Connects to Anthropic models

### Embedding Providers
- **OpenAIEmbeddingProvider**: OpenAI text embeddings
- **GoogleEmbeddingProvider**: Google text embeddings
- **HuggingFaceEmbeddingProvider**: (Planned) HuggingFace embeddings

### Vector Store Providers
- **ChromaProvider**: Local ChromaDB vector store
- **PineconeProvider**: Pinecone cloud vector database
- **WeaviateProvider**: Weaviate vector database

## Pipelines

Specialized processing workflows that handle specific tasks.

### Indexing Pipeline
Handles document ingestion and processing:
1. Document loading from various sources
2. Text chunking and processing
3. Embedding generation
4. Storage in vector databases

### Query Pipeline
Processes user queries:
1. Query parsing and enhancement
2. Embedding generation
3. Document retrieval
4. Response generation

### Retrieval Pipeline
Manages document retrieval:
1. Similarity search in vector stores
2. Result filtering and ranking
3. Metadata enrichment

## Interfaces

Different ways to interact with the system.

### CLI (Command-Line Interface)
Built with Click, provides direct access to all system features:
- Document indexing
- Query processing
- System management
- Evaluation tools

### REST API
FastAPI-based HTTP interface:
- Health checks
- Query endpoints
- Indexing endpoints
- Management endpoints

### Python SDK
Programmatic access for integration into other Python applications.

## Core Models

Dataclasses that represent key system entities:

### Document
Represents a piece of content to be indexed:
- `content`: The actual text content
- `metadata`: Additional information about the document
- `doc_id`: Unique identifier

### RAGResponse
Represents a response to a user query:
- `answer`: The generated response
- `confidence_score`: Confidence in the answer
- `source_documents`: Documents used to generate the response
- `processing_time`: Time taken to process the query

### PipelineConfig
System configuration:
- LLM and embedding settings
- Vector store configuration
- Processing parameters

## Utilities

Helper components that support the main functionality:

### Configuration Manager
Handles loading and validating system configuration from files or environment variables.

### Health API
Provides system health checks and monitoring endpoints.

### Monitoring
Observability features including logging and metrics collection.

### Resilience
Error handling, retry mechanisms, and fault tolerance features.

For information on how these components interact, see the [Data Flow Documentation](data-flow.md).