# First Steps Guide

This guide provides a more detailed walkthrough of the RAG System's features and capabilities.

## System Architecture Overview

The RAG System is built with a modular architecture that allows for flexibility and extensibility:

1. **Core Engine** - The main RAG engine that orchestrates the retrieval and generation process
2. **Providers** - Pluggable components for LLMs, embeddings, and vector stores
3. **Indexing Pipeline** - Handles document processing and indexing
4. **Query Pipeline** - Handles query processing and response generation
5. **API Layer** - REST API and CLI interfaces
6. **Evaluation Framework** - Tools for assessing system performance

## Document Indexing

### Supported Document Types

The system supports various document formats:
- Plain text files (.txt)
- Markdown files (.md)
- JSON files (.json)
- YAML files (.yaml, .yml)
- Python source files (.py)

### Indexing Options

When indexing documents, you can use several options:

```bash
# Clear existing documents before indexing
rag index files --clear /path/to/new/documents/

# Index with custom file patterns
rag index files --pattern "*.md" --recursive /docs/

# Index web content with specific depth
rag index web --max-depth 2 https://example.com/
```

## Querying Features

### Basic Queries

```bash
# Simple query
rag query "What is machine learning?"

# Query with source documents
rag query --include-sources "Explain neural networks"
```

### Advanced Query Options

```bash
# Adjust the number of retrieved documents
rag query --k 15 "Complex question requiring more context"

# Get structured output
rag query --output-format json "Question for automation" > result.json
```

## System Management

### Checking System Status

```bash
# View system information
rag status

# Detailed configuration
rag config show

# Validate configuration file
rag config validate config.yaml
```

### Index Management

```bash
# View current document count
rag status

# Clear all indexed documents
rag index clear --confirm
```

## API Usage

### Starting the Server

```bash
# Start with default settings
rag serve

# Start with custom host and port
rag serve --host 127.0.0.1 --port 8080

# Enable auto-reload for development
rag serve --reload
```

### API Endpoints

Once the server is running, you can access these endpoints:
- `GET /health` - System health check
- `POST /query` - Query the RAG system
- `POST /index/files` - Index local files
- `POST /index/web` - Index web content
- `DELETE /index` - Clear all documents

## Evaluation and Testing

### Creating Test Cases

Create a JSON file with test cases:

```json
{
  "test_cases": [
    {
      "question": "What is retrieval-augmented generation?",
      "expected_answer": "RAG is a technique that combines retrieval and generation for improved AI responses.",
      "context": [
        {
          "content": "Retrieval-augmented generation (RAG) is a technique...",
          "metadata": {"source": "rag-intro.md"},
          "doc_id": "doc1"
        }
      ]
    }
  ]
}
```

### Running Evaluations

```bash
# Evaluate with custom test cases
rag evaluate test_cases.json

# Save results to file
rag evaluate --output results.json test_cases.json
```

## Next Steps

- [User Guides](../user-guides/basic-usage.md) - Comprehensive usage documentation
- [Configuration Reference](../configuration/configuration-reference.md) - Detailed configuration options
- [API Reference](../api-reference/rest-api.md) - Complete API documentation