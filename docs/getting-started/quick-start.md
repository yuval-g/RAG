# Quick Start Guide

This guide will help you get started with the RAG System quickly by walking you through basic operations.

## Initialize the System

After installation, you can check the system status:

```bash
rag status
```

## Index Documents

Before querying, you need to index some documents:

### Index Local Files

```bash
# Index a single file
rag index files /path/to/document.txt

# Index all text files in a directory recursively
rag index files --recursive --pattern "*.txt" /path/to/documents/
```

### Index Web Content

```bash
# Index content from URLs
rag index web https://example.com/page1 https://example.com/page2
```

## Query the System

Once documents are indexed, you can query the system:

```bash
rag query "What is retrieval-augmented generation?"
```

To get more results:
```bash
rag query --k 10 "Explain the RAG architecture"
```

For JSON output (useful for scripting):
```bash
rag query --output-format json "What are the benefits of RAG?" > response.json
```

## Start the API Server

To use the RAG system via REST API:

```bash
rag serve --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

## Evaluate the System

You can evaluate the system's performance using test cases:

```bash
rag evaluate test_cases.json
```

## Configuration

The system can be configured using a YAML configuration file:

```bash
rag --config config.yaml status
```

See the [Configuration Reference](../configuration/configuration-reference.md) for details on available options.

## Next Steps

- [First Steps Guide](first-steps.md) - More detailed walkthrough of system features
- [CLI Reference](../api-reference/cli-reference.md) - Complete reference for all CLI commands
- [API Documentation](../api-reference/rest-api.md) - REST API endpoints and usage