# Basic Usage

This guide covers the fundamental operations you can perform with the RAG System.

## System Status and Information

Before performing any operations, it's useful to check the system status:

```bash
# Check overall system status
rag status

# View current configuration
rag config show

# View configuration in JSON format
rag config show --format json
```

The status command provides information about:
- System version
- Active providers (LLM, embedding, vector store)
- Indexed document count
- System component status

## Document Indexing

### Indexing Local Files

To index documents from your local filesystem:

```bash
# Index a single file
rag index files /path/to/document.txt

# Index multiple files
rag index files /path/to/doc1.txt /path/to/doc2.md

# Index all text files in a directory recursively
rag index files --recursive --pattern "*.txt" /path/to/documents/

# Clear existing documents before indexing
rag index files --clear /path/to/new/documents/
```

Supported file formats:
- Plain text (.txt)
- Markdown (.md)
- JSON (.json)
- YAML (.yaml, .yml)
- Python source files (.py)

### Indexing Web Content

To index content from websites:

```bash
# Index a single web page
rag index web https://example.com/page

# Index multiple URLs
rag index web https://example.com/page1 https://example.com/page2

# Index with crawling depth
rag index web --max-depth 2 https://example.com/

# Clear existing documents before indexing
rag index web --clear https://example.com/new-content
```

### Managing the Index

```bash
# View current document count
rag status

# Clear all indexed documents
rag index clear

# Clear with confirmation skip
rag index clear --confirm
```

## Querying Documents

### Basic Queries

```bash
# Simple query
rag query "What is machine learning?"

# Query with more retrieved documents
rag query --k 10 "Explain neural networks"

# Query without showing source documents
rag query --no-include-sources "What is deep learning?"
```

### Output Formats

```bash
# Default text output with rich formatting
rag query "What is RAG?"

# JSON output for scripting
rag query --output-format json "What is RAG?" > response.json

# Pretty-printed JSON
rag query --output-format json "What is RAG?" | jq .
```

## API Server

### Starting the Server

```bash
# Start with default settings (0.0.0.0:8000)
rag serve

# Start with custom host and port
rag serve --host 127.0.0.1 --port 8080

# Enable auto-reload for development
rag serve --reload

# Start with verbose logging
rag serve --verbose
```

### Server Endpoints

Once running, the server provides these endpoints:
- `GET /health` - Health check endpoint
- `POST /query` - Query the RAG system
- `POST /index/files` - Index local files
- `POST /index/web` - Index web content
- `DELETE /index` - Clear all documents

## System Configuration

### Viewing Configuration

```bash
# View current configuration in YAML format
rag config show

# View in JSON format
rag config show --format json
```

### Validating Configuration Files

```bash
# Validate a configuration file
rag config validate config.yaml
```

## Evaluation

### Running Evaluations

```bash
# Run evaluation with test cases
rag evaluate test_cases.json

# Save results to file
rag evaluate --output results.json test_cases.json

# Get results in JSON format
rag evaluate --output-format json test_cases.json
```

### Test Case Format

Create a JSON file with your test cases:

```json
{
  "test_cases": [
    {
      "question": "What is retrieval-augmented generation?",
      "expected_answer": "A technique that combines information retrieval with language model generation.",
      "context": [
        {
          "content": "Retrieval-augmented generation (RAG) combines retrieval and generation...",
          "metadata": {"source": "rag-intro.md"},
          "doc_id": "doc1"
        }
      ],
      "metadata": {
        "category": "definition",
        "difficulty": "easy"
      }
    }
  ]
}
```

## Advanced Features

For more advanced usage, see the [Advanced Features Guide](advanced-features.md).