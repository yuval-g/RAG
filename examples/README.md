# RAG Engine Examples

This directory contains example applications and use cases demonstrating how to use the RAG Engine for various scenarios.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Example Applications](#example-applications)
3. [Basic Examples](#basic-examples)
4. [Advanced Examples](#advanced-examples)
5. [Integration Examples](#integration-examples)
6. [Running Examples](#running-examples)
7. [Customization Guide](#customization-guide)

## Quick Start

### Prerequisites

1. **Install Dependencies**:
   ```bash
   cd rag-engine
   uv sync
   ```

2. **Choose Your Setup**:

   **Option A: With API Keys (Full functionality)**
   ```bash
   export GOOGLE_API_KEY="your_google_api_key"
   ```

   **Option B: Without API Keys (Perfect for testing/demos)**
   ```bash
   # No API keys needed! Use local/mock providers
   # See example_no_api_keys.py for details
   ```

   **Option C: Local LLM with Ollama**
   ```bash
   # Install Ollama: https://ollama.ai/
   ollama pull llama2
   ollama serve
   ```

3. **Start Required Services** (if using Docker):
   ```bash
   docker-compose up -d chroma redis
   ```

### Run Your First Example

```bash
# Without API keys (recommended for first try)
python examples/example_no_api_keys.py

# Simple usage example
python examples/example_simple.py

# Multi-provider LLM demo
python examples/example_multi_provider_llm.py

# With applications (requires API keys or Ollama)
python examples/applications/simple_chatbot.py
python examples/applications/document_qa_system.py
```

## Example Applications

### 1. Simple Chatbot (`applications/simple_chatbot.py`)

A conversational AI chatbot that uses RAG for knowledge-based responses.

**Features:**
- Interactive chat interface
- Knowledge base loading from files
- Conversation history
- Multiple query strategies
- Source attribution

**Usage:**
```bash
# Interactive mode
python examples/applications/simple_chatbot.py

# Batch demo mode
python examples/applications/simple_chatbot.py --mode batch

# Load custom files
python examples/applications/simple_chatbot.py --files doc1.txt doc2.md
```

**Example Interaction:**
```
🤖 RAG Engine Chatbot
==================================================
Loading knowledge base...
✅ Knowledge base loaded successfully!

🧑 You: What is artificial intelligence?

🤖 Bot: Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation.

📊 Confidence: 0.85
⏱️  Processing time: 1.23s

📚 Sources:
  1. What is AI (ai_textbook)
```

### 2. Document Q&A System (`applications/document_qa_system.py`)

A comprehensive document-based question-answering system that can process various file formats.

**Features:**
- Multiple file format support (TXT, MD, JSON)
- Directory-based document loading
- Batch question processing
- Document search functionality
- Evaluation capabilities
- Performance statistics

**Usage:**
```bash
# Interactive mode with sample documents
python examples/applications/document_qa_system.py

# Load specific directory
python examples/applications/document_qa_system.py --directory /path/to/docs

# Load specific files
python examples/applications/document_qa_system.py --files doc1.txt doc2.md

# Batch processing demo
python examples/applications/document_qa_system.py --mode batch

# Evaluation demo
python examples/applications/document_qa_system.py --mode evaluation
```

**Example Commands:**
```
📚 Document Q&A System
==================================================
📊 Document Statistics:
Total documents: 4
File types: {'.txt': 2, '.md': 1, '.json': 1}
Total size: 0.05 MB

❓ Your question: What programming languages are best for AI?
Strategy (basic/multi_query/rag_fusion) [basic]: multi_query

💡 Answer: Python is the most popular programming language for AI development due to its excellent libraries like TensorFlow, PyTorch, scikit-learn, pandas, and numpy. Other languages include R for statistics, Java for enterprise applications, and C++ for performance-critical applications.

📊 Confidence: 0.92
⏱️  Processing time: 1.45s
🔧 Strategy: multi_query

📚 Sources (2):
  1. programming_guide.txt (txt)
     Preview: Programming for AI and Machine Learning Getting started with AI programming requires...
  2. ai_overview.txt (txt)
     Preview: Artificial Intelligence: A Comprehensive Overview Artificial Intelligence (AI) represents...
```

### 3. Web Scraper RAG (`applications/web_scraper_rag.py`)

A RAG system that can scrape web content and answer questions based on the scraped information.

**Features:**
- Web page scraping and content extraction
- Asynchronous processing for multiple URLs
- Link discovery and crawling
- Wikipedia integration demo
- Interactive web RAG interface

**Usage:**
```bash
# Wikipedia scraping demo
python examples/applications/web_scraper_rag.py --mode wikipedia

# News scraping demo (example URLs)
python examples/applications/web_scraper_rag.py --mode news

# Link discovery demo
python examples/applications/web_scraper_rag.py --mode discovery

# Interactive mode
python examples/applications/web_scraper_rag.py --mode interactive

# Custom URLs
python examples/applications/web_scraper_rag.py --urls https://example.com/page1 https://example.com/page2
```

**Example Interaction:**
```
📖 Wikipedia Scraping Demo
==================================================
🕷️  Scraping 5 URLs...
✅ Successfully scraped 5 pages
📚 Added 5 documents to knowledge base

🤖 Testing with 5 questions...

1. Q: What is artificial intelligence?
   A: Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals...
   Confidence: 0.88
   Sources: 2 web pages
     - Artificial intelligence (en.wikipedia.org)
     - Machine learning (en.wikipedia.org)
```

## No API Keys Required

Perfect for getting started, testing, and demos:

| Example | Description | Requirements |
|---------|-------------|--------------|
| `example_no_api_keys.py` | Complete demo without any API keys | None |
| `example_multi_provider_llm.py` | Shows mock provider usage | None |
| `example_embedding_providers.py` | Local embeddings only | sentence-transformers |

### Mock Provider Features
- Contextual responses based on keywords
- Structured output support
- Perfect for CI/CD and testing
- No external dependencies

### Local LLM with Ollama
- Real LLM inference without API keys
- Supports llama2, mistral, codellama
- Full RAG functionality
- Requires Ollama installation

## Basic Examples

### Simple Usage (`example_simple.py`)

Basic RAG engine usage with minimal setup:

```python
from rag_engine.core.engine import RAGEngine
from rag_engine.core.models import Document

# Initialize engine
engine = RAGEngine()

# Add documents
documents = [
    Document(content="AI is artificial intelligence.", metadata={"topic": "AI"})
]
engine.add_documents(documents)

# Query
response = engine.query("What is AI?")
print(response.answer)
```

### Configuration Example (`example_configuration_management.py`)

Demonstrates different configuration approaches:

```python
from rag_engine.core.config import PipelineConfig, LLMProvider

# Custom configuration
config = PipelineConfig(
    llm_provider=LLMProvider.GOOGLE,
    llm_model="gemini-2.0-flash-lite",
    temperature=0.1,
    chunk_size=800,
    retrieval_k=5
)

engine = RAGEngine(config)
```

### Embedding Providers (`example_embedding_providers.py`)

Shows how to use different embedding providers:

```python
# Google embeddings
config = PipelineConfig(embedding_provider="google")

# OpenAI embeddings
config = PipelineConfig(embedding_provider="openai")

# Hugging Face embeddings
config = PipelineConfig(embedding_provider="huggingface")
```

## Advanced Examples

### Multi-Query Processing (`example_multi_query.py`)

Advanced query processing with multiple strategies:

```python
from rag_engine.query.processor import QueryProcessor

processor = QueryProcessor(config)

# Multi-query generation
processed = processor.process("What is AI?", "multi_query")

# RAG-Fusion
processed = processor.process("What is AI?", "rag_fusion")

# Query decomposition
processed = processor.process("Complex question about AI and ML", "decomposition")
```

### Advanced Indexing (`example_advanced_indexing.py`)

Different indexing strategies:

```python
from rag_engine.indexing.indexing_manager import IndexingManager

manager = IndexingManager(config)

# Basic indexing
manager.index_documents(documents, "basic")

# Multi-representation indexing
manager.index_documents(documents, "multi_representation")

# ColBERT indexing
manager.index_documents(documents, "colbert")
```

### Evaluation Demo (`example_evaluation_demo.py`)

Comprehensive evaluation with multiple frameworks:

```python
from rag_engine.evaluation.evaluation_manager import EvaluationManager

evaluator = EvaluationManager(config)

# Custom evaluation
result = evaluator.evaluate(test_cases, metrics=["custom"])

# RAGAS evaluation
result = evaluator.evaluate(test_cases, metrics=["ragas"])

# DeepEval evaluation
result = evaluator.evaluate(test_cases, metrics=["deepeval"])
```

### Hybrid Retrieval (`example_hybrid_retrieval.py`)

Advanced retrieval techniques:

```python
from rag_engine.retrieval.retrieval_engine import RetrievalEngine

retriever = RetrievalEngine(config)

# Vector retrieval
docs = retriever.retrieve("query", method="vector")

# Hybrid retrieval (vector + keyword)
docs = retriever.retrieve("query", method="hybrid")

# With reranking
docs = retriever.retrieve_with_rerank("query", k=10)
```

## Integration Examples

### API Integration (`example_api_integration.py`)

Using the RAG Engine through the REST API:

```python
import requests

# Add documents via API
response = requests.post(
    "http://localhost:8000/api/v1/documents",
    json={"documents": [{"content": "AI content", "metadata": {}}]},
    headers={"Authorization": "Bearer your-api-key"}
)

# Query via API
response = requests.post(
    "http://localhost:8000/api/v1/query",
    json={"query": "What is AI?", "strategy": "multi_query"},
    headers={"Authorization": "Bearer your-api-key"}
)
```

### Monitoring Integration (`example_monitoring_and_health_checks.py`)

System monitoring and health checks:

```python
from rag_engine.core.monitoring import get_monitoring_manager

# Start monitoring
manager = get_monitoring_manager()
manager.start()

# Record custom metrics
manager.record_metric("custom_metric", 1.0)

# Get health status
health = manager.get_health_status()
print(f"System status: {health.status}")
```

### Error Handling (`example_error_handling_resilience.py`)

Robust error handling and resilience:

```python
from rag_engine.core.exceptions import RAGEngineError

try:
    response = engine.query("What is AI?")
except RAGEngineError as e:
    print(f"RAG Engine error: {e}")
    # Implement fallback logic
```

## Running Examples

### Environment Setup

1. **Virtual Environment**:
   ```bash
   cd rag-engine
   uv sync
   source .venv/bin/activate  # Linux/Mac
   # or .venv\Scripts\activate  # Windows
   ```

2. **Environment Variables**:
   ```bash
   # Required
   export GOOGLE_API_KEY="your_google_api_key"
   
   # Optional
   export OPENAI_API_KEY="your_openai_key"
   ```

3. **Services** (if needed):
   ```bash
   # Start Chroma and Redis
   docker-compose up -d chroma redis
   
   # Or start individual services
   docker run -p 8001:8000 chromadb/chroma:latest
   docker run -p 6379:6379 redis:alpine
   ```

### Running Individual Examples

```bash
# Basic examples
python examples/example_simple.py
python examples/example_usage.py

# Advanced examples
python examples/example_advanced_indexing.py
python examples/example_advanced_reranking.py
python examples/example_comprehensive_evaluation.py

# Integration examples
python examples/example_integration_test.py
python examples/example_monitoring_and_health_checks.py

# Application examples
python examples/applications/simple_chatbot.py
python examples/applications/document_qa_system.py
python examples/applications/web_scraper_rag.py
```

### Running All Examples

```bash
# Run all basic examples
python run_examples.py --category basic

# Run all advanced examples
python run_examples.py --category advanced

# Run all examples
python run_examples.py --all
```

## Customization Guide

### Creating Custom Applications

1. **Basic Structure**:
   ```python
   from rag_engine.core.engine import RAGEngine
   from rag_engine.core.config import PipelineConfig
   from rag_engine.core.models import Document
   
   class MyRAGApp:
       def __init__(self):
           self.config = PipelineConfig(
               # Your custom configuration
           )
           self.engine = RAGEngine(self.config)
       
       def load_data(self, data_source):
           # Your data loading logic
           pass
       
       def process_query(self, query):
           # Your query processing logic
           return self.engine.query(query)
   ```

2. **Custom Document Processing**:
   ```python
   def process_custom_format(file_path):
       # Your custom file processing
       content = extract_content(file_path)
       metadata = extract_metadata(file_path)
       
       return Document(
           content=content,
           metadata=metadata,
           doc_id=generate_id(file_path)
       )
   ```

3. **Custom Query Strategies**:
   ```python
   from rag_engine.query.processor import QueryProcessor
   
   class CustomQueryProcessor(QueryProcessor):
       def custom_strategy(self, query: str):
           # Your custom query processing
           return processed_query
   ```

### Configuration Customization

1. **Environment-Specific Configs**:
   ```yaml
   # config/config.myapp.yaml
   llm:
     provider: "google"
     model: "gemini-2.0-flash-lite"
     temperature: 0.0
   
   retrieval:
     k: 10
     enable_reranking: true
   
   custom_settings:
     my_feature_enabled: true
     my_threshold: 0.8
   ```

2. **Runtime Configuration**:
   ```python
   # Override config at runtime
   config = PipelineConfig()
   config.temperature = 0.2
   config.retrieval_k = 7
   
   engine = RAGEngine(config)
   ```

### Adding New Features

1. **Custom Evaluators**:
   ```python
   from rag_engine.evaluation.custom_evaluator import CustomEvaluator
   
   class MyEvaluator(CustomEvaluator):
       def evaluate_response(self, query, response, expected):
           # Your evaluation logic
           return score
   ```

2. **Custom Providers**:
   ```python
   from rag_engine.generation.llm_providers import BaseLLMProvider
   
   class MyLLMProvider(BaseLLMProvider):
       def generate(self, prompt: str) -> str:
           # Your LLM integration
           return response
   ```

### Best Practices

1. **Error Handling**:
   ```python
   try:
       response = engine.query(user_input)
   except Exception as e:
       logger.error(f"Query failed: {e}")
       response = create_fallback_response()
   ```

2. **Performance Monitoring**:
   ```python
   import time
   
   start_time = time.time()
   response = engine.query(query)
   processing_time = time.time() - start_time
   
   logger.info(f"Query processed in {processing_time:.2f}s")
   ```

3. **Resource Management**:
   ```python
   # Use context managers for resources
   with RAGEngine(config) as engine:
       response = engine.query(query)
   # Engine automatically cleaned up
   ```

## Example Data

The examples include sample data in various formats:

- **Text files**: AI and ML documentation
- **Markdown files**: Technical guides and tutorials
- **JSON files**: Configuration and structured data
- **Web content**: Scraped articles and documentation

You can replace this sample data with your own content for testing.

## Troubleshooting Examples

### Common Issues

1. **Import Errors**:
   ```bash
   # Ensure you're in the right directory
   cd rag-engine
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   ```

2. **API Key Issues**:
   ```bash
   # Verify API key is set
   echo $GOOGLE_API_KEY
   
   # Test API key
   curl -H "Authorization: Bearer $GOOGLE_API_KEY" \
        https://generativelanguage.googleapis.com/v1/models
   ```

3. **Service Connection Issues**:
   ```bash
   # Check if services are running
   curl http://localhost:8001/api/v1/heartbeat  # Chroma
   curl http://localhost:6379  # Redis
   ```

### Getting Help

- **Documentation**: Check the main documentation in `docs/`
- **API Reference**: See `docs/API_DOCUMENTATION.md`
- **Configuration**: See `docs/CONFIGURATION_GUIDE.md`
- **Troubleshooting**: See `docs/TROUBLESHOOTING_FAQ.md`
- **Issues**: Open an issue on the project repository

## Contributing Examples

To contribute new examples:

1. **Follow the existing structure**
2. **Include comprehensive documentation**
3. **Add error handling and logging**
4. **Test with different configurations**
5. **Update this README**

Example template:
```python
#!/usr/bin/env python3
"""
Example: [Your Example Name]

Description of what this example demonstrates.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_engine.core.engine import RAGEngine

def main():
    """Main example function"""
    # Your example code here
    pass

if __name__ == "__main__":
    main()
```