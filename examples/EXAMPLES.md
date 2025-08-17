# RAG Engine Examples

This directory contains comprehensive examples demonstrating all features of the RAG Engine.

## Quick Start Examples

### 🚀 Basic Usage
- **`example_simple.py`** - Simple introduction to query processing without API keys
- **`example_usage.py`** - Practical usage patterns and integration examples

## Core Component Examples

### 🔍 Query Processing
- **`example_simple.py`** - Query transformation strategies (multi-query, RAG-Fusion, HyDE, etc.)
- **`example_usage.py`** - Advanced query processing workflows

### 📚 Indexing & Retrieval
- **`example_advanced_indexing.py`** - Advanced indexing strategies (RAPTOR, ColBERT, Multi-representation)
- **`example_hybrid_retrieval.py`** - Hybrid retrieval combining vector and keyword search
- **`example_advanced_reranking.py`** - Advanced reranking techniques

### 🧠 Generation & Grounding
- **`example_grounded_generation.py`** - Grounded generation with citations and validation
- **`example_multi_provider_llm.py`** - Multiple LLM provider integration
- **`example_self_correction.py`** - Self-correction mechanisms

### 🔧 Providers & Configuration
- **`example_embedding_providers.py`** - Embedding provider configuration and usage
- **`example_vector_store_providers.py`** - Vector store provider examples

### 📊 Evaluation & Testing
- **`example_evaluation_demo.py`** - Basic evaluation framework demonstration
- **`example_comprehensive_evaluation.py`** - Comprehensive evaluation with multiple metrics
- **`example_integration_test.py`** - Full system integration testing

## Running Examples

### Prerequisites
1. Install dependencies:
   ```bash
   uv sync
   ```

2. Set up environment variables (optional for some examples):
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

### Run Individual Examples
```bash
# Basic examples (no API keys required)
uv run python examples/example_simple.py

# Advanced examples (may require API keys)
uv run python examples/example_usage.py
uv run python examples/example_grounded_generation.py
```

### Run All Examples
```bash
python run_examples.py
```
This will run all examples and save logs to `examples.log`.

## Example Categories

### 🟢 No API Keys Required
These examples work without any external API keys:
- `example_simple.py` - Basic structure and capabilities
- `example_embedding_providers.py` - Provider configuration
- `example_vector_store_providers.py` - Vector store setup

### 🟡 Optional API Keys
These examples have fallback behavior without API keys:
- `example_usage.py` - Shows structure even without keys
- `example_advanced_indexing.py` - Demonstrates indexing concepts
- `example_evaluation_demo.py` - Basic evaluation framework

### 🔴 API Keys Required
These examples require Google API keys for full functionality:
- `example_grounded_generation.py` - Requires GOOGLE_API_KEY
- `example_integration_test.py` - Requires GOOGLE_API_KEY
- `example_self_correction.py` - Requires GOOGLE_API_KEY

## API Key Setup

To enable full functionality:

1. **Google Gemini API**:
   - Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Set `GOOGLE_API_KEY` environment variable

2. **Optional Services** (for specific examples):
   - Pinecone: `PINECONE_API_KEY`
   - Weaviate: `WEAVIATE_URL`, `WEAVIATE_API_KEY`
   - OpenAI (for embeddings): `OPENAI_API_KEY`

## Example Output

All examples include:
- ✅ Success indicators for working features
- ⚠️ Warnings for missing dependencies or API keys
- ❌ Clear error messages with solutions
- 💡 Tips and next steps

## Contributing

When adding new examples:
1. Follow the naming convention: `example_[feature].py`
2. Include comprehensive docstrings
3. Handle missing API keys gracefully
4. Add appropriate success/error indicators
5. Update this README with the new example