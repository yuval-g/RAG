# Test Organization

This directory contains all tests for the RAG Engine, organized to mirror the source code structure.

## Directory Structure

```
tests/
├── core/           # Core component tests (models, interfaces, providers)
├── evaluation/     # Evaluation framework tests
├── generation/     # Generation and LLM provider tests
├── indexing/       # Document indexing tests
├── integration/    # End-to-end integration tests
├── query/          # Query processing tests
├── retrieval/      # Retrieval engine tests
└── routing/        # Query routing tests
```

## Test Categories

### Core Tests (`tests/core/`)
- `test_core_*.py` - Core functionality tests
- `test_embedding_providers*.py` - Embedding provider tests
- `test_vector_store_providers.py` - Vector store provider tests
- `test_exceptions_simple.py` - Exception handling tests

### Evaluation Tests (`tests/evaluation/`)
- `test_evaluation_manager.py` - Main evaluation manager
- `test_custom_evaluator.py` - Custom evaluation metrics
- `test_*_integration.py` - Third-party evaluation integrations (RAGAS, DeepEval)

### Generation Tests (`tests/generation/`)
- `test_generation_*.py` - Generation engine tests
- `test_llm_providers*.py` - LLM provider tests

### Indexing Tests (`tests/indexing/`)
- `test_*_indexer.py` - Various indexing strategies
- `test_indexing_manager.py` - Indexing orchestration

### Query Tests (`tests/query/`)
- `test_*.py` - Query processing techniques (HyDE, RAG Fusion, etc.)

### Retrieval Tests (`tests/retrieval/`)
- `test_retrieval_engine.py` - Main retrieval engine
- `test_*_retrieval.py` - Retrieval strategies
- `test_reranker.py` - Result reranking
- `test_self_correction.py` - Self-correction mechanisms

### Routing Tests (`tests/routing/`)
- `test_*_router.py` - Query routing strategies
- `test_query_structurer.py` - Query structuring

### Integration Tests (`tests/integration/`)
- End-to-end system tests
- Cross-component integration tests

## Running Tests

Run all tests:
```bash
uv run pytest tests/
```

Run tests for a specific component:
```bash
uv run pytest tests/core/
uv run pytest tests/retrieval/
```

Run a specific test file:
```bash
uv run pytest tests/core/test_embedding_providers.py
```