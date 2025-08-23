# Examples Documentation

This directory contains documentation for the various examples available in the RAG Engine, organized by complexity and use case.

## Example Categories

### [Basic Examples](./basic-examples/)
Simple examples to get you started with the RAG Engine:
- `example_simple.py` - Basic querying without API keys
- `example_usage.py` - Common usage patterns and integrations

### [Advanced Examples](./advanced-examples/)
Sophisticated examples showcasing advanced features:
- `example_advanced_indexing.py` - RAPTOR, ColBERT, and multi-representation indexing
- `example_hybrid_retrieval.py` - Combining vector and keyword search
- `example_grounded_generation.py` - Citations and fact validation
- `example_self_correction.py` - Self-correction mechanisms

### [Integration Examples](./integration-examples/)
Examples showing how to integrate RAG Engine with other systems:
- `example_integration_test.py` - Full system integration testing
- `applications/document_qa_system.py` - Document Q&A system
- `applications/simple_chatbot.py` - Simple chatbot application
- `applications/web_scraper_rag.py` - Web scraper integration

## Learning Path

1. Start with [Basic Examples](./basic-examples/) to understand fundamental concepts
2. Move to [Advanced Examples](./advanced-examples/) to explore sophisticated features
3. Review [Integration Examples](./integration-examples/) to see how to build complete applications

## How to Run Examples

All examples can be run from the project root directory:

```bash
# Run a basic example
uv run python examples/example_simple.py

# Run an advanced example
uv run python examples/example_grounded_generation.py

# Run an integration example
uv run python examples/applications/simple_chatbot.py
```

Note that some examples may require API keys or specific configurations. Check the example files and the main [Examples README](../examples/README.md) for prerequisites.

## Example Structure

Each example category contains:
1. A README explaining the examples in that category
2. References to the actual executable files in the project's `examples/` directory
3. Instructions for running the examples
4. Prerequisites and setup information

## Best Practices Demonstrated

The examples showcase:
- Proper error handling
- Configuration management
- Resource cleanup
- Performance optimization
- Testing strategies
- Integration patterns

## Related Documentation

- [Getting Started](../getting-started/) - Basic system usage
- [User Guides](../user-guides/) - Comprehensive feature documentation
- [API Reference](../api-reference/) - Detailed API documentation