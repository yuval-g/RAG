# Advanced Examples

This directory contains documentation for advanced examples showcasing more complex and specialized features of the RAG System.

For the actual executable example files, please refer to the main `examples/` directory in the project root.

## Examples Covered

*   **`example_advanced_indexing.py`**: Demonstrates advanced indexing strategies such as RAPTOR, ColBERT, and Multi-representation indexing.
*   **`example_advanced_reranking.py`**: Explores advanced reranking techniques to improve the relevance of retrieved documents.
*   **`example_comprehensive_evaluation.py`**: Provides a comprehensive example of evaluating the RAG System with multiple metrics and frameworks.
*   **`example_grounded_generation.py`**: Showcases how to achieve grounded generation with citations and validation, ensuring responses are factual and attributable.
*   **`example_hybrid_retrieval.py`**: Illustrates hybrid retrieval, combining vector and keyword search for more robust document retrieval.
*   **`example_multi_provider_llm.py`**: Demonstrates integration with and switching between multiple LLM providers.
*   **`example_self_correction.py`**: Explores mechanisms for self-correction within the RAG pipeline.

## How to Run

To run these examples, navigate to the project root directory and execute them using `uv run`:

```bash
uv run python examples/example_advanced_indexing.py
uv run python examples/example_grounded_generation.py
# ... and so on for other advanced examples
```

Note that some advanced examples may require specific API keys or configurations. Refer to the example file's comments and the [RAG Engine Examples README](../../../examples/README.md) for prerequisites.
