# Integration Examples

This directory contains documentation for examples demonstrating how to integrate the RAG System with other components or systems, and how to perform end-to-end testing.

For the actual executable example files, please refer to the main `examples/` directory in the project root.

## Examples Covered

*   **`example_integration_test.py`**: A comprehensive example demonstrating a full system integration test, verifying the end-to-end workflow of the RAG System.
*   **`applications/document_qa_system.py`**: An example of building a document Q&A system using the RAG Engine.
*   **`applications/simple_chatbot.py`**: An example of creating a simple chatbot application powered by the RAG Engine.
*   **`applications/web_scraper_rag.py`**: An example demonstrating how to integrate a web scraper with the RAG Engine for indexing web content.

## How to Run

To run these examples, navigate to the project root directory and execute them using `uv run`:

```bash
uv run python examples/example_integration_test.py
uv run python examples/applications/simple_chatbot.py
# ... and so on for other integration examples
```

Note that some integration examples may require specific API keys or configurations. Refer to the example file's comments and the [RAG Engine Examples README](../../../examples/README.md) for prerequisites.
