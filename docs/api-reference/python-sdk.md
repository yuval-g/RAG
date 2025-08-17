# Python SDK Reference

Documentation for the RAG System Python SDK.

## Overview

The RAG System provides a Python SDK for programmatic access to all system functionality, allowing seamless integration with your Python applications.

## Installation

The RAG System Python SDK can be installed using pip:

```bash
pip install rag-engine-sdk
```

## Client Initialization

To interact with the RAG Engine API, you need to initialize the `RAGClient` with the base URL of your API and your API key.

```python
from rag_engine_sdk import RAGClient

# Initialize client
client = RAGClient(
    base_url="https://api.yourdomain.com/api/v1",  # Replace with your API base URL
    api_key="your-api-key"                         # Replace with your actual API key
)
```

## Core Methods

The `RAGClient` provides the following core methods to interact with the RAG Engine:

### `add_documents(documents: List[Dict]) -> Dict`

Indexes new documents into the RAG knowledge base.

**Parameters:**
- `documents` (List[Dict]): A list of document dictionaries, where each dictionary should contain at least `content` and optionally `metadata` and `doc_id`.

**Example:**
```python
documents = [
    {
        "content": "AI is a branch of computer science that aims to create intelligent machines.",
        "metadata": {"title": "AI Introduction", "source": "textbook", "category": "technology"},
        "doc_id": "doc_ai_intro"
    },
    {
        "content": "Machine learning is a subset of AI that enables systems to learn from data.",
        "metadata": {"title": "Machine Learning Basics", "source": "online_course", "category": "technology"},
        "doc_id": "doc_ml_basics"
    }
]
result = client.add_documents(documents)
print(f"Indexed {result['indexed_count']} documents. IDs: {result['document_ids']}")
# Expected output: Indexed 2 documents. IDs: ['doc_ai_intro', 'doc_ml_basics']
```

### `query(query: str, strategy: str = "basic", k: int = 5, metadata_filter: Dict = None, options: Dict = None) -> Dict`

Processes a user query using the RAG system and returns a generated answer along with source documents.

**Parameters:**
- `query` (str): The user's question or prompt.
- `strategy` (str, optional): The query processing strategy to use (e.g., "basic", "multi_query", "rag_fusion"). Defaults to "basic".
- `k` (int, optional): The number of relevant documents to retrieve. Defaults to 5.
- `metadata_filter` (Dict, optional): A dictionary to filter documents based on their metadata.
- `options` (Dict, optional): Additional options for the query, such as `temperature` or `max_tokens` for the LLM.

**Example:**
```python
response = client.query(
    query="What is artificial intelligence?",
    strategy="multi_query",
    k=3,
    metadata_filter={"category": "technology"},
    options={"temperature": 0.1, "max_tokens": 200}
)
print(f"Answer: {response['answer']}")
print(f"Confidence: {response['confidence_score']}")
if 'source_documents' in response and response['source_documents']:
    print("Source Documents:")
    for doc in response['source_documents']:
        print(f"- {doc['metadata'].get('title', 'N/A')} (Score: {doc['relevance_score']:.2f})")
```

### `evaluate(test_cases: List[Dict], metrics: List[str] = None, options: Dict = None) -> Dict`

Evaluates the RAG system's performance using a set of predefined test cases.

**Parameters:**
- `test_cases` (List[Dict]): A list of test case dictionaries, each containing a `question` and an `expected_answer`.
- `metrics` (List[str], optional): A list of metrics to calculate (e.g., "custom", "ragas"). Defaults to None, which implies default metrics.
- `options` (Dict, optional): Additional evaluation options, such as `detailed_results`.

**Example:**
```python
test_cases = [
    {
        "question": "What is AI?",
        "expected_answer": "AI is a branch of computer science that aims to create intelligent machines."
    },
    {
        "question": "What is ML?",
        "expected_answer": "ML is a subset of AI that enables systems to learn from data."
    }
]
evaluation = client.evaluate(test_cases, metrics=["custom", "ragas"], options={"detailed_results": True})
print(f"Overall Evaluation Score: {evaluation['overall_score']:.2f}")
print("Metric Scores:", evaluation['metric_scores'])
if 'test_case_results' in evaluation:
    print("First Test Case Result:", evaluation['test_case_results'][0])
```

### `query_stream(query: str, on_progress: Callable, on_result: Callable, strategy: str = "basic", k: int = 5, metadata_filter: Dict = None, options: Dict = None)`

Connects to the WebSocket API to stream query processing results in real-time.

**Parameters:**
- `query` (str): The user's question or prompt.
- `on_progress` (Callable): A callback function `(progress: Dict) -> None` to handle progress updates.
- `on_result` (Callable): A callback function `(result: Dict) -> None` to handle the final result.
- `strategy` (str, optional): The query processing strategy to use. Defaults to "basic".
- `k` (int, optional): The number of relevant documents to retrieve. Defaults to 5.
- `metadata_filter` (Dict, optional): A dictionary to filter documents based on their metadata.
- `options` (Dict, optional): Additional options for the query.

**Example (JavaScript SDK example adapted for Python concept):**
```python
import asyncio

async def main():
    # Assuming an async RAGClient or similar for streaming
    # This is a conceptual example based on the JavaScript SDK snippet
    # Actual implementation would depend on the Python SDK's async capabilities
    
    # client.query_stream(
    #     query='What is AI?',
    #     on_progress=lambda progress: print('Progress:', progress),
    #     on_result=lambda result: print('Result:', result)
    # )
    print("Streaming functionality would be implemented here, likely using websockets library.")

if __name__ == "__main__":
    asyncio.run(main())
```
