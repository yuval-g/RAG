# Testing Guide

This guide provides an overview of the testing strategy for the RAG System, including how to run tests, write new tests, and ensure code quality through testing.

## Table of Contents

1.  [Testing Framework](#testing-framework)
2.  [Types of Tests](#types-of-tests)
3.  [Running Tests](#running-tests)
4.  [Test Structure](#test-structure)
5.  [Writing Tests](#writing-tests)
6.  [Test Coverage](#test-coverage)
7.  [Continuous Integration (CI)](#continuous-integration-ci)

## 1. Testing Framework

The RAG System uses `pytest` as its primary testing framework. `pytest` is a powerful and flexible framework that makes it easy to write and run tests.

## 2. Types of Tests

We categorize tests into the following types:

*   **Unit Tests**: Focus on testing individual functions, methods, or classes in isolation. They ensure that each small piece of code works as expected.
    *   **Location**: Typically found in `tests/<module_name>/test_*.py`.
*   **Integration Tests**: Verify that different components or modules of the system work correctly together. They test the interactions between units.
    *   **Location**: Often found in `tests/integration/` or within module-specific `tests/` directories if they involve multiple components.
*   **End-to-End (E2E) Tests**: Test the entire system flow from start to finish, simulating real-user scenarios. These often involve the API, database, and external services.
    *   **Location**: Typically found in `tests/integration/test_end_to_end_workflows.py`.

## 3. Running Tests

Ensure your Python virtual environment is activated before running tests.

### Run All Tests

To run all tests in the project:

```bash
uv run python -m pytest
```

### Run Tests with Verbose Output

To see more detailed output during test execution:

```bash
uv run python -m pytest -v
```

### Run Specific Test Files

To run tests from a specific file:

```bash
uv run python -m pytest tests/core/test_configuration_manager.py
```

### Run Tests by Keyword

To run tests whose names match a specific keyword expression:

```bash
uv run python -m pytest -k "test_query_basic"
```

### Pytest Configuration

The `pytest` configuration is defined in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
addopts = "-v --tb=short"
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
filterwarnings = [
    "ignore::pytest.PytestCollectionWarning"
]
```

## 4. Test Structure

Tests are organized in the `tests/` directory, mirroring the structure of the `src/` directory.

```
tests/
├── api/
│   └── test_api_integration.py
├── cli/
│   └── test_cli_integration.py
├── core/
│   ├── test_configuration_manager.py
│   ├── test_resilience.py
│   └── ...
├── integration/
│   └── test_end_to_end_workflows.py
└── ...
```

*   Test files should start with `test_` (e.g., `test_engine.py`).
*   Test functions within a file should start with `test_` (e.g., `def test_my_function():`).
*   Test classes (if used) should start with `Test` (e.g., `class TestMyFeature:`).

## 5. Writing Tests

When writing tests, consider the following guidelines:

*   **Arrange-Act-Assert (AAA)**: Organize your tests into three distinct sections:
    *   **Arrange**: Set up the test data and environment.
    *   **Act**: Perform the action being tested.
    *   **Assert**: Verify the outcome of the action.
*   **Fixtures**: Use `pytest` fixtures to set up common test data or resources (e.g., a RAG Engine instance, a temporary database).
*   **Mocks**: For unit tests, use `unittest.mock` or `pytest-mock` to mock external dependencies (e.g., LLM API calls, database interactions) to isolate the unit under test.
*   **Clear Assertions**: Use clear and specific assertions to verify expected behavior.
*   **Edge Cases**: Test edge cases, invalid inputs, and error conditions.

**Example Test Structure:**

```python
import pytest
from src.rag_engine.core.engine import RAGEngine
from src.rag_engine.core.config import PipelineConfig

@pytest.fixture
def mock_rag_engine():
    # Arrange: Setup a mock or minimal RAGEngine instance
    config = PipelineConfig(llm_provider="mock", embedding_provider="mock", vector_store="memory")
    engine = RAGEngine(config)
    return engine

def test_simple_query(mock_rag_engine):
    # Act: Perform a query
    response = mock_rag_engine.query("What is AI?")

    # Assert: Verify the response
    assert response.answer is not None
    assert "AI" in response.answer
    assert response.confidence_score > 0.5
```

## 6. Test Coverage

Aim for high test coverage, especially for core logic and critical paths. While 100% coverage is not always practical or necessary, strive for a good balance that ensures confidence in the codebase.

*   Use tools like `pytest-cov` to measure test coverage.
    ```bash
uv run python -m pytest --cov=src --cov-report=term-missing
    ```

## 7. Continuous Integration (CI)

Tests are automatically run as part of the Continuous Integration (CI) pipeline whenever changes are pushed to the repository. This ensures that new code does not introduce regressions and maintains the overall health of the codebase.

Before submitting a Pull Request, ensure all tests pass locally.
