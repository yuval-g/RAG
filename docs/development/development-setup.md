# Development Setup Guide

This guide provides instructions for setting up your local development environment for the RAG System, enabling you to contribute to the codebase, run tests, and develop new features.

## Table of Contents

1.  [Prerequisites](#prerequisites)
2.  [Getting the Code](#getting-the-code)
3.  [Environment Setup](#environment-setup)
4.  [Installing Dependencies](#installing-dependencies)
5.  [Configuration for Development](#configuration-for-development)
6.  [Running the Application in Development Mode](#running-the-application-in-development-mode)
7.  [Running Tests](#running-tests)
8.  [Linting and Formatting](#linting-and-formatting)
9.  [Debugging](#debugging)

## 1. Prerequisites

Before you begin, ensure you have the following installed on your system:

*   **Git**: For cloning the repository.
*   **Python 3.13 or higher**: The RAG System is developed with Python 3.13+.
*   **`pip` or `uv`**: Python package managers. `uv` is recommended for faster installations.
*   **Docker (Optional)**: If you plan to develop using Docker containers or run Docker-based services (like ChromaDB or Redis).

## 2. Getting the Code

1.  **Fork the Repository**: Go to the RAG System GitHub repository and click the "Fork" button.
2.  **Clone Your Fork**: Clone your forked repository to your local machine:
    ```bash
git clone https://github.com/<your-username>/rag-engine.git
cd rag-engine
    ```

## 3. Environment Setup

It is highly recommended to use a Python virtual environment to manage project dependencies.

1.  **Create a Virtual Environment**:
    ```bash
python3 -m venv .venv
    ```
2.  **Activate the Virtual Environment**:
    *   **On Linux/macOS:**
        ```bash
source .venv/bin/activate
        ```
    *   **On Windows (Command Prompt):**
        ```bash
.venv\Scripts\activate.bat
        ```
    *   **On Windows (PowerShell):**
        ```powershell
.venv\Scripts\Activate.ps1
        ```

    Your terminal prompt should now show `(.venv)` indicating the virtual environment is active.

## 4. Installing Dependencies

Install the project dependencies in editable mode. This allows your local changes to the source code to be reflected without reinstallation.

### Using `uv` (Recommended for speed)

```bash
uv sync
```

### Using `pip`

```bash
pip install -e .
```

## 5. Configuration for Development

For development, you'll typically use a `.env` file to manage API keys and other settings.

1.  **Create `.env` file**: Copy the example environment file:
    ```bash
cp .env.example .env
    ```
2.  **Edit `.env`**: Open the `.env` file and provide your API keys (e.g., `GOOGLE_API_KEY`, `OPENAI_API_KEY`) and set `ENVIRONMENT=development`.

    ```ini
# .env
ENVIRONMENT=development
LOG_LEVEL=DEBUG
GOOGLE_API_KEY=your-google-api-key-here
# ... other API keys and settings
    ```

3.  **Local Services (Optional)**: If you plan to use local ChromaDB or Redis instances, ensure they are running. You can use Docker Compose for this:
    ```bash
docker-compose up -d chroma redis
    ```

## 6. Running the Application in Development Mode

### Running the API Server

To run the FastAPI server with auto-reloading (restarts on code changes):

```bash
rag serve --reload
```

This will typically run the server on `http://localhost:8000`.

### Running CLI Commands

You can execute any CLI command directly from your activated virtual environment:

```bash
rag status
rag index files /path/to/your/test_documents/
rag query "What is the RAG system?"
```

## 7. Running Tests

The project uses `pytest` for testing. It's crucial to run tests before submitting any changes.

### Run All Tests

```bash
uv run python -m pytest
```

### Run Specific Test Files

```bash
uv run python -m pytest tests/core/test_engine.py
```

### Run Tests with Verbose Output

```bash
uv run python -m pytest -v
```

## 8. Linting and Formatting

Maintain code quality and consistency using linters and formatters.

*   **Black**: For code formatting.
*   **Ruff**: For linting.

Run them manually:

```bash
black .
ruff check .
```

It's recommended to integrate these tools with your IDE for automatic formatting and linting.

## 9. Debugging

### Using `pdb` (Python Debugger)

You can insert `breakpoint()` (Python 3.7+) or `import pdb; pdb.set_trace()` into your code to start an interactive debugger session.

```python
def my_function():
    # some code
    breakpoint() # Execution will pause here
    # more code
```

### IDE Debugging

Most modern IDEs (VS Code, PyCharm) have excellent debugging capabilities. Configure your IDE to run the `rag serve` command or specific test files in debug mode.

### Docker Container Debugging

If you are developing within Docker containers, you can exec into the container to debug:

```bash
docker-compose exec rag-engine bash
```

From within the container, you can run Python scripts or use `pdb`.

For more advanced debugging techniques, refer to the [Troubleshooting Guide](../operations/troubleshooting.md).
