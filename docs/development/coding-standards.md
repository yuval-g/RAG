# Coding Standards

This document outlines the coding standards and best practices for the RAG System codebase. Adhering to these standards ensures code consistency, readability, maintainability, and quality across the project.

## Table of Contents

1.  [General Principles](#general-principles)
2.  [Code Formatting](#code-formatting)
3.  [Linting](#linting)
4.  [Naming Conventions](#naming-conventions)
5.  [Docstrings and Comments](#docstrings-and-comments)
6.  [Type Hinting](#type-hinting)
7.  [Imports](#imports)
8.  [Error Handling](#error-handling)
9.  [Testing](#testing)

## 1. General Principles

*   **Readability**: Code should be easy to understand for anyone reading it, not just the author.
*   **Simplicity**: Prefer simple, straightforward solutions over complex ones.
*   **Consistency**: Follow existing patterns and styles within the codebase.
*   **Modularity**: Break down complex problems into smaller, manageable functions or classes.
*   **DRY (Don't Repeat Yourself)**: Avoid code duplication.

## 2. Code Formatting

We use `Black` as an uncompromising code formatter to ensure consistent formatting across the entire codebase. This eliminates bikeshedding over style and allows developers to focus on content.

**How to format your code:**

```bash
black .
```

It is highly recommended to configure your IDE (e.g., VS Code, PyCharm) to automatically format code with Black on save.

## 3. Linting

We use `Ruff` for fast Python linting. Ruff helps enforce code quality, detect potential bugs, and highlight stylistic issues.

**How to run the linter:**

```bash
ruff check .
```

Address any warnings or errors reported by Ruff before submitting your code. Integrate Ruff with your IDE for real-time feedback.

## 4. Naming Conventions

Follow Python's standard naming conventions (PEP 8):

*   **Modules**: Short, all-lowercase names. Underscores can be used if it improves readability (e.g., `my_module.py`).
*   **Packages**: Short, all-lowercase names. Should not contain underscores (e.g., `my_package`).
*   **Classes**: `PascalCase` (e.g., `MyClass`, `RAGEngine`).
*   **Functions and Variables**: `snake_case` (e.g., `my_function`, `my_variable`).
*   **Constants**: `ALL_CAPS_WITH_UNDERSCORES` (e.g., `MAX_RETRIES`).
*   **Private Members**: Prefix with a single underscore (e.g., `_private_method`). Avoid using double underscores (`__`).

## 5. Docstrings and Comments

*   **Docstrings**: All modules, classes, and public functions/methods should have docstrings. Use [Google Style Docstrings](https://google.github.io/styleguide/pyguide.html#pyguide-python-language-rules-and-python-style) for consistency.
    ```python
    def my_function(arg1: str, arg2: int) -> bool:
        """Brief description of what the function does.

        Args:
            arg1: Description of arg1.
            arg2: Description of arg2.

        Returns:
            Description of the return value.
        """
        # ... code
    ```
*   **Comments**: Use comments sparingly to explain *why* a piece of code exists or *why* a particular approach was taken, rather than *what* the code does (which should be clear from the code itself).

## 6. Type Hinting

Use [type hints](https://docs.python.org/3/library/typing.html) for function arguments, return values, and variables to improve code clarity, enable static analysis, and reduce bugs.

```python
from typing import List, Dict, Any

def process_data(data: List[Dict[str, Any]]) -> Dict[str, int]:
    # ... code
    pass
```

## 7. Imports

Organize imports according to PEP 8 guidelines:

1.  Standard library imports.
2.  Third-party library imports.
3.  Local application/project-specific imports.

Sort imports alphabetically within each group. Use `isort` (often integrated with `ruff` or `black`) to automate this.

```python
import os
import sys

import click
from rich.console import Console

from .core.engine import RAGEngine
from .core.config import ConfigurationManager
```

## 8. Error Handling

*   **Use Exceptions**: Use Python's exception mechanism for error handling.
*   **Specific Exceptions**: Raise and catch specific exceptions rather than broad `Exception` types.
*   **Custom Exceptions**: Define custom exceptions for application-specific error conditions.
*   **Logging**: Log errors with appropriate severity levels.

## 9. Testing

All new features and bug fixes should be accompanied by unit and/or integration tests. Refer to the [Testing Guide](testing.md) for more details.
