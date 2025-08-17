# CLI Reference

Documentation for the RAG System Command Line Interface.

## Overview

The RAG System provides a comprehensive Command Line Interface (CLI) for managing the system, ingesting documents, performing queries, evaluating performance, and managing configurations. It's built using Click and provides a user-friendly experience.

## Installation

The RAG CLI is installed as part of the main RAG System package. After cloning the repository and setting up your Python environment (e.g., using `pip install -e .` or `uv sync`), the `rag` command will be available in your terminal.

For detailed installation instructions, refer to the [Installation Guide](../../getting-started/installation.md).

## Basic Usage

All `rag` commands follow a similar structure:

```bash
rag [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS] [ARGUMENTS]
```

You can get help for the main CLI or any specific command using the `--help` flag:

```bash
rag --help
rag query --help
rag index --help
```

## Global Options

These options can be applied to any `rag` command:

*   `--config`, `-c <path>`: Specify a custom configuration file path.
*   `--environment`, `-e <name>`: Specify the environment to load configuration for (e.g., `development`, `testing`, `production`).
*   `--verbose`, `-v`: Enable verbose logging for more detailed output.

## Commands

### `rag serve`

Starts the RAG API server, making the system accessible via HTTP endpoints.

```bash
rag serve [OPTIONS]
```

**Options:**
*   `--host <address>`: Host to bind the server to (default: `0.0.0.0`).
*   `--port <number>`: Port to bind the server to (default: `8000`).
*   `--reload`: Enable auto-reload for development (server restarts on code changes).

**Example:**
```bash
rag serve --host 127.0.0.1 --port 8080 --reload
```

### `rag query <question>`

Processes a natural language question using the RAG system and returns an answer.

```bash
rag query <question> [OPTIONS]
```

**Arguments:**
*   `<question>`: The natural language question or prompt to query the RAG system with.

**Options:**
*   `--k <number>`: Number of relevant documents to retrieve for context (default: `5`).
*   `--include-sources / --no-include-sources`: Whether to include the content and metadata of the source documents in the output (default: `True`).
*   `--output-format <format>`: Specify the output format. Choose from `text` (default, human-readable) or `json` (machine-readable).

**Examples:**
```bash
rag query "What is retrieval-augmented generation?"
rag query "Explain the RAG architecture" --k 10 --no-include-sources
rag query "What are the benefits of RAG?" --output-format json
```

### `rag index` (Group)

A group of commands for managing the document index.

#### `rag index files <paths...>`

Indexes documents from local files or directories into the RAG knowledge base.

```bash
rag index files <paths...> [OPTIONS]
```

**Arguments:**
*   `<paths...>`: One or more paths to files or directories to index.

**Options:**
*   `--clear`: Clear all existing documents from the index before adding new ones.
*   `--recursive`, `-r`: Recursively index files within specified directories.
*   `--pattern <glob_pattern>`: A glob pattern (e.g., `*.txt`, `*.md`) to filter files when indexing directories.

**Examples:**
```bash
rag index files /path/to/document.txt
rag index files /path/to/doc1.md /path/to/doc2.json
rag index files --recursive --pattern "*.txt" /path/to/my_docs/
rag index files --clear /path/to/new_documents/
```

#### `rag index web <urls...>`

Indexes content from specified web URLs into the RAG knowledge base.

```bash
rag index web <urls...> [OPTIONS]
```

**Arguments:**
*   `<urls...>`: One or more web URLs to crawl and index.

**Options:**
*   `--clear`: Clear all existing documents from the index before adding new ones.
*   `--max-depth <number>`: Maximum crawling depth for linked pages (default: `1`).
*   `--include-links`: Include the text of links found on pages in the indexed content.

**Examples:**
```bash
rag index web https://example.com/page1
rag index web https://example.com/article1 https://example.com/article2 --max-depth 2
rag index web --clear https://example.com/new_content/
```

#### `rag index clear`

Clears all indexed documents from the RAG knowledge base.

```bash
rag index clear [OPTIONS]
```

**Options:**
*   `--confirm`: Skip the confirmation prompt (use with caution).

**Example:**
```bash
rag index clear
rag index clear --confirm
```

### `rag evaluate <test_file>`

Evaluates the RAG system's performance using a set of test cases defined in a JSON file.

```bash
rag evaluate <test_file> [OPTIONS]
```

**Arguments:**
*   `<test_file>`: Path to a JSON file containing the test cases.

**Options:**
*   `--frameworks <name>`: Specify one or more evaluation frameworks to use (e.g., `custom`, `ragas`). Can be specified multiple times. (default: `custom`).
*   `--output <path>`: Path to a file where the evaluation results will be saved.
*   `--output-format <format>`: Specify the output format. Choose from `text` (default, human-readable) or `json` (machine-readable).

**Example:**
```bash
rag evaluate my_test_cases.json
rag evaluate my_test_cases.json --frameworks custom --frameworks ragas --output results.json --output-format json
```

### `rag status`

Displays the current status and information about the RAG system, including configuration, indexed document counts, and component health.

```bash
rag status
```

**Example:**
```bash
rag status
```

### `rag config` (Group)

A group of commands for managing system configuration.

#### `rag config show`

Displays the current active configuration of the RAG system.

```bash
rag config show [OPTIONS]
```

**Options:**
*   `--format <format>`: Specify the output format. Choose from `yaml` (default) or `json`.

**Examples:**
```bash
rag config show
rag config show --format json
```

#### `rag config validate <config_file>`

Validates a specified configuration file against the system's schema.

```bash
rag config validate <config_file>
```

**Arguments:**
*   `<config_file>`: Path to the configuration file to validate.

**Example:**
```bash
rag config validate config/my_custom_config.yaml
```
