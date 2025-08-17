# Bare Metal Deployment

This guide provides a general approach to deploying the RAG System directly on bare metal servers.

## Overview

The RAG System can be deployed directly on physical servers or virtual machines for maximum control over performance, resources, and security. This approach is suitable for environments where containerization (Docker) or orchestration (Kubernetes) is not preferred or feasible.

## Prerequisites

Before deploying on bare metal, ensure your server meets the following requirements:

*   **Operating System**: A Linux-based operating system (e.g., Ubuntu, CentOS, Debian) is recommended.
*   **Python**: Python 3.13 or higher must be installed.
*   **Package Manager**: `pip` or `uv` for Python package management.
*   **Git**: For cloning the RAG System repository.
*   **System Resources**: Refer to the [System Requirements](../../deployment_guide.md#system-requirements) in the main deployment guide for CPU, memory, and storage recommendations.
*   **External Dependencies**: Depending on your chosen configuration, you might need to install and run external services like:
    *   **ChromaDB**: If using a local persistent Chroma instance, ensure it's running and accessible.
    *   **Redis**: If enabling caching, a Redis server should be installed and configured.

## Installation

1.  **Clone the RAG System Repository:**
    ```bash
git clone <repository-url>
cd rag-engine # Or your project root directory
    ```

2.  **Set up a Python Virtual Environment (Recommended):**
    ```bash
python3 -m venv .venv
source .venv/bin/activate
    ```

3.  **Install Python Dependencies:**
    ```bash
pip install -e .
# Or using uv:
uv sync
    ```

4.  **Configure Environment Variables:**
    Create a `.env` file based on `.env.example` and populate it with your API keys and other necessary configurations.
    ```bash
cp .env.example .env
# Edit .env file
    ```

For more detailed installation steps, refer to the [Installation Guide](../../getting-started/installation.md).

## Running the RAG Engine

Once installed, you can run the RAG Engine components directly.

### Starting the API Server

To run the RAG API server:

```bash
# Ensure your virtual environment is active
rag serve --host 0.0.0.0 --port 8000
```

This will start the FastAPI server, making the RAG API accessible on port 8000.

### Running CLI Commands

You can execute any CLI command directly:

```bash
rag status
rag index files /path/to/your/documents
rag query "What is the RAG system?"
```

### Running as a Background Service

For production environments, it's recommended to run the RAG API server as a background service using a process manager like `systemd` or `supervisor`.

#### Example `systemd` Service File (`/etc/systemd/system/rag-engine.service`)

```ini
[Unit]
Description=RAG Engine API Server
After=network.target

[Service]
User=raguser # Create a dedicated user for the service
Group=raguser
WorkingDirectory=/path/to/your/rag-engine-directory
ExecStart=/path/to/your/rag-engine-directory/.venv/bin/rag serve --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=rag-engine

[Install]
WantedBy=multi-user.target
```

**Steps to use `systemd`:**

1.  Replace `/path/to/your/rag-engine-directory` with the actual path to your RAG System installation.
2.  Create a dedicated user and group (e.g., `raguser`).
3.  Reload `systemd` and enable/start the service:
    ```bash
sudo systemctl daemon-reload
sudo systemctl enable rag-engine
sudo systemctl start rag-engine
    ```
4.  Check service status and logs:
    ```bash
sudo systemctl status rag-engine
sudo journalctl -u rag-engine -f
    ```

## Service Management

Once deployed as a service, you can manage it using your chosen process manager (e.g., `systemd` commands like `start`, `stop`, `restart`, `status`).

## Configuration

Refer to the [Configuration Reference](../configuration/configuration-reference.md) for details on configuring the RAG Engine, including LLM providers, vector stores, and performance settings.

## Monitoring

For monitoring, ensure your system's Prometheus agent (Node Exporter) is running and configured to scrape the RAG Engine's metrics endpoint (if exposed directly or via a proxy). Refer to the [Monitoring Guide](../operations/monitoring.md) for more details.
