# RAG Engine Project Map

This document provides a visual map of the RAG Engine project structure and documentation organization.

## Project Structure

```
RAG/
├── config/                           # Configuration files
│   ├── config.development.yaml       # Development environment config
│   ├── config.production.yaml        # Production environment config
│   ├── config.testing.yaml           # Testing environment config
│   └── example_config.yaml           # Example configuration template
├── data/                             # Data files and datasets
├── deployment/                       # Deployment configurations
│   ├── grafana/                      # Grafana dashboards and datasources
│   ├── prometheus.yml                # Prometheus configuration
│   ├── prometheus.prod.yml           # Production Prometheus config
│   ├── nginx.conf                    # Development Nginx configuration
│   └── nginx.prod.conf               # Production Nginx configuration
├── docs/                             # Documentation
│   ├── api-reference/                # API documentation
│   │   ├── cli-reference.md          # CLI commands and options
│   │   ├── python-sdk.md             # Python SDK reference
│   │   ├── rest-api.md               # REST API endpoints
│   │   └── websocket-api.md          # WebSocket API
│   ├── architecture/                 # Architecture documentation
│   │   ├── components.md             # System components
│   │   ├── data-flow.md              # Data flow diagrams
│   │   ├── design-patterns.md        # Design patterns used
│   │   └── overview.md               # Architecture overview
│   ├── configuration/                # Configuration guides
│   │   ├── configuration-reference.md# Configuration options
│   │   ├── environment-setup.md      # Environment variables
│   │   ├── performance-tuning.md     # Performance optimization
│   │   └── security.md               # Security configuration
│   ├── deployment/                   # Deployment guides
│   │   ├── bare-metal.md             # Bare metal deployment
│   │   ├── cloud-providers.md        # Cloud provider deployment
│   │   ├── docker.md                 # Docker deployment
│   │   └── kubernetes.md             # Kubernetes deployment
│   ├── development/                  # Development guides
│   │   ├── architecture-decisions/   # Architecture Decision Records
│   │   ├── coding-standards.md       # Coding standards
│   │   ├── contributing.md           # Contributing guide
│   │   ├── development-setup.md      # Development setup
│   │   ├── error_handling_and_resilience.md # Error handling
│   │   ├── testing.md                # Testing guide
│   │   └── README.md                 # Development documentation overview
│   ├── examples/                     # Example use cases
│   │   ├── advanced-examples/        # Advanced feature examples
│   │   ├── basic-examples/           # Basic usage examples
│   │   ├── integration-examples/     # Integration examples
│   │   └── README.md                 # Examples overview
│   ├── getting-started/              # Getting started guides
│   │   ├── first-steps.md            # First steps guide
│   │   ├── installation.md           # Installation guide
│   │   └── quick-start.md            # Quick start tutorial
│   ├── operations/                   # Operations guides
│   │   ├── maintenance.md            # Maintenance procedures
│   │   ├── monitoring.md             # Monitoring and health checks
│   │   ├── scaling.md                # Scaling strategies
│   │   └── troubleshooting.md        # Troubleshooting guide
│   ├── user-guides/                  # User guides
│   │   ├── advanced-features.md      # Advanced features guide
│   │   ├── basic-usage.md            # Basic usage guide
│   │   └── README.md                 # User guides overview
│   ├── PROJECT_MAP.md                # This document
│   └── README.md                     # Documentation overview
├── examples/                         # Example scripts and notebooks
├── logs/                             # Log files
├── REPORTS/                          # Generated reports
├── scripts/                          # Utility scripts
├── src/                              # Source code
│   └── rag_engine/                   # Main RAG engine
│       ├── api/                      # REST API components
│       ├── cli/                      # Command-line interface
│       ├── common/                   # Shared utilities
│       ├── core/                     # Core components and interfaces
│       ├── evaluation/               # Evaluation tools
│       ├── generation/               # Text generation components
│       ├── indexing/                 # Document indexing
│       ├── query/                    # Query processing techniques
│       ├── retrieval/                # Document retrieval
│       ├── routing/                  # Query routing
│       └── __init__.py
├── systemDocs/                       # System documentation
├── tests/                            # Test suite
├── workplan/                         # Project planning documents
├── main.py                           # Main entry point
├── rag_cli.py                        # CLI interface entry point
├── run_examples.py                   # Examples runner
├── pyproject.toml                    # Project configuration and dependencies
├── uv.lock                           # Dependency lock file
├── Dockerfile                        # Production Docker image
├── Dockerfile.dev                    # Development Docker image
├── docker-compose.yml                # Development environment setup
├── docker-compose.prod.yml           # Production environment setup
├── .env.example                      # Example environment variables
├── .gitignore                        # Git ignore patterns
├── .python-version                   # Python version specification
└── README.md                         # Project README
```

## Documentation Organization

The documentation is organized into two main sections:

### 🚀 User Documentation
For users who want to deploy, configure, and use the RAG Engine:
- Getting Started guides
- User Guides (basic and advanced usage)
- API Reference (CLI, REST, WebSocket, Python SDK)
- Configuration guides
- Deployment instructions
- Operations guides (monitoring, troubleshooting, maintenance)
- Examples

### 💻 Developer Documentation
For developers who want to contribute to or extend the RAG Engine:
- Contributing guide
- Development setup
- Coding standards
- Testing procedures
- Architecture documentation
- Error handling and resilience patterns
- Architecture Decision Records (ADRs)

## Key Documentation Paths

### For New Users
1. [Getting Started](./getting-started/) → [Installation](./getting-started/installation.md)
2. [Getting Started](./getting-started/) → [Quick Start](./getting-started/quick-start.md)
3. [User Guides](./user-guides/) → [Basic Usage](./user-guides/basic-usage.md)

### For Advanced Users
1. [User Guides](./user-guides/) → [Advanced Features](./user-guides/advanced-features.md)
2. [API Reference](./api-reference/) → [REST API](./api-reference/rest-api.md)
3. [Configuration](./configuration/) → [Configuration Reference](./configuration/configuration-reference.md)

### For Developers
1. [Development](./development/) → [Contributing Guide](./development/contributing.md)
2. [Development](./development/) → [Development Setup](./development/development-setup.md)
3. [Architecture](./architecture/) → [Overview](./architecture/overview.md)

### For Operations
1. [Deployment](./deployment/) → [Docker](./deployment/docker.md)
2. [Operations](./operations/) → [Monitoring](./operations/monitoring.md)
3. [Operations](./operations/) → [Troubleshooting](./operations/troubleshooting.md)