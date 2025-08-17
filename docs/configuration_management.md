# Configuration Management System

## Overview

The RAG system now includes a comprehensive configuration management system that provides:

- **Schema-based validation** using Pydantic for type safety and constraint checking
- **Environment-specific configurations** for development, testing, staging, and production
- **Multiple configuration sources** with proper precedence (files, environment variables)
- **YAML and JSON support** for configuration files
- **Comprehensive validation** with detailed error messages
- **CLI tools** for configuration management
- **Extensive test coverage** with 31 unit tests

## Key Features

### 1. Configuration Schema Validation

The system uses Pydantic models to validate configuration:

```python
from rag_engine.core.config import PipelineConfigSchema, ConfigurationManager

# Automatic validation
config_manager = ConfigurationManager()
config = config_manager.load_config("config.yaml")  # Validates automatically
```

### 2. Environment-Specific Configurations

Support for environment-specific configuration files:

```
config/
├── config.yaml              # Base configuration
├── config.development.yaml  # Development overrides
├── config.production.yaml   # Production overrides
└── config.testing.yaml      # Testing overrides
```

### 3. Configuration Precedence

Configuration values are loaded in order of precedence:

1. **Base configuration file** (e.g., `config.yaml`)
2. **Environment-specific file** (e.g., `config.production.yaml`)
3. **Environment variables** (e.g., `RAG_LLM_PROVIDER=openai`)

### 4. Comprehensive Validation

The system validates:

- **Enum values** for providers and strategies
- **Numeric ranges** for temperatures, chunk sizes, etc.
- **Cross-field validation** (e.g., chunk overlap < chunk size)
- **Required API keys** based on selected providers
- **Environment-specific warnings** (e.g., debug logging in production)

## Usage Examples

### Basic Configuration Loading

```python
from rag_engine.core.config import ConfigurationManager

# Load default configuration
config_manager = ConfigurationManager()
config = config_manager.load_config()

# Load from specific file
config_manager = ConfigurationManager("config/my_config.yaml")
config = config_manager.load_config()

# Load with environment
config_manager = ConfigurationManager("config/config.yaml", "production")
config = config_manager.load_config()
```

### Environment Variables

Set configuration via environment variables:

```bash
export RAG_LLM_PROVIDER=openai
export RAG_LLM_MODEL=gpt-4
export RAG_TEMPERATURE=0.7
export RAG_CHUNK_SIZE=1500
export OPENAI_API_KEY=your-api-key
```

### Configuration Validation

```python
# Validate a configuration file
config_manager = ConfigurationManager()
is_valid = config_manager.validate_config_file("config.yaml")

# Get validation schema
schema = config_manager.get_config_schema()
```

### Saving Configurations

```python
# Save configuration (secrets excluded by default)
config_manager.save_config(config, "output.yaml")

# Save with secrets included
config_manager.save_config(config, "output.yaml", exclude_secrets=False)
```

### Merging Configurations

```python
# Merge two configuration dictionaries
base_config = {"llm_provider": "google", "temperature": 0.0}
override_config = {"temperature": 0.5, "chunk_size": 1500}
merged = config_manager.merge_configs(base_config, override_config)
```

## CLI Tool

Use the configuration management CLI tool:

```bash
# Validate a configuration file
python scripts/config_manager.py validate config/example_config.yaml

# Show current configuration
python scripts/config_manager.py show --config-file config/config.yaml

# Show configuration for specific environment
python scripts/config_manager.py show --config-file config/config.yaml --environment production

# Create a new configuration file
python scripts/config_manager.py create --output config/new_config.yaml --llm-provider openai

# List available environments
python scripts/config_manager.py list-environments --config-dir config/

# Show configuration schema
python scripts/config_manager.py schema
```

## Configuration Options

### Core Configuration

- **Environment**: `development`, `testing`, `staging`, `production`
- **LLM Provider**: `google`, `openai`, `anthropic`, `local`, `ollama`
- **Embedding Provider**: `openai`, `huggingface`
- **Vector Store**: `chroma`, `pinecone`, `weaviate`
- **Indexing Strategy**: `basic`, `multi_representation`, `colbert`, `raptor`

### Advanced Features

- **Query Processing**: Multi-query, RAG-Fusion, decomposition, step-back, HyDE
- **Routing**: Logical, semantic, hybrid routing strategies
- **Retrieval**: Hybrid retrieval, re-ranking, self-correction
- **Production**: Logging, metrics, caching, monitoring

### Validation Rules

- **Temperature**: 0.0 to 2.0
- **Chunk Size**: 1 to 10,000 characters
- **Chunk Overlap**: Must be less than chunk size
- **Retrieval K**: 1 to 100 documents
- **Weights**: Vector and keyword weights must sum to 1.0

## Environment-Specific Examples

### Development Configuration

```yaml
environment: "development"
log_level: "DEBUG"
chunk_size: 500
retrieval_k: 3
enable_metrics: false
enable_caching: false
```

### Production Configuration

```yaml
environment: "production"
log_level: "WARNING"
chunk_size: 1000
retrieval_k: 10
enable_metrics: true
enable_caching: true
enable_hybrid_retrieval: true
use_reranking: true
```

### Testing Configuration

```yaml
environment: "testing"
log_level: "ERROR"
chunk_size: 200
retrieval_k: 2
enable_metrics: false
vector_store_config:
  persist_directory: null  # In-memory for testing
```

## Error Handling

The system provides comprehensive error handling:

- **Configuration file not found**
- **Invalid YAML/JSON syntax**
- **Schema validation errors** with detailed messages
- **Missing required API keys** (warnings)
- **Invalid value ranges** with specific constraints
- **Cross-field validation errors**

## Testing

The configuration system includes 31 comprehensive unit tests covering:

- **Schema validation** for all configuration options
- **File loading** from YAML and JSON formats
- **Environment variable** processing
- **Environment-specific** configuration loading
- **Configuration precedence** and merging
- **Error handling** for various failure scenarios
- **Configuration saving** with and without secrets
- **CLI tool functionality**

Run tests with:

```bash
uv run python -m pytest tests/core/test_configuration_manager.py -v
```

## Integration

The configuration system is fully integrated with the RAG engine:

```python
from rag_engine import RAGEngine, ConfigurationManager

# Load configuration and create engine
config_manager = ConfigurationManager("config/production.yaml", "production")
config = config_manager.load_config()
engine = RAGEngine(config)
```

This configuration management system provides a robust foundation for deploying the RAG system across different environments with proper validation, security, and maintainability.