# RAG Engine Configuration Guide

This guide covers all configuration options for the RAG Engine, including environment-specific settings, performance tuning, and best practices.

## Table of Contents

1. [Configuration Overview](#configuration-overview)
2. [Environment Variables](#environment-variables)
3. [Configuration Files](#configuration-files)
4. [Core Configuration](#core-configuration)
5. [Provider Configuration](#provider-configuration)
6. [Performance Tuning](#performance-tuning)
7. [Security Configuration](#security-configuration)
8. [Environment-Specific Settings](#environment-specific-settings)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Configuration Overview

The RAG Engine uses a hierarchical configuration system:

1. **Default values** - Built-in defaults
2. **Configuration files** - YAML/JSON files in `config/` directory
3. **Environment variables** - Override file settings
4. **Runtime parameters** - Override all other settings

### Configuration Priority (highest to lowest)
1. Runtime parameters (API calls, CLI arguments)
2. Environment variables
3. Configuration files
4. Default values

## Environment Variables

### Core Settings

```bash
# Environment
ENVIRONMENT=production              # Environment type (development, testing, production)
LOG_LEVEL=INFO                     # Logging level (DEBUG, INFO, WARNING, ERROR)
DEBUG=false                        # Enable debug mode

# API Configuration
API_HOST=0.0.0.0                   # API server host
API_PORT=8000                      # API server port
HEALTH_PORT=8089                   # Health check port
WORKERS=4                          # Number of worker processes

# Security
API_KEY=your_api_key_here          # API authentication key
JWT_SECRET=your_jwt_secret         # JWT signing secret
CORS_ORIGINS=*                     # CORS allowed origins
```

### Provider API Keys

```bash
# LLM Providers
GOOGLE_API_KEY=your_google_key     # Google AI API key
OPENAI_API_KEY=your_openai_key     # OpenAI API key
ANTHROPIC_API_KEY=your_anthropic_key # Anthropic API key
COHERE_API_KEY=your_cohere_key     # Cohere API key (for reranking)

# Vector Store Providers
PINECONE_API_KEY=your_pinecone_key # Pinecone API key
PINECONE_ENVIRONMENT=us-west1-gcp  # Pinecone environment
WEAVIATE_URL=http://localhost:8080 # Weaviate instance URL
WEAVIATE_API_KEY=your_weaviate_key # Weaviate API key
```

### Database and Cache

```bash
# Vector Database
CHROMA_HOST=localhost              # Chroma database host
CHROMA_PORT=8001                   # Chroma database port
CHROMA_PERSIST_DIRECTORY=./chroma_db # Chroma data directory

# Cache
REDIS_URL=redis://localhost:6379   # Redis connection URL
CACHE_TTL=3600                     # Cache time-to-live (seconds)
ENABLE_CACHING=true                # Enable/disable caching
```

### Performance Settings

```bash
# Processing
MAX_CONCURRENT_REQUESTS=100        # Maximum concurrent requests
REQUEST_TIMEOUT=300                # Request timeout (seconds)
CHUNK_SIZE=1000                    # Default chunk size for documents
CHUNK_OVERLAP=200                  # Overlap between chunks
RETRIEVAL_K=5                      # Default number of documents to retrieve

# Memory and Resources
MAX_MEMORY_USAGE=4GB               # Maximum memory usage
ENABLE_GPU=false                   # Enable GPU acceleration
BATCH_SIZE=32                      # Batch size for processing
```

### Monitoring and Observability

```bash
# Monitoring
ENABLE_MONITORING=true             # Enable metrics collection
METRICS_PORT=8089                  # Metrics endpoint port
PROMETHEUS_ENABLED=true            # Enable Prometheus metrics
HEALTH_CHECK_INTERVAL=30           # Health check interval (seconds)

# Logging
LOG_FORMAT=json                    # Log format (json, text)
LOG_FILE=/app/logs/rag-engine.log  # Log file path
ENABLE_REQUEST_LOGGING=true        # Log all requests
```

## Configuration Files

### File Structure

```
config/
├── config.development.yaml        # Development environment
├── config.testing.yaml           # Testing environment
├── config.production.yaml        # Production environment
├── config.performance.yaml       # Performance testing
└── example_config.yaml           # Example configuration
```

### Base Configuration Format

```yaml
# config/config.production.yaml
environment: production
debug: false

# API Configuration
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  cors_origins: ["https://yourdomain.com"]
  rate_limit:
    requests_per_minute: 100
    burst_size: 20

# LLM Configuration
llm:
  provider: "google"
  model: "gemini-2.0-flash-lite"
  temperature: 0.1
  max_tokens: 1000
  timeout: 30

# Embedding Configuration
embedding:
  provider: "google"
  model: "models/embedding-001"
  batch_size: 100

# Vector Store Configuration
vector_store:
  provider: "chroma"
  host: "chroma-service"
  port: 8000
  collection_name: "rag_documents"
  
# Indexing Configuration
indexing:
  strategy: "basic"
  chunk_size: 1000
  chunk_overlap: 200
  max_chunk_size: 2000
  min_chunk_size: 100

# Retrieval Configuration
retrieval:
  k: 5
  score_threshold: 0.7
  enable_reranking: true
  reranker_model: "cohere"
  max_retrieved_docs: 20

# Query Processing
query_processing:
  strategies: ["multi_query", "rag_fusion"]
  default_strategy: "basic"
  enable_query_expansion: true
  max_query_length: 1000

# Evaluation Configuration
evaluation:
  frameworks: ["custom", "ragas"]
  metrics: ["faithfulness", "relevancy", "correctness"]
  batch_size: 10

# Caching Configuration
cache:
  enabled: true
  provider: "redis"
  ttl: 3600
  max_size: "1GB"

# Monitoring Configuration
monitoring:
  enabled: true
  metrics_port: 8089
  health_check_interval: 30
  log_level: "INFO"
  
# Security Configuration
security:
  enable_auth: true
  jwt_expiry: 3600
  rate_limiting: true
  cors_enabled: true
```

## Core Configuration

### PipelineConfig Class

The main configuration class with all available options:

```python
from rag_engine.core.config import PipelineConfig, LLMProvider, VectorStore

config = PipelineConfig(
    # Environment
    environment="production",
    debug=False,
    
    # LLM Settings
    llm_provider=LLMProvider.GOOGLE,
    llm_model="gemini-2.0-flash-lite",
    temperature=0.1,
    max_tokens=1000,
    
    # Embedding Settings
    embedding_provider="google",
    embedding_model="models/embedding-001",
    
    # Vector Store Settings
    vector_store=VectorStore.CHROMA,
    vector_store_config={
        "host": "localhost",
        "port": 8001,
        "collection_name": "documents"
    },
    
    # Indexing Settings
    indexing_strategy="basic",
    chunk_size=1000,
    chunk_overlap=200,
    
    # Retrieval Settings
    retrieval_k=5,
    use_reranking=True,
    reranker_model="cohere",
    
    # Query Processing
    query_strategies=["multi_query", "rag_fusion"],
    
    # Performance
    max_concurrent_requests=100,
    request_timeout=300,
    
    # Monitoring
    enable_monitoring=True,
    enable_logging=True,
    log_level="INFO"
)
```

## Provider Configuration

### LLM Providers

#### Google AI (Gemini)
```yaml
llm:
  provider: "google"
  model: "gemini-2.0-flash-lite"  # or "gemini-2.0-flash-lite"
  temperature: 0.1
  max_tokens: 1000
  top_p: 0.9
  top_k: 40
```

#### OpenAI
```yaml
llm:
  provider: "openai"
  model: "gpt-3.5-turbo"  # or "gpt-4"
  temperature: 0.1
  max_tokens: 1000
  frequency_penalty: 0.0
  presence_penalty: 0.0
```

#### Anthropic
```yaml
llm:
  provider: "anthropic"
  model: "claude-3-sonnet-20240229"
  temperature: 0.1
  max_tokens: 1000
```

### Vector Store Providers

#### Chroma
```yaml
vector_store:
  provider: "chroma"
  host: "localhost"
  port: 8001
  collection_name: "rag_documents"
  persist_directory: "./chroma_db"
  distance_function: "cosine"
```

#### Pinecone
```yaml
vector_store:
  provider: "pinecone"
  api_key: "${PINECONE_API_KEY}"
  environment: "us-west1-gcp"
  index_name: "rag-index"
  dimension: 1536
  metric: "cosine"
```

#### Weaviate
```yaml
vector_store:
  provider: "weaviate"
  url: "http://localhost:8080"
  api_key: "${WEAVIATE_API_KEY}"
  class_name: "Document"
  vectorizer: "text2vec-openai"
```

### Embedding Providers

#### Google Embeddings
```yaml
embedding:
  provider: "google"
  model: "models/embedding-001"
  batch_size: 100
  max_retries: 3
```

#### OpenAI Embeddings
```yaml
embedding:
  provider: "openai"
  model: "text-embedding-ada-002"
  batch_size: 100
  max_retries: 3
```

#### Hugging Face Embeddings
```yaml
embedding:
  provider: "huggingface"
  model: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu"  # or "cuda"
  batch_size: 32
```

## Performance Tuning

### Indexing Performance

```yaml
indexing:
  # Chunk size affects retrieval precision vs context
  chunk_size: 1000          # Larger = more context, fewer chunks
  chunk_overlap: 200        # Overlap prevents information loss
  
  # Batch processing
  batch_size: 100           # Documents processed together
  max_workers: 4            # Parallel processing threads
  
  # Memory management
  max_memory_per_worker: "2GB"
  enable_streaming: true    # Process large files in chunks
```

### Retrieval Performance

```yaml
retrieval:
  # Number of documents to retrieve
  k: 5                      # Balance between relevance and speed
  
  # Scoring and filtering
  score_threshold: 0.7      # Minimum relevance score
  max_retrieved_docs: 20    # Maximum before reranking
  
  # Reranking (improves quality but adds latency)
  enable_reranking: true
  reranker_model: "cohere"
  rerank_top_k: 10
  
  # Caching
  enable_result_caching: true
  cache_ttl: 3600
```

### Generation Performance

```yaml
llm:
  # Model selection affects speed vs quality
  model: "gemini-2.0-flash-lite"  # Fast model
  # model: "gemini-2.0-flash-lite"   # High-quality model
  
  # Generation parameters
  max_tokens: 500           # Shorter responses are faster
  temperature: 0.1          # Lower temperature is more deterministic
  
  # Timeout and retries
  timeout: 30               # Request timeout
  max_retries: 3            # Retry failed requests
  
  # Batching (if supported)
  enable_batching: false
  batch_size: 10
```

### System Performance

```yaml
# API Server
api:
  workers: 4                # Number of worker processes
  worker_class: "uvicorn.workers.UvicornWorker"
  max_requests: 1000        # Requests per worker before restart
  timeout: 300              # Worker timeout
  
# Concurrency
concurrency:
  max_concurrent_requests: 100
  max_concurrent_indexing: 10
  max_concurrent_queries: 50
  
# Memory Management
memory:
  max_memory_usage: "4GB"
  gc_threshold: 0.8         # Trigger garbage collection at 80%
  enable_memory_profiling: false

# Connection Pooling
connections:
  vector_store_pool_size: 10
  llm_pool_size: 5
  cache_pool_size: 20
```

## Security Configuration

### Authentication and Authorization

```yaml
security:
  # API Authentication
  enable_auth: true
  auth_type: "api_key"      # or "jwt", "oauth"
  api_key_header: "Authorization"
  
  # JWT Configuration
  jwt:
    secret_key: "${JWT_SECRET}"
    algorithm: "HS256"
    expiry: 3600            # Token expiry in seconds
    
  # Rate Limiting
  rate_limiting:
    enabled: true
    requests_per_minute: 100
    burst_size: 20
    
  # CORS
  cors:
    enabled: true
    origins: ["https://yourdomain.com"]
    methods: ["GET", "POST"]
    headers: ["Content-Type", "Authorization"]
```

### Data Security

```yaml
security:
  # Encryption
  encryption:
    enable_at_rest: true
    enable_in_transit: true
    algorithm: "AES-256"
    
  # Data Privacy
  privacy:
    enable_pii_detection: true
    mask_sensitive_data: true
    retention_days: 90
    
  # Audit Logging
  audit:
    enabled: true
    log_requests: true
    log_responses: false    # Don't log response content
    log_file: "/app/logs/audit.log"
```

### Network Security

```yaml
security:
  # TLS/SSL
  tls:
    enabled: true
    cert_file: "/etc/ssl/certs/server.crt"
    key_file: "/etc/ssl/private/server.key"
    min_version: "TLSv1.2"
    
  # Firewall Rules
  firewall:
    allowed_ips: ["10.0.0.0/8", "192.168.0.0/16"]
    blocked_ips: []
    
  # Request Validation
  validation:
    max_request_size: "10MB"
    max_query_length: 1000
    sanitize_input: true
```

## Environment-Specific Settings

### Development Environment

```yaml
# config/config.development.yaml
environment: development
debug: true

api:
  host: "127.0.0.1"
  port: 8000
  workers: 1
  reload: true

llm:
  provider: "google"
  model: "gemini-2.0-flash-lite"
  temperature: 0.2

monitoring:
  enabled: true
  log_level: "DEBUG"

security:
  enable_auth: false
  rate_limiting: false
```

### Testing Environment

```yaml
# config/config.testing.yaml
environment: testing
debug: false

api:
  host: "127.0.0.1"
  port: 8001
  workers: 1

llm:
  provider: "mock"          # Use mock provider for testing
  model: "test-model"

vector_store:
  provider: "memory"        # In-memory store for tests

cache:
  enabled: false            # Disable caching for consistent tests

monitoring:
  enabled: false
```

### Production Environment

```yaml
# config/config.production.yaml
environment: production
debug: false

api:
  host: "0.0.0.0"
  port: 8000
  workers: 4

llm:
  provider: "google"
  model: "gemini-2.0-flash-lite"
  temperature: 0.1

vector_store:
  provider: "pinecone"      # Managed vector store

cache:
  enabled: true
  provider: "redis"

monitoring:
  enabled: true
  log_level: "WARNING"

security:
  enable_auth: true
  rate_limiting: true
  tls:
    enabled: true
```

## Best Practices

### Configuration Management

1. **Use Environment Variables for Secrets**
   ```yaml
   # Good
   api_key: "${GOOGLE_API_KEY}"
   
   # Bad
   api_key: "actual-api-key-here"
   ```

2. **Separate Configs by Environment**
   ```
   config/
   ├── config.development.yaml
   ├── config.production.yaml
   └── config.testing.yaml
   ```

3. **Use Configuration Validation**
   ```python
   from rag_engine.core.config import ConfigurationManager
   
   config_manager = ConfigurationManager()
   config = config_manager.load_config()
   config_manager.validate_config(config)
   ```

### Performance Optimization

1. **Chunk Size Optimization**
   ```yaml
   # For factual Q&A
   chunk_size: 500
   chunk_overlap: 50
   
   # For conversational AI
   chunk_size: 1000
   chunk_overlap: 200
   
   # For long-form content
   chunk_size: 1500
   chunk_overlap: 300
   ```

2. **Retrieval Tuning**
   ```yaml
   # High precision
   retrieval:
     k: 3
     score_threshold: 0.8
     enable_reranking: true
   
   # High recall
   retrieval:
     k: 10
     score_threshold: 0.5
     enable_reranking: false
   ```

3. **Caching Strategy**
   ```yaml
   cache:
     # Cache frequent queries
     query_cache_ttl: 3600
     
     # Cache embeddings
     embedding_cache_ttl: 86400
     
     # Cache retrieval results
     retrieval_cache_ttl: 1800
   ```

### Security Best Practices

1. **API Key Management**
   ```bash
   # Use environment variables
   export GOOGLE_API_KEY="your-key-here"
   
   # Or use secret management systems
   # AWS Secrets Manager, HashiCorp Vault, etc.
   ```

2. **Network Security**
   ```yaml
   security:
     # Enable HTTPS in production
     tls:
       enabled: true
     
     # Restrict CORS origins
     cors:
       origins: ["https://yourdomain.com"]
     
     # Enable rate limiting
     rate_limiting:
       enabled: true
   ```

3. **Data Privacy**
   ```yaml
   privacy:
     # Enable PII detection
     enable_pii_detection: true
     
     # Set data retention
     retention_days: 90
     
     # Enable audit logging
     audit_logging: true
   ```

## Troubleshooting

### Common Configuration Issues

1. **API Key Not Found**
   ```
   Error: Google API key not found
   
   Solution:
   - Set GOOGLE_API_KEY environment variable
   - Or add to configuration file:
     llm:
       api_key: "${GOOGLE_API_KEY}"
   ```

2. **Vector Store Connection Failed**
   ```
   Error: Failed to connect to Chroma at localhost:8001
   
   Solution:
   - Check if Chroma is running
   - Verify host and port in configuration
   - Check network connectivity
   ```

3. **Memory Issues**
   ```
   Error: Out of memory during indexing
   
   Solution:
   - Reduce chunk_size
   - Reduce batch_size
   - Increase max_memory_usage
   - Enable streaming for large files
   ```

### Configuration Validation

```python
# Validate configuration
from rag_engine.core.config import ConfigurationManager

config_manager = ConfigurationManager()

try:
    config = config_manager.load_config()
    config_manager.validate_config(config)
    print("✅ Configuration is valid")
except Exception as e:
    print(f"❌ Configuration error: {e}")
```

### Debug Configuration

```yaml
# Enable debug mode
debug: true
log_level: "DEBUG"

# Enable detailed logging
monitoring:
  log_requests: true
  log_responses: true
  log_performance: true

# Enable configuration dumping
dump_config_on_start: true
```

### Performance Monitoring

```yaml
# Monitor configuration impact
monitoring:
  track_config_changes: true
  performance_baseline: true
  alert_on_degradation: true
```

For more specific configuration scenarios, see the deployment guides and example configurations in the `config/` directory.
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