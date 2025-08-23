"""
Configuration management system for the RAG engine
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
import os
import yaml
import json
from pathlib import Path
from enum import Enum
import logging
from pydantic import BaseModel, Field, field_validator, ValidationError, ConfigDict


class Environment(str, Enum):
    """Supported deployment environments"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    GOOGLE = "google"
    OPENAI = "openai"
    LOCAL = "local"
    OLLAMA = "ollama"


class EmbeddingProvider(str, Enum):
    """Supported embedding providers"""
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


class VectorStore(str, Enum):
    """Supported vector stores"""
    CHROMA = "chroma"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"


class IndexingStrategy(str, Enum):
    """Supported indexing strategies"""
    BASIC = "basic"
    MULTI_REPRESENTATION = "multi_representation"
    COLBERT = "colbert"
    RAPTOR = "raptor"


class RoutingStrategy(str, Enum):
    """Supported routing strategies"""
    LOGICAL = "logical"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class LogLevel(str, Enum):
    """Supported log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ObservabilityProvider(str, Enum):
    """Supported observability providers"""
    LANGFUSE = "langfuse"
    PHOENIX = "phoenix"
    DISABLED = "disabled"


class PipelineConfigSchema(BaseModel):
    """Pydantic schema for configuration validation"""
    
    # Environment Configuration
    environment: Environment = Environment.DEVELOPMENT
    
    # LLM Configuration
    llm_provider: LLMProvider = LLMProvider.GOOGLE
    llm_model: str = "gemini-2.0-flash-lite"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    
    # Provider-specific API keys
    google_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    ollama_base_url: Optional[str] = None
    
    # Embedding Configuration
    embedding_provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    embedding_model: str = "text-embedding-ada-002"
    embedding_dimensions: Optional[int] = Field(default=None, gt=0)
    embedding_device: Optional[str] = None
    normalize_embeddings: bool = True
    
    # Vector Store Configuration
    vector_store: VectorStore = VectorStore.CHROMA
    vector_store_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Indexing Configuration
    indexing_strategy: IndexingStrategy = IndexingStrategy.BASIC
    chunk_size: int = Field(default=1000, gt=0, le=10000)
    chunk_overlap: int = Field(default=200, ge=0)
    
    # Query Processing Configuration
    query_strategies: List[str] = Field(default_factory=lambda: ["basic"])
    enable_multi_query: bool = False
    enable_rag_fusion: bool = False
    enable_decomposition: bool = False
    enable_step_back: bool = False
    enable_hyde: bool = False
    
    # Routing Configuration
    routing_enabled: bool = False
    routing_strategy: RoutingStrategy = RoutingStrategy.LOGICAL
    
    # Retrieval Configuration
    retrieval_k: int = Field(default=5, gt=0, le=100)
    use_reranking: bool = False
    
    reranker_top_k: int = Field(default=10, gt=0, le=100)
    
    # Hybrid Retrieval Configuration
    enable_hybrid_retrieval: bool = False
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    keyword_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    enable_long_context: bool = False
    context_window_size: int = Field(default=100000, gt=0)
    adaptive_retrieval: bool = False
    
    # Self-Correction Configuration
    enable_self_correction: bool = False
    relevance_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    factuality_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    min_relevant_docs: int = Field(default=2, gt=0)
    
    # Generation Configuration
    prompt_template: Optional[str] = None
    include_sources: bool = True
    
    # Evaluation Configuration
    evaluation_frameworks: List[str] = Field(default_factory=lambda: ["custom"])
    
    # Production Configuration
    enable_logging: bool = True
    log_level: LogLevel = LogLevel.INFO
    enable_metrics: bool = False
    enable_caching: bool = False
    
    # Observability Configuration
    observability_provider: ObservabilityProvider = ObservabilityProvider.DISABLED
    observability_enabled: bool = False
    langfuse_secret_key: Optional[str] = None
    langfuse_public_key: Optional[str] = None
    langfuse_host: Optional[str] = None
    phoenix_endpoint: Optional[str] = None
    phoenix_api_key: Optional[str] = None
    observability_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    trace_llm_calls: bool = True
    trace_retrieval: bool = True
    trace_embeddings: bool = True
    trace_evaluation: bool = True
    capture_inputs: bool = True
    capture_outputs: bool = True
    
    # Performance Configuration
    connection_pool_size: int = Field(default=10, gt=0, le=100)
    connection_timeout: float = Field(default=30.0, gt=0.0)
    cache_size: int = Field(default=1000, gt=0)
    cache_ttl: float = Field(default=3600.0, gt=0.0)
    async_enabled: bool = False
    max_concurrent_requests: int = Field(default=10, gt=0, le=1000)
    batch_size: int = Field(default=10, gt=0, le=1000)
    enable_connection_pooling: bool = True
    
    # API Configuration
    api_key_openai: Optional[str] = None
    
    @field_validator('chunk_overlap')
    @classmethod
    def validate_chunk_overlap(cls, v, info):
        """Validate that chunk overlap is less than chunk size"""
        if info.data and 'chunk_size' in info.data:
            chunk_size = info.data['chunk_size']
            if v >= chunk_size:
                raise ValueError(f"Chunk overlap ({v}) must be less than chunk size ({chunk_size})")
        return v
    
    @field_validator('keyword_weight')
    @classmethod
    def validate_weights_sum(cls, v, info):
        """Validate that vector and keyword weights sum to 1.0"""
        if info.data and 'vector_weight' in info.data:
            vector_weight = info.data['vector_weight']
            if abs(vector_weight + v - 1.0) > 0.001:  # Allow small floating point errors
                raise ValueError(f"Vector weight ({vector_weight}) and keyword weight ({v}) must sum to 1.0")
        return v
    
    model_config = ConfigDict(use_enum_values=True)


@dataclass
class PipelineConfig:
    """Main configuration for the RAG pipeline"""
    
    # Environment Configuration
    environment: str = "development"
    
    # LLM Configuration
    llm_provider: str = "google"  # Default to Google as per steering rules
    llm_model: str = "gemini-2.0-flash-lite"
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    
    # Provider-specific API keys
    google_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    ollama_base_url: Optional[str] = None
    
    # Embedding Configuration
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-ada-002"
    embedding_dimensions: Optional[int] = None
    embedding_device: Optional[str] = None  # For HuggingFace models
    normalize_embeddings: bool = True  # For HuggingFace models
    
    # Vector Store Configuration
    vector_store: str = "chroma"
    vector_store_config: Dict[str, Any] = field(default_factory=dict)
    
    # Indexing Configuration
    indexing_strategy: str = "basic"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Query Processing Configuration
    query_strategies: List[str] = field(default_factory=lambda: ["basic"])
    enable_multi_query: bool = False
    enable_rag_fusion: bool = False
    enable_decomposition: bool = False
    enable_step_back: bool = False
    enable_hyde: bool = False
    
    # Routing Configuration
    routing_enabled: bool = False
    routing_strategy: str = "logical"  # logical, semantic, or hybrid
    
    # Retrieval Configuration
    retrieval_k: int = 5
    use_reranking: bool = False
    
    reranker_top_k: int = 10
    
    # Hybrid Retrieval Configuration
    enable_hybrid_retrieval: bool = False
    vector_weight: float = 0.7
    keyword_weight: float = 0.3
    enable_long_context: bool = False
    context_window_size: int = 100000
    adaptive_retrieval: bool = False
    
    # Self-Correction Configuration
    enable_self_correction: bool = False
    relevance_threshold: float = 0.7
    factuality_threshold: float = 0.7
    min_relevant_docs: int = 2
    
    # Generation Configuration
    prompt_template: Optional[str] = None
    include_sources: bool = True
    
    # Evaluation Configuration
    evaluation_frameworks: List[str] = field(default_factory=lambda: ["custom"])
    
    # Production Configuration
    enable_logging: bool = True
    log_level: str = "INFO"
    enable_metrics: bool = False
    enable_caching: bool = False
    
    # Observability Configuration
    observability_provider: str = "disabled"
    observability_enabled: bool = False
    langfuse_secret_key: Optional[str] = None
    langfuse_public_key: Optional[str] = None
    langfuse_host: Optional[str] = None
    phoenix_endpoint: Optional[str] = None
    phoenix_api_key: Optional[str] = None
    observability_sample_rate: float = 1.0
    trace_llm_calls: bool = True
    trace_retrieval: bool = True
    trace_embeddings: bool = True
    trace_evaluation: bool = True
    capture_inputs: bool = True
    capture_outputs: bool = True
    
    # Performance Configuration
    connection_pool_size: int = 10
    connection_timeout: float = 30.0
    cache_size: int = 1000
    cache_ttl: float = 3600.0
    async_enabled: bool = False
    max_concurrent_requests: int = 10
    batch_size: int = 10
    enable_connection_pooling: bool = True
    
    # API Configuration
    api_key_openai: Optional[str] = None


class ConfigurationError(Exception):
    """Configuration-related errors"""
    pass


class ConfigurationManager:
    """Manages configuration loading and validation"""
    
    def __init__(self, config_path: Optional[str] = None, environment: Optional[str] = None):
        self.config_path = config_path
        self.environment = environment or os.getenv('RAG_ENVIRONMENT', 'development')
        self._config: Optional[PipelineConfig] = None
        self._logger = logging.getLogger(__name__)
    
    def load_config(self, config_path: Optional[str] = None, environment: Optional[str] = None) -> PipelineConfig:
        """Load configuration from file or environment"""
        if config_path:
            self.config_path = config_path
        if environment:
            self.environment = environment
        
        # Start with default configuration
        config_dict = {}
        
        # Load base configuration file
        if self.config_path:
            if Path(self.config_path).exists():
                config_dict = self._load_config_file(self.config_path)
                self._logger.info(f"Loaded base configuration from {self.config_path}")
            else:
                raise ConfigurationError(f"Configuration file not found: {self.config_path}")
        
        # Load environment-specific configuration
        env_specific_config = self._load_environment_specific_config()
        if env_specific_config:
            config_dict.update(env_specific_config)
            self._logger.info(f"Applied {self.environment} environment configuration")
        
        # Override with environment variables
        env_config = self._load_from_environment()
        if env_config:
            config_dict.update(env_config)
            self._logger.info("Applied environment variable overrides")
        
        # Validate configuration using Pydantic schema
        try:
            validated_config = PipelineConfigSchema(**config_dict)
            config_dict = validated_config.model_dump()
            self._logger.info("Configuration validation successful")
        except ValidationError as e:
            raise ConfigurationError(f"Configuration validation failed: {e}")
        
        # Create PipelineConfig instance
        self._config = PipelineConfig(**config_dict)
        
        # Additional custom validation
        self._validate_config(self._config)
        
        return self._config
    
    def _load_environment_specific_config(self) -> Dict[str, Any]:
        """Load environment-specific configuration file"""
        if not self.config_path:
            return {}
        
        base_path = Path(self.config_path)
        env_config_path = base_path.parent / f"{base_path.stem}.{self.environment}{base_path.suffix}"
        
        if env_config_path.exists():
            self._logger.info(f"Loading environment-specific config: {env_config_path}")
            return self._load_config_file(str(env_config_path))
        
        return {}
    
    def _load_config_file(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file"""
        path = Path(config_path)
        
        try:
            with open(path, 'r') as f:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                elif path.suffix.lower() == '.json':
                    return json.load(f) or {}
                else:
                    raise ConfigurationError(f"Unsupported config file format: {path.suffix}")
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in configuration file {config_path}: {e}")
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in configuration file {config_path}: {e}")
    
    def _load_from_environment(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        env_config = {}
        
        # Map environment variables to config keys
        env_mappings = {
            'RAG_ENVIRONMENT': 'environment',
            'RAG_LLM_PROVIDER': 'llm_provider',
            'RAG_LLM_MODEL': 'llm_model',
            'RAG_TEMPERATURE': ('temperature', float),
            'RAG_MAX_TOKENS': ('max_tokens', int),
            'RAG_EMBEDDING_PROVIDER': 'embedding_provider',
            'RAG_EMBEDDING_MODEL': 'embedding_model',
            'RAG_EMBEDDING_DIMENSIONS': ('embedding_dimensions', int),
            'RAG_EMBEDDING_DEVICE': 'embedding_device',
            'RAG_NORMALIZE_EMBEDDINGS': ('normalize_embeddings', lambda x: x.lower() == 'true'),
            'RAG_VECTOR_STORE': 'vector_store',
            'RAG_INDEXING_STRATEGY': 'indexing_strategy',
            'RAG_CHUNK_SIZE': ('chunk_size', int),
            'RAG_CHUNK_OVERLAP': ('chunk_overlap', int),
            'RAG_RETRIEVAL_K': ('retrieval_k', int),
            'RAG_USE_RERANKING': ('use_reranking', lambda x: x.lower() == 'true'),
            'RAG_RERANKER_MODEL': 'reranker_model',
            'RAG_RERANKER_TOP_K': ('reranker_top_k', int),
            'RAG_ENABLE_HYBRID_RETRIEVAL': ('enable_hybrid_retrieval', lambda x: x.lower() == 'true'),
            'RAG_VECTOR_WEIGHT': ('vector_weight', float),
            'RAG_KEYWORD_WEIGHT': ('keyword_weight', float),
            'RAG_ENABLE_SELF_CORRECTION': ('enable_self_correction', lambda x: x.lower() == 'true'),
            'RAG_RELEVANCE_THRESHOLD': ('relevance_threshold', float),
            'RAG_FACTUALITY_THRESHOLD': ('factuality_threshold', float),
            'RAG_ENABLE_LOGGING': ('enable_logging', lambda x: x.lower() == 'true'),
            'RAG_LOG_LEVEL': 'log_level',
            'RAG_ENABLE_METRICS': ('enable_metrics', lambda x: x.lower() == 'true'),
            'RAG_ENABLE_CACHING': ('enable_caching', lambda x: x.lower() == 'true'),
            'RAG_CONNECTION_POOL_SIZE': ('connection_pool_size', int),
            'RAG_CONNECTION_TIMEOUT': ('connection_timeout', float),
            'RAG_CACHE_SIZE': ('cache_size', int),
            'RAG_CACHE_TTL': ('cache_ttl', float),
            'RAG_ASYNC_ENABLED': ('async_enabled', lambda x: x.lower() == 'true'),
            'RAG_MAX_CONCURRENT_REQUESTS': ('max_concurrent_requests', int),
            'RAG_BATCH_SIZE': ('batch_size', int),
            'RAG_ENABLE_CONNECTION_POOLING': ('enable_connection_pooling', lambda x: x.lower() == 'true'),
            'GOOGLE_API_KEY': 'google_api_key',
            'OPENAI_API_KEY': 'openai_api_key',
            'ANTHROPIC_API_KEY': 'anthropic_api_key',
            'OLLAMA_BASE_URL': 'ollama_base_url',
            'COHERE_API_KEY': 'api_key_cohere',
            # Observability
            'RAG_OBSERVABILITY_PROVIDER': 'observability_provider',
            'RAG_OBSERVABILITY_ENABLED': ('observability_enabled', lambda x: x.lower() == 'true'),
            'LANGFUSE_SECRET_KEY': 'langfuse_secret_key',
            'LANGFUSE_PUBLIC_KEY': 'langfuse_public_key',
            'LANGFUSE_HOST': 'langfuse_host',
            'PHOENIX_ENDPOINT': 'phoenix_endpoint',
            'PHOENIX_API_KEY': 'phoenix_api_key',
            'RAG_OBSERVABILITY_SAMPLE_RATE': ('observability_sample_rate', float),
            'RAG_TRACE_LLM_CALLS': ('trace_llm_calls', lambda x: x.lower() == 'true'),
            'RAG_TRACE_RETRIEVAL': ('trace_retrieval', lambda x: x.lower() == 'true'),
            'RAG_TRACE_EMBEDDINGS': ('trace_embeddings', lambda x: x.lower() == 'true'),
            'RAG_TRACE_EVALUATION': ('trace_evaluation', lambda x: x.lower() == 'true'),
            'RAG_CAPTURE_INPUTS': ('capture_inputs', lambda x: x.lower() == 'true'),
            'RAG_CAPTURE_OUTPUTS': ('capture_outputs', lambda x: x.lower() == 'true'),
        }
        
        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    if isinstance(config_key, tuple):
                        key, type_func = config_key
                        env_config[key] = type_func(value)
                    else:
                        env_config[config_key] = value
                except (ValueError, TypeError) as e:
                    self._logger.warning(f"Invalid value for environment variable {env_var}: {value}. Error: {e}")
        
        return env_config
    
    def _validate_config(self, config: PipelineConfig) -> None:
        """Additional custom validation beyond Pydantic schema"""
        
        # Validate API keys based on providers
        if config.llm_provider == "google" and not config.google_api_key:
            self._logger.warning("Google API key not provided for Google LLM provider")
        elif config.llm_provider == "openai" and not config.openai_api_key:
            self._logger.warning("OpenAI API key not provided for OpenAI LLM provider")
        elif config.llm_provider == "anthropic" and not config.anthropic_api_key:
            self._logger.warning("Anthropic API key not provided for Anthropic LLM provider")
        
        if config.embedding_provider == "openai" and not config.openai_api_key:
            self._logger.warning("OpenAI API key not provided for OpenAI embedding provider")
        
        
        
        # Validate environment-specific settings
        if config.environment == "production":
            if not config.enable_logging:
                self._logger.warning("Logging is disabled in production environment")
            if config.log_level == "DEBUG":
                self._logger.warning("Debug logging enabled in production environment")
        
        # Validate feature combinations
        if config.enable_hybrid_retrieval and not (0 < config.vector_weight < 1):
            raise ConfigurationError("Vector weight must be between 0 and 1 for hybrid retrieval")
        
        if config.enable_self_correction and not config.use_reranking:
            self._logger.warning("Self-correction is enabled but reranking is disabled. Consider enabling reranking for better results.")
        
        self._logger.info("Configuration validation completed successfully")
    
    def save_config(self, config: PipelineConfig, output_path: str, exclude_secrets: bool = True) -> None:
        """Save configuration to file"""
        path = Path(output_path)
        
        # Convert dataclass to dict
        config_dict = {
            field.name: getattr(config, field.name)
            for field in config.__dataclass_fields__.values()
        }
        
        # Optionally exclude sensitive information
        if exclude_secrets:
            secret_fields = [
                'google_api_key', 'openai_api_key',
                'api_key_openai'
            ]
            for field in secret_fields:
                if field in config_dict and config_dict[field]:
                    config_dict[field] = "***REDACTED***"
        
        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save based on file extension
        try:
            with open(path, 'w') as f:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config_dict, f, default_flow_style=False, sort_keys=True)
                elif path.suffix.lower() == '.json':
                    json.dump(config_dict, f, indent=2, sort_keys=True)
                else:
                    raise ConfigurationError(f"Unsupported output format: {path.suffix}")
            
            self._logger.info(f"Configuration saved to {output_path}")
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration to {output_path}: {e}")
    
    def validate_config_file(self, config_path: str) -> bool:
        """Validate a configuration file without loading it"""
        try:
            config_dict = self._load_config_file(config_path)
            PipelineConfigSchema(**config_dict)
            return True
        except (ConfigurationError, ValidationError) as e:
            self._logger.error(f"Configuration validation failed for {config_path}: {e}")
            return False
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Get the configuration schema for documentation purposes"""
        return PipelineConfigSchema.model_json_schema()
    
    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries"""
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = self.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def create_environment_config(self, base_config_path: str, environment: str, overrides: Dict[str, Any]) -> str:
        """Create an environment-specific configuration file"""
        base_path = Path(base_config_path)
        env_config_path = base_path.parent / f"{base_path.stem}.{environment}{base_path.suffix}"
        
        # Load base configuration
        base_config = self._load_config_file(base_config_path)
        
        # Merge with overrides
        env_config = self.merge_configs(base_config, overrides)
        
        # Save environment-specific config
        self.save_config(PipelineConfig(**env_config), str(env_config_path))
        
        return str(env_config_path)
    
    def list_available_environments(self, config_dir: str) -> List[str]:
        """List available environment configurations in a directory"""
        config_dir_path = Path(config_dir)
        environments = set()
        
        for config_file in config_dir_path.glob("*.*.yaml"):
            parts = config_file.stem.split('.')
            if len(parts) >= 2:
                environments.add(parts[-1])
        
        for config_file in config_dir_path.glob("*.*.yml"):
            parts = config_file.stem.split('.')
            if len(parts) >= 2:
                environments.add(parts[-1])
        
        for config_file in config_dir_path.glob("*.*.json"):
            parts = config_file.stem.split('.')
            if len(parts) >= 2:
                environments.add(parts[-1])
        
        return sorted(list(environments))
    
    @property
    def config(self) -> PipelineConfig:
        """Get current configuration"""
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def reload_config(self) -> PipelineConfig:
        """Reload configuration from source"""
        self._config = None
        return self.load_config()