"""
Test configuration management system
"""

import pytest
import os
import tempfile
import json
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock
from pydantic import ValidationError

from src.rag_engine.core.config import (
    ConfigurationManager, 
    PipelineConfig, 
    PipelineConfigSchema,
    ConfigurationError,
    Environment,
    LLMProvider,
    EmbeddingProvider,
    VectorStore,
    IndexingStrategy,
    RoutingStrategy,
    LogLevel
)


class TestPipelineConfigSchema:
    """Test Pydantic schema validation"""
    
    def test_default_config_validation(self):
        """Test that default configuration passes validation"""
        config = PipelineConfigSchema()
        assert config.environment == Environment.DEVELOPMENT
        assert config.llm_provider == LLMProvider.GOOGLE
        assert config.embedding_provider == EmbeddingProvider.OPENAI
        assert config.vector_store == VectorStore.CHROMA
        assert config.indexing_strategy == IndexingStrategy.BASIC
    
    def test_valid_config_validation(self):
        """Test validation with valid values"""
        config_data = {
            "llm_provider": "openai",
            "temperature": 0.5,
            "chunk_size": 500,
            "chunk_overlap": 50,
            "retrieval_k": 10,
            "vector_weight": 0.8,
            "keyword_weight": 0.2
        }
        config = PipelineConfigSchema(**config_data)
        assert config.llm_provider == LLMProvider.OPENAI
        assert config.temperature == 0.5
        assert config.chunk_size == 500
    
    def test_invalid_temperature_validation(self):
        """Test validation fails for invalid temperature"""
        with pytest.raises(ValidationError):
            PipelineConfigSchema(temperature=3.0)  # Too high
        
        with pytest.raises(ValidationError):
            PipelineConfigSchema(temperature=-1.0)  # Too low
    
    def test_invalid_chunk_size_validation(self):
        """Test validation fails for invalid chunk size"""
        with pytest.raises(ValidationError):
            PipelineConfigSchema(chunk_size=0)  # Too small
        
        with pytest.raises(ValidationError):
            PipelineConfigSchema(chunk_size=20000)  # Too large
    
    def test_chunk_overlap_validation(self):
        """Test chunk overlap validation"""
        with pytest.raises(ValidationError):
            PipelineConfigSchema(chunk_size=1000, chunk_overlap=1000)  # Equal
        
        with pytest.raises(ValidationError):
            PipelineConfigSchema(chunk_size=1000, chunk_overlap=1500)  # Greater
    
    def test_weights_sum_validation(self):
        """Test that vector and keyword weights sum to 1.0"""
        with pytest.raises(ValidationError):
            PipelineConfigSchema(vector_weight=0.8, keyword_weight=0.3)  # Sum > 1
        
        # This should pass
        config = PipelineConfigSchema(vector_weight=0.6, keyword_weight=0.4)
        assert config.vector_weight == 0.6
        assert config.keyword_weight == 0.4
    
    def test_invalid_provider_validation(self):
        """Test validation fails for invalid providers"""
        with pytest.raises(ValidationError):
            PipelineConfigSchema(llm_provider="invalid_provider")
        
        with pytest.raises(ValidationError):
            PipelineConfigSchema(embedding_provider="invalid_embedding")
        
        with pytest.raises(ValidationError):
            PipelineConfigSchema(vector_store="invalid_store")


class TestConfigurationManager:
    """Test ConfigurationManager class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigurationManager()
    
    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_default_config_loading(self):
        """Test loading default configuration"""
        config = self.config_manager.load_config()
        
        assert isinstance(config, PipelineConfig)
        assert config.llm_provider == "google"
        assert config.embedding_provider == "openai"
        assert config.vector_store == "chroma"
        assert config.chunk_size == 1000
        assert config.retrieval_k == 5
    
    def test_yaml_config_loading(self):
        """Test loading configuration from YAML file"""
        config_data = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "temperature": 0.5,
            "chunk_size": 500,
            "retrieval_k": 10
        }
        
        config_path = Path(self.temp_dir) / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        config = self.config_manager.load_config(str(config_path))
        
        assert config.llm_provider == "openai"
        assert config.llm_model == "gpt-4"
        assert config.temperature == 0.5
        assert config.chunk_size == 500
        assert config.retrieval_k == 10
    
    def test_json_config_loading(self):
        """Test loading configuration from JSON file"""
        config_data = {
            "llm_provider": "anthropic",
            "llm_model": "claude-3-sonnet",
            "temperature": 0.3,
            "chunk_size": 800,
            "retrieval_k": 7
        }
        
        config_path = Path(self.temp_dir) / "test_config.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        config = self.config_manager.load_config(str(config_path))
        
        assert config.llm_provider == "anthropic"
        assert config.llm_model == "claude-3-sonnet"
        assert config.temperature == 0.3
        assert config.chunk_size == 800
        assert config.retrieval_k == 7
    
    def test_environment_variable_loading(self):
        """Test loading configuration from environment variables"""
        env_vars = {
            'RAG_LLM_PROVIDER': 'openai',
            'RAG_LLM_MODEL': 'gpt-4',
            'RAG_TEMPERATURE': '0.7',
            'RAG_CHUNK_SIZE': '1500',
            'RAG_RETRIEVAL_K': '8',
            'RAG_ENABLE_LOGGING': 'false',
            'OPENAI_API_KEY': 'test-key-123'
        }
        
        with patch.dict(os.environ, env_vars):
            config = self.config_manager.load_config()
        
        assert config.llm_provider == "openai"
        assert config.llm_model == "gpt-4"
        assert config.temperature == 0.7
        assert config.chunk_size == 1500
        assert config.retrieval_k == 8
        assert config.enable_logging is False
        assert config.openai_api_key == "test-key-123"
    
    def test_environment_specific_config_loading(self):
        """Test loading environment-specific configuration"""
        # Create base config
        base_config = {
            "llm_provider": "google",
            "chunk_size": 1000,
            "enable_logging": True,
            "log_level": "INFO"
        }
        
        base_config_path = Path(self.temp_dir) / "config.yaml"
        with open(base_config_path, 'w') as f:
            yaml.dump(base_config, f)
        
        # Create production-specific config
        prod_config = {
            "enable_metrics": True,
            "log_level": "WARNING",
            "enable_caching": True
        }
        
        prod_config_path = Path(self.temp_dir) / "config.production.yaml"
        with open(prod_config_path, 'w') as f:
            yaml.dump(prod_config, f)
        
        # Load with production environment
        config_manager = ConfigurationManager(str(base_config_path), "production")
        config = config_manager.load_config()
        
        assert config.llm_provider == "google"  # From base
        assert config.chunk_size == 1000  # From base
        assert config.enable_metrics is True  # From production
        assert config.log_level == "WARNING"  # Overridden by production
        assert config.enable_caching is True  # From production
    
    def test_config_priority_order(self):
        """Test that environment variables override file configuration"""
        # Create config file
        config_data = {
            "llm_provider": "google",
            "temperature": 0.0,
            "chunk_size": 1000
        }
        
        config_path = Path(self.temp_dir) / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        # Set environment variables that should override
        env_vars = {
            'RAG_LLM_PROVIDER': 'openai',
            'RAG_TEMPERATURE': '0.8'
        }
        
        with patch.dict(os.environ, env_vars):
            config = self.config_manager.load_config(str(config_path))
        
        assert config.llm_provider == "openai"  # From env var
        assert config.temperature == 0.8  # From env var
        assert config.chunk_size == 1000  # From file
    
    def test_invalid_config_file_format(self):
        """Test error handling for invalid config file format"""
        config_path = Path(self.temp_dir) / "test_config.txt"
        config_path.write_text("invalid config")
        
        with pytest.raises(ConfigurationError):
            self.config_manager.load_config(str(config_path))
    
    def test_invalid_yaml_content(self):
        """Test error handling for invalid YAML content"""
        config_path = Path(self.temp_dir) / "test_config.yaml"
        config_path.write_text("invalid: yaml: content: [")
        
        with pytest.raises(ConfigurationError):
            self.config_manager.load_config(str(config_path))
    
    def test_invalid_json_content(self):
        """Test error handling for invalid JSON content"""
        config_path = Path(self.temp_dir) / "test_config.json"
        config_path.write_text('{"invalid": json content}')
        
        with pytest.raises(ConfigurationError):
            self.config_manager.load_config(str(config_path))
    
    def test_missing_config_file(self):
        """Test error handling for missing config file"""
        # When a config file is specified but doesn't exist, it should raise an error
        config_manager = ConfigurationManager("/nonexistent/config.yaml")
        with pytest.raises(ConfigurationError):
            config_manager.load_config()
    
    def test_invalid_config_values(self):
        """Test validation error for invalid configuration values"""
        config_data = {
            "llm_provider": "invalid_provider",
            "temperature": 5.0,  # Too high
            "chunk_size": -100   # Negative
        }
        
        config_path = Path(self.temp_dir) / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        with pytest.raises(ConfigurationError):
            self.config_manager.load_config(str(config_path))
    
    def test_save_config_yaml(self):
        """Test saving configuration to YAML file"""
        config = PipelineConfig(
            llm_provider="openai",
            llm_model="gpt-4",
            temperature=0.5,
            chunk_size=800
        )
        
        output_path = Path(self.temp_dir) / "output_config.yaml"
        self.config_manager.save_config(config, str(output_path))
        
        assert output_path.exists()
        
        # Load and verify
        with open(output_path, 'r') as f:
            saved_data = yaml.safe_load(f)
        
        assert saved_data["llm_provider"] == "openai"
        assert saved_data["llm_model"] == "gpt-4"
        assert saved_data["temperature"] == 0.5
        assert saved_data["chunk_size"] == 800
    
    def test_save_config_json(self):
        """Test saving configuration to JSON file"""
        config = PipelineConfig(
            llm_provider="anthropic",
            temperature=0.3,
            chunk_size=1200
        )
        
        output_path = Path(self.temp_dir) / "output_config.json"
        self.config_manager.save_config(config, str(output_path))
        
        assert output_path.exists()
        
        # Load and verify
        with open(output_path, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data["llm_provider"] == "anthropic"
        assert saved_data["temperature"] == 0.3
        assert saved_data["chunk_size"] == 1200
    
    def test_save_config_exclude_secrets(self):
        """Test saving configuration with secrets excluded"""
        config = PipelineConfig(
            llm_provider="openai",
            openai_api_key="secret-key-123",
            api_key_cohere="cohere-secret"
        )
        
        output_path = Path(self.temp_dir) / "output_config.yaml"
        self.config_manager.save_config(config, str(output_path), exclude_secrets=True)
        
        with open(output_path, 'r') as f:
            saved_data = yaml.safe_load(f)
        
        assert saved_data["openai_api_key"] == "***REDACTED***"
        assert saved_data["api_key_cohere"] == "***REDACTED***"
    
    def test_save_config_include_secrets(self):
        """Test saving configuration with secrets included"""
        config = PipelineConfig(
            llm_provider="openai",
            openai_api_key="secret-key-123"
        )
        
        output_path = Path(self.temp_dir) / "output_config.yaml"
        self.config_manager.save_config(config, str(output_path), exclude_secrets=False)
        
        with open(output_path, 'r') as f:
            saved_data = yaml.safe_load(f)
        
        assert saved_data["openai_api_key"] == "secret-key-123"
    
    def test_validate_config_file_valid(self):
        """Test validation of valid configuration file"""
        config_data = {
            "llm_provider": "google",
            "temperature": 0.5,
            "chunk_size": 1000
        }
        
        config_path = Path(self.temp_dir) / "valid_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        assert self.config_manager.validate_config_file(str(config_path)) is True
    
    def test_validate_config_file_invalid(self):
        """Test validation of invalid configuration file"""
        config_data = {
            "llm_provider": "invalid_provider",
            "temperature": 5.0
        }
        
        config_path = Path(self.temp_dir) / "invalid_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        assert self.config_manager.validate_config_file(str(config_path)) is False
    
    def test_get_config_schema(self):
        """Test getting configuration schema"""
        schema = self.config_manager.get_config_schema()
        
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "llm_provider" in schema["properties"]
        assert "temperature" in schema["properties"]
    
    def test_merge_configs(self):
        """Test merging configuration dictionaries"""
        base_config = {
            "llm_provider": "google",
            "temperature": 0.0,
            "vector_store_config": {
                "host": "localhost",
                "port": 8000
            }
        }
        
        override_config = {
            "temperature": 0.5,
            "chunk_size": 1500,
            "vector_store_config": {
                "port": 9000,
                "ssl": True
            }
        }
        
        merged = self.config_manager.merge_configs(base_config, override_config)
        
        assert merged["llm_provider"] == "google"  # From base
        assert merged["temperature"] == 0.5  # Overridden
        assert merged["chunk_size"] == 1500  # New
        assert merged["vector_store_config"]["host"] == "localhost"  # From base
        assert merged["vector_store_config"]["port"] == 9000  # Overridden
        assert merged["vector_store_config"]["ssl"] is True  # New
    
    def test_create_environment_config(self):
        """Test creating environment-specific configuration"""
        # Create base config
        base_config = {
            "llm_provider": "google",
            "chunk_size": 1000,
            "enable_logging": True
        }
        
        base_config_path = Path(self.temp_dir) / "config.yaml"
        with open(base_config_path, 'w') as f:
            yaml.dump(base_config, f)
        
        # Create environment-specific overrides
        overrides = {
            "enable_metrics": True,
            "log_level": "ERROR",
            "chunk_size": 1500
        }
        
        env_config_path = self.config_manager.create_environment_config(
            str(base_config_path), "production", overrides
        )
        
        assert Path(env_config_path).exists()
        
        # Verify the environment config
        with open(env_config_path, 'r') as f:
            env_config = yaml.safe_load(f)
        
        assert env_config["llm_provider"] == "google"  # From base
        assert env_config["chunk_size"] == 1500  # Overridden
        assert env_config["enable_metrics"] is True  # New
        assert env_config["log_level"] == "ERROR"  # New
    
    def test_list_available_environments(self):
        """Test listing available environment configurations"""
        # Create multiple environment configs
        configs = {
            "config.yaml": {"base": True},
            "config.development.yaml": {"env": "dev"},
            "config.staging.yaml": {"env": "staging"},
            "config.production.yaml": {"env": "prod"},
            "other.testing.json": {"env": "test"}
        }
        
        for filename, content in configs.items():
            config_path = Path(self.temp_dir) / filename
            if filename.endswith('.json'):
                with open(config_path, 'w') as f:
                    json.dump(content, f)
            else:
                with open(config_path, 'w') as f:
                    yaml.dump(content, f)
        
        environments = self.config_manager.list_available_environments(self.temp_dir)
        
        assert "development" in environments
        assert "staging" in environments
        assert "production" in environments
        assert "testing" in environments
        assert len(environments) == 4
    
    def test_config_property(self):
        """Test config property getter"""
        # First access should load config
        config1 = self.config_manager.config
        assert isinstance(config1, PipelineConfig)
        
        # Second access should return cached config
        config2 = self.config_manager.config
        assert config1 is config2
    
    def test_reload_config(self):
        """Test reloading configuration"""
        # Load initial config
        config1 = self.config_manager.config
        
        # Reload should create new instance
        config2 = self.config_manager.reload_config()
        
        assert config1 is not config2
        assert isinstance(config2, PipelineConfig)
    
    @patch('src.rag_engine.core.config.logging.getLogger')
    def test_logging_integration(self, mock_logger):
        """Test that configuration manager uses logging properly"""
        mock_logger_instance = MagicMock()
        mock_logger.return_value = mock_logger_instance
        
        config_manager = ConfigurationManager()
        config_manager.load_config()
        
        # Verify logging calls were made
        mock_logger_instance.info.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])