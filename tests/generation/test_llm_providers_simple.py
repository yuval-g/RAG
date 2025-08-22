"""
Simple unit tests for multi-provider LLM support.
Tests requirement 7.2 for supporting multiple LLM backends.
"""

import pytest
from unittest.mock import Mock, patch
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check if optional dependencies are available
try:
    import langchain_anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import langchain_openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import langchain_community
    HAS_COMMUNITY = True
except ImportError:
    HAS_COMMUNITY = False

from src.rag_engine.generation.llm_providers import (
    LLMProviderFactory,
    get_default_provider_config,
    validate_provider_config
)
from src.rag_engine.core.config import PipelineConfig
from src.rag_engine.core.exceptions import ConfigurationError


class TestLLMProviderFactory:
    """Test LLM provider factory"""
    
    def test_get_available_providers(self):
        """Test getting available providers"""
        providers = LLMProviderFactory.get_available_providers()
        
        expected_providers = ["google", "openai", "anthropic", "local", "ollama"]
        assert all(provider in providers for provider in expected_providers)
    
    def test_create_unsupported_provider(self):
        """Test creating unsupported provider"""
        config = PipelineConfig(llm_provider="unsupported")
        
        with pytest.raises(ConfigurationError, match="Unsupported LLM provider"):
            LLMProviderFactory.create_provider("unsupported", config)
    
    def test_register_custom_provider(self):
        """Test registering custom provider"""
        from src.rag_engine.core.interfaces import BaseLLMProvider
        
        class CustomProvider(BaseLLMProvider):
            def __init__(self, config):
                self.config = config
            
            def generate(self, prompt, **kwargs):
                return "custom response"
            
            def generate_with_structured_output(self, prompt, schema, **kwargs):
                return {"custom": "response"}
            
            def get_model_info(self):
                return {"provider": "custom"}
        
        LLMProviderFactory.register_provider("custom", CustomProvider)
        
        config = PipelineConfig(llm_provider="custom")
        provider = LLMProviderFactory.create_provider("custom", config)
        
        assert isinstance(provider, CustomProvider)
        assert "custom" in LLMProviderFactory.get_available_providers()
    
    def test_register_invalid_provider(self):
        """Test registering invalid provider class"""
        class InvalidProvider:
            pass
        
        with pytest.raises(ValueError, match="must implement BaseLLMProvider"):
            LLMProviderFactory.register_provider("invalid", InvalidProvider)


class TestProviderConfiguration:
    """Test provider configuration utilities"""
    
    def test_get_default_provider_config(self):
        """Test getting default provider configurations"""
        configs = get_default_provider_config()
        
        assert "google" in configs
        assert "openai" in configs
        assert "anthropic" in configs
        assert "local" in configs
        
        # Check Google config
        google_config = configs["google"]
        assert "gemini-2.0-flash-lite" in google_config["models"]
        assert google_config["default_model"] == "gemini-2.0-flash-lite"
        assert google_config["supports_structured_output"] is True
        assert google_config["api_key_env"] == "GOOGLE_API_KEY"
    
    def test_validate_provider_config_valid(self):
        """Test validating valid provider configuration"""
        config = PipelineConfig(
            llm_provider="google",
            llm_model="gemini-pro",
            google_api_key="test-key"
        )
        
        issues = validate_provider_config("google", config)
        assert len(issues) == 0
    
    def test_validate_provider_config_missing_api_key(self):
        """Test validating configuration with missing API key"""
        config = PipelineConfig(
            llm_provider="google",
            llm_model="gemini-pro"
        )
        
        with patch.dict(os.environ, {}, clear=True):
            issues = validate_provider_config("google", config)
            assert len(issues) > 0
            assert any("API key required" in issue for issue in issues)
    
    def test_validate_provider_config_api_key_from_env(self):
        """Test validating configuration with API key from environment"""
        config = PipelineConfig(
            llm_provider="google",
            llm_model="gemini-pro"
        )
        
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "env-key"}):
            issues = validate_provider_config("google", config)
            assert len(issues) == 0
    
    def test_validate_provider_config_unsupported_model(self):
        """Test validating configuration with unsupported model"""
        config = PipelineConfig(
            llm_provider="google",
            llm_model="unsupported-model",
            google_api_key="test-key"
        )
        
        issues = validate_provider_config("google", config)
        assert len(issues) > 0
        assert any("not supported" in issue for issue in issues)
    
    def test_validate_provider_config_unknown_provider(self):
        """Test validating configuration for unknown provider"""
        config = PipelineConfig(llm_provider="unknown")
        
        issues = validate_provider_config("unknown", config)
        assert len(issues) > 0
        assert any("Unknown provider" in issue for issue in issues)
    
    def test_validate_provider_config_local_no_api_key(self):
        """Test validating local provider configuration (no API key required)"""
        config = PipelineConfig(
            llm_provider="local",
            llm_model="llama2"
        )
        
        issues = validate_provider_config("local", config)
        # Local provider should not require API key
        assert not any("API key required" in issue for issue in issues)


class TestConfigurationManager:
    """Test configuration manager with new providers"""
    
    def test_config_validation_with_google_provider(self):
        """Test configuration validation with Google provider"""
        from src.rag_engine.core.config import ConfigurationManager
        
        # This should not raise an exception
        config_manager = ConfigurationManager()
        config = PipelineConfig(
            llm_provider="google",
            llm_model="gemini-pro"
        )
        
        # The validation should pass for Google provider
        config_manager._validate_config(config)
    
    def test_config_validation_with_invalid_provider(self):
        """Test configuration validation with invalid provider"""
        from src.rag_engine.core.config import ConfigurationManager
        
        config_manager = ConfigurationManager()
        config = PipelineConfig(llm_provider="invalid")
        
        # The validation should not raise an exception for invalid provider name
        # Provider validation is done when creating the provider, not during config validation
        config_manager._validate_config(config)
    
    def test_environment_variable_mapping(self):
        """Test environment variable mapping for new providers"""
        from src.rag_engine.core.config import ConfigurationManager
        
        config_manager = ConfigurationManager()
        
        with patch.dict(os.environ, {
            "GOOGLE_API_KEY": "test-google-key",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
            "OLLAMA_BASE_URL": "http://localhost:11434"
        }):
            env_config = config_manager._load_from_environment()
            
            assert env_config.get("google_api_key") == "test-google-key"
            assert env_config.get("anthropic_api_key") == "test-anthropic-key"
            assert env_config.get("ollama_base_url") == "http://localhost:11434"


class TestProviderIntegrationMocked:
    """Test provider integration with mocked dependencies"""
    
    @patch('langchain_google_genai.ChatGoogleGenerativeAI')
    def test_google_provider_creation(self, mock_chat):
        """Test Google provider creation with mocked dependency"""
        config = PipelineConfig(
            llm_provider="google",
            llm_model="gemini-pro"
        )
        
        from src.rag_engine.generation.llm_providers import GoogleLLMProvider
        
        provider = GoogleLLMProvider(config)
        assert provider.config == config
        mock_chat.assert_called_once()
    
    @pytest.mark.skipif(not HAS_OPENAI, reason="langchain-openai not installed")
    @patch('langchain_openai.ChatOpenAI')
    def test_openai_provider_creation(self, mock_chat):
        """Test OpenAI provider creation with mocked dependency"""
        config = PipelineConfig(
            llm_provider="openai",
            llm_model="gpt-3.5-turbo"
        )
        
        from src.rag_engine.generation.llm_providers import OpenAILLMProvider
        
        provider = OpenAILLMProvider(config)
        assert provider.config == config
        mock_chat.assert_called_once()
    
    @pytest.mark.skipif(not HAS_ANTHROPIC, reason="langchain-anthropic not installed")
    @patch('langchain_anthropic.ChatAnthropic')
    def test_anthropic_provider_creation(self, mock_chat):
        """Test Anthropic provider creation with mocked dependency"""
        config = PipelineConfig(
            llm_provider="anthropic",
            llm_model="claude-3-sonnet"
        )
        
        from src.rag_engine.generation.llm_providers import AnthropicLLMProvider
        
        provider = AnthropicLLMProvider(config)
        assert provider.config == config
        mock_chat.assert_called_once()
    
    @pytest.mark.skipif(not HAS_COMMUNITY, reason="langchain-community not installed")
    @patch('langchain_community.llms.Ollama')
    def test_local_provider_creation(self, mock_ollama):
        """Test Local provider creation with mocked dependency"""
        config = PipelineConfig(
            llm_provider="local",
            llm_model="llama2"
        )
        
        from src.rag_engine.generation.llm_providers import LocalLLMProvider
        
        provider = LocalLLMProvider(config)
        assert provider.config == config
        mock_ollama.assert_called_once()
    
    @patch('langchain_google_genai.ChatGoogleGenerativeAI')
    def test_factory_creates_correct_provider(self, mock_chat):
        """Test factory creates correct provider type"""
        config = PipelineConfig(llm_provider="google")
        
        from src.rag_engine.generation.llm_providers import GoogleLLMProvider
        
        provider = LLMProviderFactory.create_provider("google", config)
        assert isinstance(provider, GoogleLLMProvider)
        mock_chat.assert_called_once()