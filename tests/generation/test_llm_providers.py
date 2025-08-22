"""
Unit tests for multi-provider LLM support.
Tests requirement 7.2 for supporting multiple LLM backends.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
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
    GoogleLLMProvider,
    OpenAILLMProvider,
    AnthropicLLMProvider,
    LocalLLMProvider,
    LLMProviderFactory,
    get_default_provider_config,
    validate_provider_config
)
from src.rag_engine.core.config import PipelineConfig
from src.rag_engine.core.exceptions import ConfigurationError, GenerationError


class TestGoogleLLMProvider:
    """Test Google LLM provider"""
    
    def test_google_provider_initialization(self):
        """Test Google provider initialization"""
        config = PipelineConfig(
            llm_provider="google",
            llm_model="gemini-pro",
            temperature=0.5,
            max_tokens=1000
        )
        
        with patch('langchain_google_genai.ChatGoogleGenerativeAI') as mock_chat:
            mock_llm = Mock()
            mock_chat.return_value = mock_llm
            
            provider = GoogleLLMProvider(config)
            
            assert provider.config == config
            assert provider.model_name == "gemini-2.0-flash-lite"
            mock_chat.assert_called_once()
    
    def test_google_provider_model_mapping(self):
        """Test Google provider model name mapping"""
        config = PipelineConfig(
            llm_provider="google",
            llm_model="gpt-4",  # Should be mapped to gemini-pro
            temperature=0.0
        )
        
        with patch('langchain_google_genai.ChatGoogleGenerativeAI') as mock_chat:
            provider = GoogleLLMProvider(config)
            
            # Check that gpt-4 was mapped to gemini-2.0-flash-lite
            call_args = mock_chat.call_args[1]
            assert call_args["model"] == "gemini-2.0-flash-lite"
    
    def test_google_provider_generate(self):
        """Test Google provider text generation"""
        config = PipelineConfig(llm_provider="google")
        
        with patch('langchain_google_genai.ChatGoogleGenerativeAI') as mock_chat:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = "Generated response"
            mock_llm.invoke.return_value = mock_response
            mock_chat.return_value = mock_llm
            
            provider = GoogleLLMProvider(config)
            result = provider.generate("Test prompt")
            
            assert result == "Generated response"
            mock_llm.invoke.assert_called_once_with("Test prompt")
    
    def test_google_provider_generate_error(self):
        """Test Google provider generation error handling"""
        config = PipelineConfig(llm_provider="google")
        
        with patch('langchain_google_genai.ChatGoogleGenerativeAI') as mock_chat:
            mock_llm = Mock()
            mock_llm.invoke.side_effect = Exception("API Error")
            mock_chat.return_value = mock_llm
            
            provider = GoogleLLMProvider(config)
            result = provider.generate("Test prompt")
            
            # Should return fallback response when generation fails
            assert "Generation temporarily unavailable" in result
    
    def test_google_provider_structured_output(self):
        """Test Google provider structured output generation"""
        config = PipelineConfig(llm_provider="google")
        
        with patch('langchain_google_genai.ChatGoogleGenerativeAI') as mock_chat:
            mock_llm = Mock()
            mock_structured_llm = Mock()
            mock_structured_llm.invoke.return_value = {"key": "value"}
            mock_llm.with_structured_output.return_value = mock_structured_llm
            mock_chat.return_value = mock_llm
            
            provider = GoogleLLMProvider(config)
            schema = {"type": "object"}
            result = provider.generate_with_structured_output("Test prompt", schema)
            
            assert result == {"key": "value"}
            mock_llm.with_structured_output.assert_called_once_with(schema)
    
    def test_google_provider_get_model_info(self):
        """Test Google provider model info"""
        config = PipelineConfig(
            llm_provider="google",
            llm_model="gemini-pro",
            temperature=0.7,
            max_tokens=500
        )
        
        with patch('langchain_google_genai.ChatGoogleGenerativeAI'):
            provider = GoogleLLMProvider(config)
            info = provider.get_model_info()
            
            assert info["provider"] == "google"
            assert info["model"] == "gemini-2.0-flash-lite"
            assert info["temperature"] == 0.7
            assert info["max_tokens"] == 500


class TestOpenAILLMProvider:
    """Test OpenAI LLM provider"""
    
    @ pytest.mark.skipif(not HAS_OPENAI, reason="langchain-openai not installed")
    def test_openai_provider_initialization(self):
        """Test OpenAI provider initialization"""
        config = PipelineConfig(
            llm_provider="openai",
            llm_model="gpt-3.5-turbo",
            temperature=0.5
        )
        
        with patch('langchain_openai.ChatOpenAI') as mock_chat:
            provider = OpenAILLMProvider(config)
            
            assert provider.config == config
            mock_chat.assert_called_once()
    
    @ pytest.mark.skipif(not HAS_OPENAI, reason="langchain-openai not installed")
    def test_openai_provider_missing_dependency(self):
        """Test OpenAI provider with missing dependency"""
        config = PipelineConfig(llm_provider="openai")
        
        with patch('langchain_openai.ChatOpenAI', side_effect=ImportError("No module")):
            with pytest.raises(ConfigurationError, match="OpenAI not available"):
                OpenAILLMProvider(config)
    
    @ pytest.mark.skipif(not HAS_OPENAI, reason="langchain-openai not installed")
    def test_openai_provider_generate(self):
        """Test OpenAI provider text generation"""
        config = PipelineConfig(llm_provider="openai")
        
        with patch('langchain_openai.ChatOpenAI') as mock_chat:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = "OpenAI response"
            mock_llm.invoke.return_value = mock_response
            mock_chat.return_value = mock_llm
            
            provider = OpenAILLMProvider(config)
            result = provider.generate("Test prompt")
            
            assert result == "OpenAI response"


class TestAnthropicLLMProvider:
    """Test Anthropic LLM provider"""
    
    @ pytest.mark.skipif(not HAS_ANTHROPIC, reason="langchain-anthropic not installed")
    def test_anthropic_provider_initialization(self):
        """Test Anthropic provider initialization"""
        config = PipelineConfig(
            llm_provider="anthropic",
            llm_model="claude-3-sonnet",
            temperature=0.3
        )
        
        with patch('langchain_anthropic.ChatAnthropic') as mock_chat:
            provider = AnthropicLLMProvider(config)
            
            assert provider.config == config
            mock_chat.assert_called_once()
    
    @ pytest.mark.skipif(not HAS_ANTHROPIC, reason="langchain-anthropic not installed")
    def test_anthropic_provider_model_mapping(self):
        """Test Anthropic provider model name mapping"""
        config = PipelineConfig(
            llm_provider="anthropic",
            llm_model="gpt-4",  # Should be mapped to claude-3-opus
            temperature=0.0
        )
        
        with patch('langchain_anthropic.ChatAnthropic') as mock_chat:
            provider = AnthropicLLMProvider(config)
            
            # Check that gpt-4 was mapped to claude-3-opus
            call_args = mock_chat.call_args[1]
            assert call_args["model"] == "claude-3-opus-20240229"
    
    @ pytest.mark.skipif(not HAS_ANTHROPIC, reason="langchain-anthropic not installed")
    def test_anthropic_provider_generate(self):
        """Test Anthropic provider text generation"""
        config = PipelineConfig(llm_provider="anthropic")
        
        with patch('langchain_anthropic.ChatAnthropic') as mock_chat:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = "Claude response"
            mock_llm.invoke.return_value = mock_response
            mock_chat.return_value = mock_llm
            
            provider = AnthropicLLMProvider(config)
            result = provider.generate("Test prompt")
            
            assert result == "Claude response"


class TestLocalLLMProvider:
    """Test Local LLM provider"""
    
    @ pytest.mark.skipif(not HAS_COMMUNITY, reason="langchain-community not installed")
    def test_local_provider_initialization(self):
        """Test Local provider initialization"""
        config = PipelineConfig(
            llm_provider="local",
            llm_model="llama2",
            temperature=0.5
        )
        
        with patch('langchain_community.llms.Ollama') as mock_ollama:
            provider = LocalLLMProvider(config)
            
            assert provider.config == config
            mock_ollama.assert_called_once()
    
    @ pytest.mark.skipif(not HAS_COMMUNITY, reason="langchain-community not installed")
    def test_local_provider_generate(self):
        """Test Local provider text generation"""
        config = PipelineConfig(llm_provider="local")
        
        with patch('langchain_community.llms.Ollama') as mock_ollama:
            mock_llm = Mock()
            mock_llm.invoke.return_value = "Local response"
            mock_ollama.return_value = mock_llm
            
            provider = LocalLLMProvider(config)
            result = provider.generate("Test prompt")
            
            assert result == "Local response"
    
    @ pytest.mark.skipif(not HAS_COMMUNITY, reason="langchain-community not installed")
    def test_local_provider_structured_output(self):
        """Test Local provider structured output (JSON parsing)"""
        config = PipelineConfig(llm_provider="local")
        
        with patch('langchain_community.llms.Ollama') as mock_ollama:
            mock_llm = Mock()
            mock_llm.invoke.return_value = '{"result": "success"}'
            mock_ollama.return_value = mock_llm
            
            provider = LocalLLMProvider(config)
            schema = {"type": "object"}
            result = provider.generate_with_structured_output("Test prompt", schema)
            
            assert result == {"result": "success"}
    
    @ pytest.mark.skipif(not HAS_COMMUNITY, reason="langchain-community not installed")
    def test_local_provider_structured_output_fallback(self):
        """Test Local provider structured output fallback for invalid JSON"""
        config = PipelineConfig(llm_provider="local")
        
        with patch('langchain_community.llms.Ollama') as mock_ollama:
            mock_llm = Mock()
            mock_llm.invoke.return_value = "Invalid JSON response"
            mock_ollama.return_value = mock_llm
            
            provider = LocalLLMProvider(config)
            schema = {"type": "object"}
            result = provider.generate_with_structured_output("Test prompt", schema)
            
            assert result == {"response": "Invalid JSON response"}


class TestLLMProviderFactory:
    """Test LLM provider factory"""
    
    def test_create_google_provider(self):
        """Test creating Google provider"""
        config = PipelineConfig(llm_provider="google")
        
        with patch('langchain_google_genai.ChatGoogleGenerativeAI'):
            provider = LLMProviderFactory.create_provider("google", config)
            assert isinstance(provider, GoogleLLMProvider)
    
    @ pytest.mark.skipif(not HAS_OPENAI, reason="langchain-openai not installed")
    def test_create_openai_provider(self):
        """Test creating OpenAI provider"""
        config = PipelineConfig(llm_provider="openai")
        
        with patch('langchain_openai.ChatOpenAI'):
            provider = LLMProviderFactory.create_provider("openai", config)
            assert isinstance(provider, OpenAILLMProvider)
    
    @ pytest.mark.skipif(not HAS_ANTHROPIC, reason="langchain-anthropic not installed")
    def test_create_anthropic_provider(self):
        """Test creating Anthropic provider"""
        config = PipelineConfig(llm_provider="anthropic")
        
        with patch('langchain_anthropic.ChatAnthropic'):
            provider = LLMProviderFactory.create_provider("anthropic", config)
            assert isinstance(provider, AnthropicLLMProvider)
    
    @ pytest.mark.skipif(not HAS_COMMUNITY, reason="langchain-community not installed")
    def test_create_local_provider(self):
        """Test creating Local provider"""
        config = PipelineConfig(llm_provider="local")
        
        with patch('langchain_community.llms.Ollama'):
            provider = LLMProviderFactory.create_provider("local", config)
            assert isinstance(provider, LocalLLMProvider)
    
    @ pytest.mark.skipif(not HAS_COMMUNITY, reason="langchain-community not installed")
    def test_create_ollama_alias(self):
        """Test creating provider with ollama alias"""
        config = PipelineConfig(llm_provider="ollama")
        
        with patch('langchain_community.llms.Ollama'):
            provider = LLMProviderFactory.create_provider("ollama", config)
            assert isinstance(provider, LocalLLMProvider)
    
    def test_create_unsupported_provider(self):
        """Test creating unsupported provider"""
        config = PipelineConfig(llm_provider="unsupported")
        
        with pytest.raises(ConfigurationError, match="Unsupported LLM provider"):
            LLMProviderFactory.create_provider("unsupported", config)
    
    def test_get_available_providers(self):
        """Test getting available providers"""
        providers = LLMProviderFactory.get_available_providers()
        
        expected_providers = ["google", "openai", "anthropic", "local", "ollama"]
        assert all(provider in providers for provider in expected_providers)
    
    def test_register_custom_provider(self):
        """Test registering custom provider"""
        from src.rag_engine.core.interfaces import BaseLLMProvider
        
        class CustomProvider(BaseLLMProvider):
            def __init__(self, config):
                pass
            
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


class TestProviderIntegration:
    """Test provider integration with generation engine"""
    
    def test_generation_engine_with_google_provider(self):
        """Test generation engine with Google provider"""
        config = PipelineConfig(
            llm_provider="google",
            llm_model="gemini-pro"
        )
        
        with patch('langchain_google_genai.ChatGoogleGenerativeAI'):
            with patch('langchain.hub.pull') as mock_hub:
                mock_hub.return_value = Mock()
                
                from src.rag_engine.generation.generation_engine import GenerationEngine
                engine = GenerationEngine(config)
                
                assert isinstance(engine.llm_provider, GoogleLLMProvider)
                assert engine.get_model_info()["provider"] == "google"
    
    def test_generation_engine_provider_switching(self):
        """Test switching providers in generation engine"""
        config = PipelineConfig(llm_provider="google")
        
        with patch('langchain_google_genai.ChatGoogleGenerativeAI'):
            with patch('langchain_openai.ChatOpenAI'):
                with patch('langchain.hub.pull') as mock_hub:
                    mock_hub.return_value = Mock()
                    
                    from src.rag_engine.generation.generation_engine import GenerationEngine
                    engine = GenerationEngine(config)
                    
                    # Initially Google
                    assert engine.get_model_info()["provider"] == "google"
                    
                    # Switch to OpenAI
                    new_config = PipelineConfig(llm_provider="openai")
                    engine.update_config(new_config)
                    
                    assert engine.get_model_info()["provider"] == "openai"