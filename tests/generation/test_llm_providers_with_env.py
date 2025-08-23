"""
Test LLM providers with .env file loaded.
This test loads the .env file to test with actual API keys if available.
"""

import pytest
import os
from dotenv import load_dotenv

from src.rag_engine.generation.llm_providers import (
    LLMProviderFactory,
    validate_provider_config
)
from src.rag_engine.core.config import PipelineConfig
from src.rag_engine.core.exceptions import ConfigurationError

# Load environment variables from .env file
load_dotenv()


class TestLLMProvidersWithEnv:
    """Test LLM providers with environment variables loaded"""
    
    def test_google_provider_with_env_key(self):
        """Test Google provider with API key from .env file"""
        google_api_key = os.getenv("GOOGLE_API_KEY")
        
        if not google_api_key:
            pytest.skip("GOOGLE_API_KEY not set in environment or .env file")
        
        config = PipelineConfig(
            llm_provider="google",
            llm_model="gemini-2.0-flash-lite",
            temperature=0.5
        )
        
        # Validate configuration - should pass with API key
        issues = validate_provider_config("google", config)
        assert len(issues) == 0, f"Configuration issues: {issues}"
        
        # Try to create provider - should succeed with valid API key
        try:
            provider = LLMProviderFactory.create_provider("google", config)
            assert provider is not None
            
            # Get model info
            model_info = provider.get_model_info()
            assert model_info["provider"] == "google"
            assert model_info["model"] == "gemini-2.0-flash-lite"
            
            print(f"✅ Google provider created successfully: {model_info}")
            
        except ConfigurationError as e:
            pytest.fail(f"Failed to create Google provider with API key: {str(e)}")
    
    def test_openai_provider_with_env_key(self):
        """Test OpenAI provider with API key from .env file"""
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not openai_api_key:
            pytest.skip("OPENAI_API_KEY not set in environment or .env file")
        
        config = PipelineConfig(
            llm_provider="openai",
            llm_model="gpt-3.5-turbo",
            temperature=0.5
        )
        
        # Validate configuration - should pass with API key
        issues = validate_provider_config("openai", config)
        assert len(issues) == 0, f"Configuration issues: {issues}"
        
        # Try to create provider - should succeed with valid API key
        try:
            provider = LLMProviderFactory.create_provider("openai", config)
            assert provider is not None
            
            # Get model info
            model_info = provider.get_model_info()
            assert model_info["provider"] == "openai"
            assert model_info["model"] == "gpt-3.5-turbo"
            
            print(f"✅ OpenAI provider created successfully: {model_info}")
            
        except ConfigurationError as e:
            pytest.fail(f"Failed to create OpenAI provider with API key: {str(e)}")
    
    def test_google_provider_generation_with_env_key(self):
        """Test Google provider text generation with API key from .env file"""
        google_api_key = os.getenv("GOOGLE_API_KEY")
        
        if not google_api_key:
            pytest.skip("GOOGLE_API_KEY not set in environment or .env file")
        
        config = PipelineConfig(
            llm_provider="google",
            llm_model="gemini-2.0-flash-lite",  # Use the model from .env file
            temperature=0.1  # Low temperature for consistent results
        )
        
        try:
            provider = LLMProviderFactory.create_provider("google", config)
            
            # Test basic generation
            prompt = "What is 2 + 2? Answer with just the number."
            response = provider.generate(prompt)
            
            assert response is not None
            assert len(response.strip()) > 0
            
            print(f"✅ Google provider generation test:")
            print(f"   Prompt: {prompt}")
            print(f"   Response: {response}")
            
        except Exception as e:
            pytest.fail(f"Failed to generate text with Google provider: {str(e)}")
    
    def test_environment_variable_detection(self):
        """Test that environment variables are properly detected"""
        # Check if .env file variables are loaded
        google_key = os.getenv("GOOGLE_API_KEY")
        default_model = os.getenv("DEFAULT_LLM_MODEL")
        default_temp = os.getenv("DEFAULT_TEMPERATURE")
        
        print(f"Environment variables detected:")
        print(f"  GOOGLE_API_KEY: {'✅ Set' if google_key else '❌ Not set'}")
        print(f"  DEFAULT_LLM_MODEL: {default_model or '❌ Not set'}")
        print(f"  DEFAULT_TEMPERATURE: {default_temp or '❌ Not set'}")
        
        # At least one should be set if .env is loaded
        assert google_key or default_model or default_temp, "No environment variables detected from .env file"