"""
Multi-provider LLM support implementation.
Implements requirement 7.2 for supporting multiple LLM backends.
Follows the steering rules to use Google Gemini as the primary provider.
"""

from typing import Dict, Any, Optional, List
import logging
import os
from abc import ABC, abstractmethod

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

from ..core.interfaces import BaseLLMProvider
from ..core.config import PipelineConfig
from ..core.exceptions import ConfigurationError, GenerationError, ExternalServiceError
from ..core.resilience import (
    resilient_call, 
    RetryConfig, 
    CircuitBreakerConfig, 
    get_resilience_manager
)

logger = logging.getLogger(__name__)


class GoogleLLMProvider(BaseLLMProvider):
    """Google Gemini LLM provider implementation"""
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize Google LLM provider.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            # Map OpenAI model names to Google Gemini equivalents
            model_mapping = {
                "gpt-3.5-turbo": "gemini-2.0-flash-lite",
                "gpt-4": "gemini-2.0-flash-lite",
                "gpt-4-turbo": "gemini-2.0-flash-lite",
                "gemini-pro": "gemini-2.0-flash-lite",
                "gemini-2.0-flash-lite": "gemini-2.0-flash-lite",
                "gemini-2.0-flash-lite": "gemini-2.0-flash-lite",
                "gemini-2.0-flash-lite": "gemini-2.0-flash-lite"
            }
            
            model_name = model_mapping.get(config.llm_model, "gemini-2.0-flash-lite")
            
            # Initialize with configuration
            init_params = {
                "model": model_name,
                "temperature": config.temperature,
            }
            
            # Add max_tokens if specified
            if hasattr(config, 'max_tokens') and config.max_tokens:
                init_params["max_tokens"] = config.max_tokens
            
            # Add API key if available
            if hasattr(config, 'google_api_key') and config.google_api_key:
                init_params["google_api_key"] = config.google_api_key
            elif os.getenv("GOOGLE_API_KEY"):
                init_params["google_api_key"] = os.getenv("GOOGLE_API_KEY")
            
            self.llm = ChatGoogleGenerativeAI(**init_params)
            self.model_name = model_name
            
            logger.info(f"GoogleLLMProvider initialized with model: {model_name}")
            
        except ImportError as e:
            raise ConfigurationError(f"Google Generative AI not available. Install with: pip install langchain-google-genai. Error: {str(e)}")
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Google LLM provider: {str(e)}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using Google Gemini with resilience patterns.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            str: Generated text
        """
        def _generate_internal():
            try:
                response = self.llm.invoke(prompt, **kwargs)
                return response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                logger.error(f"Error generating text with Google provider: {str(e)}")
                raise ExternalServiceError(f"Google LLM generation failed: {str(e)}")
        
        def _fallback_generate():
            logger.warning("Using fallback generation for Google provider")
            return f"[Generation temporarily unavailable. Original prompt: {prompt[:100]}...]"
        
        try:
            return resilient_call(
                _generate_internal,
                circuit_breaker_name="google_llm",
                retry_name="google_llm_retry",
                fallback_operation="google_generate",
                fallback_func=_fallback_generate
            )
        except Exception as e:
            raise GenerationError(f"Google LLM generation failed after all resilience attempts: {str(e)}")
    
    def generate_with_structured_output(self, prompt: str, schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Generate structured output using Google Gemini with resilience patterns.
        
        Args:
            prompt: Input prompt
            schema: Output schema
            **kwargs: Additional generation parameters
            
        Returns:
            Dict[str, Any]: Structured output
        """
        def _generate_structured_internal():
            try:
                # Use structured output if available
                if hasattr(self.llm, 'with_structured_output'):
                    structured_llm = self.llm.with_structured_output(schema)
                    response = structured_llm.invoke(prompt, **kwargs)
                    return response if isinstance(response, dict) else {"response": str(response)}
                else:
                    # Fallback to regular generation
                    response = self.generate(prompt, **kwargs)
                    return {"response": response}
            except Exception as e:
                logger.error(f"Error generating structured output with Google provider: {str(e)}")
                raise ExternalServiceError(f"Google structured generation failed: {str(e)}")
        
        def _fallback_structured():
            logger.warning("Using fallback structured generation for Google provider")
            return {"error": "Structured generation temporarily unavailable", "fallback": True}
        
        try:
            return resilient_call(
                _generate_structured_internal,
                circuit_breaker_name="google_llm_structured",
                retry_name="google_llm_structured_retry",
                fallback_operation="google_structured_generate",
                fallback_func=_fallback_structured
            )
        except Exception as e:
            return {"error": f"Structured generation failed after all resilience attempts: {str(e)}"}
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            "provider": "google",
            "model": self.model_name,
            "temperature": self.config.temperature,
            "max_tokens": getattr(self.config, 'max_tokens', None),
            "supports_structured_output": hasattr(self.llm, 'with_structured_output')
        }


class OpenAILLMProvider(BaseLLMProvider):
    """OpenAI LLM provider implementation (fallback option)"""
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize OpenAI LLM provider.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        
        try:
            from langchain_openai import ChatOpenAI
            
            # Initialize with configuration
            init_params = {
                "model": config.llm_model,
                "temperature": config.temperature,
            }
            
            # Add max_tokens if specified
            if hasattr(config, 'max_tokens') and config.max_tokens:
                init_params["max_tokens"] = config.max_tokens
            
            # Add API key if available
            if hasattr(config, 'openai_api_key') and config.openai_api_key:
                init_params["openai_api_key"] = config.openai_api_key
            elif os.getenv("OPENAI_API_KEY"):
                init_params["openai_api_key"] = os.getenv("OPENAI_API_KEY")
            
            self.llm = ChatOpenAI(**init_params)
            
            logger.info(f"OpenAILLMProvider initialized with model: {config.llm_model}")
            
        except ImportError as e:
            raise ConfigurationError(f"OpenAI not available. Install with: pip install langchain-openai. Error: {str(e)}")
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize OpenAI LLM provider: {str(e)}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using OpenAI with resilience patterns.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            str: Generated text
        """
        def _generate_internal():
            try:
                response = self.llm.invoke(prompt, **kwargs)
                return response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                logger.error(f"Error generating text with OpenAI provider: {str(e)}")
                raise ExternalServiceError(f"OpenAI LLM generation failed: {str(e)}")
        
        def _fallback_generate():
            logger.warning("Using fallback generation for OpenAI provider")
            return f"[Generation temporarily unavailable. Original prompt: {prompt[:100]}...]"
        
        try:
            return resilient_call(
                _generate_internal,
                circuit_breaker_name="openai_llm",
                retry_name="openai_llm_retry",
                fallback_operation="openai_generate",
                fallback_func=_fallback_generate
            )
        except Exception as e:
            raise GenerationError(f"OpenAI LLM generation failed after all resilience attempts: {str(e)}")
    
    def generate_with_structured_output(self, prompt: str, schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Generate structured output using OpenAI with resilience patterns.
        
        Args:
            prompt: Input prompt
            schema: Output schema
            **kwargs: Additional generation parameters
            
        Returns:
            Dict[str, Any]: Structured output
        """
        def _generate_structured_internal():
            try:
                # Use structured output if available
                if hasattr(self.llm, 'with_structured_output'):
                    structured_llm = self.llm.with_structured_output(schema)
                    response = structured_llm.invoke(prompt, **kwargs)
                    return response if isinstance(response, dict) else {"response": str(response)}
                else:
                    # Fallback to regular generation
                    response = self.generate(prompt, **kwargs)
                    return {"response": response}
            except Exception as e:
                logger.error(f"Error generating structured output with OpenAI provider: {str(e)}")
                raise ExternalServiceError(f"OpenAI structured generation failed: {str(e)}")
        
        def _fallback_structured():
            logger.warning("Using fallback structured generation for OpenAI provider")
            return {"error": "Structured generation temporarily unavailable", "fallback": True}
        
        try:
            return resilient_call(
                _generate_structured_internal,
                circuit_breaker_name="openai_llm_structured",
                retry_name="openai_llm_structured_retry",
                fallback_operation="openai_structured_generate",
                fallback_func=_fallback_structured
            )
        except Exception as e:
            return {"error": f"Structured generation failed after all resilience attempts: {str(e)}"}
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            "provider": "openai",
            "model": self.config.llm_model,
            "temperature": self.config.temperature,
            "max_tokens": getattr(self.config, 'max_tokens', None),
            "supports_structured_output": hasattr(self.llm, 'with_structured_output')
        }





class LocalLLMProvider(BaseLLMProvider):
    """Local LLM provider implementation using Ollama or similar"""
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize Local LLM provider.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        
        try:
            from langchain_community.llms import Ollama
            
            # Initialize with configuration
            init_params = {
                "model": config.llm_model,
                "temperature": config.temperature,
            }
            
            # Add base URL if specified
            if hasattr(config, 'ollama_base_url') and config.ollama_base_url:
                init_params["base_url"] = config.ollama_base_url
            
            self.llm = Ollama(**init_params)
            
            logger.info(f"LocalLLMProvider initialized with model: {config.llm_model}")
            
        except ImportError as e:
            raise ConfigurationError(f"Ollama not available. Install with: pip install langchain-community. Error: {str(e)}")
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Local LLM provider: {str(e)}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using local LLM with resilience patterns.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            str: Generated text
        """
        def _generate_internal():
            try:
                response = self.llm.invoke(prompt, **kwargs)
                return response if isinstance(response, str) else str(response)
            except Exception as e:
                logger.error(f"Error generating text with Local provider: {str(e)}")
                raise ExternalServiceError(f"Local LLM generation failed: {str(e)}")
        
        def _fallback_generate():
            logger.warning("Using fallback generation for Local provider")
            return f"[Local generation temporarily unavailable. Original prompt: {prompt[:100]}...]"
        
        try:
            return resilient_call(
                _generate_internal,
                circuit_breaker_name="local_llm",
                retry_name="local_llm_retry",
                fallback_operation="local_generate",
                fallback_func=_fallback_generate
            )
        except Exception as e:
            raise GenerationError(f"Local LLM generation failed after all resilience attempts: {str(e)}")
    
    def generate_with_structured_output(self, prompt: str, schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Generate structured output using local LLM.
        
        Args:
            prompt: Input prompt
            schema: Output schema
            **kwargs: Additional generation parameters
            
        Returns:
            Dict[str, Any]: Structured output
        """
        try:
            # Local models typically don't support structured output directly
            # So we use prompt engineering to request structured format
            structured_prompt = f"{prompt}\n\nPlease respond in JSON format according to this schema: {schema}"
            response = self.generate(structured_prompt, **kwargs)
            
            # Try to parse as JSON
            import json
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return {"response": response}
                
        except Exception as e:
            logger.error(f"Error generating structured output with Local provider: {str(e)}")
            return {"error": str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            "provider": "local",
            "model": self.config.llm_model,
            "temperature": self.config.temperature,
            "max_tokens": getattr(self.config, 'max_tokens', None),
            "supports_structured_output": False,
            "base_url": getattr(self.config, 'ollama_base_url', None)
        }


class LLMProviderFactory:
    """Factory class for creating LLM providers"""
    
    _providers = {
        "google": GoogleLLMProvider,
        "openai": OpenAILLMProvider,
        "local": LocalLLMProvider,
        "ollama": LocalLLMProvider,  # Alias for local
    }
    
    @classmethod
    def create_provider(cls, provider_name: str, config: PipelineConfig) -> BaseLLMProvider:
        """
        Create an LLM provider instance with resilience configuration.
        
        Args:
            provider_name: Name of the provider to create
            config: Pipeline configuration
            
        Returns:
            BaseLLMProvider: LLM provider instance
            
        Raises:
            ConfigurationError: If provider is not supported
        """
        provider_name = provider_name.lower()
        
        if provider_name not in cls._providers:
            available_providers = list(cls._providers.keys())
            raise ConfigurationError(f"Unsupported LLM provider: {provider_name}. Available providers: {available_providers}")
        
        # Configure resilience patterns for the provider
        cls._configure_resilience_for_provider(provider_name)
        
        provider_class = cls._providers[provider_name]
        
        try:
            return provider_class(config)
        except Exception as e:
            logger.error(f"Failed to create {provider_name} provider: {str(e)}")
            raise ConfigurationError(f"Failed to create {provider_name} provider: {str(e)}")
    
    @classmethod
    def _configure_resilience_for_provider(cls, provider_name: str) -> None:
        """Configure resilience patterns for a specific provider"""
        manager = get_resilience_manager()
        
        # Configure circuit breakers
        cb_config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0,
            expected_exception=ExternalServiceError,
            name=f"{provider_name}_llm"
        )
        manager.create_circuit_breaker(f"{provider_name}_llm", cb_config)
        
        # Configure structured output circuit breaker
        cb_structured_config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0,
            expected_exception=ExternalServiceError,
            name=f"{provider_name}_llm_structured"
        )
        manager.create_circuit_breaker(f"{provider_name}_llm_structured", cb_structured_config)
        
        # Configure retry handlers
        retry_config = RetryConfig(
            max_attempts=3,
            base_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0,
            jitter=True,
            retryable_exceptions=[ExternalServiceError, ConnectionError, TimeoutError]
        )
        manager.create_retry_handler(f"{provider_name}_llm_retry", retry_config)
        manager.create_retry_handler(f"{provider_name}_llm_structured_retry", retry_config)
        
        logger.info(f"Configured resilience patterns for {provider_name} provider")
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """
        Get list of available provider names.
        
        Returns:
            List[str]: Available provider names
        """
        return list(cls._providers.keys())
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type) -> None:
        """
        Register a new LLM provider.
        
        Args:
            name: Provider name
            provider_class: Provider class that implements BaseLLMProvider
        """
        if not issubclass(provider_class, BaseLLMProvider):
            raise ValueError(f"Provider class must implement BaseLLMProvider interface")
        
        cls._providers[name.lower()] = provider_class
        logger.info(f"Registered new LLM provider: {name}")


def get_default_provider_config() -> Dict[str, Dict[str, Any]]:
    """
    Get default configuration for each provider.
    
    Returns:
        Dict[str, Dict[str, Any]]: Default configurations by provider
    """
    return {
        "google": {
            "models": ["gemini-2.0-flash-lite", "gemini-2.0-flash-lite", "gemini-2.0-flash-lite", "gemini-pro"],
            "default_model": "gemini-2.0-flash-lite",
            "supports_structured_output": True,
            "api_key_env": "GOOGLE_API_KEY"
        },
        "openai": {
            "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"],
            "default_model": "gpt-3.5-turbo",
            "supports_structured_output": True,
            "api_key_env": "OPENAI_API_KEY"
        },
        "local": {
            "models": ["llama2", "mistral", "codellama", "custom"],
            "default_model": "llama2",
            "supports_structured_output": False,
            "api_key_env": None
        }
    }


def validate_provider_config(provider_name: str, config: PipelineConfig) -> List[str]:
    """
    Validate provider configuration and return any issues.
    
    Args:
        provider_name: Name of the provider
        config: Pipeline configuration
        
    Returns:
        List[str]: List of validation issues (empty if valid)
    """
    issues = []
    provider_configs = get_default_provider_config()
    
    if provider_name not in provider_configs:
        issues.append(f"Unknown provider: {provider_name}")
        return issues
    
    provider_config = provider_configs[provider_name]
    
    # Check if API key is required and available
    if provider_config["api_key_env"]:
        api_key_attr = f"{provider_name}_api_key"
        if not (hasattr(config, api_key_attr) and getattr(config, api_key_attr)) and not os.getenv(provider_config["api_key_env"]):
            issues.append(f"API key required for {provider_name} provider. Set {provider_config['api_key_env']} environment variable or config.{api_key_attr}")
    
    # Check if model is supported
    if hasattr(config, 'llm_model') and config.llm_model:
        if config.llm_model not in provider_config["models"] and provider_name != "local":
            issues.append(f"Model {config.llm_model} not supported by {provider_name} provider. Supported models: {provider_config['models']}")
    
    return issues