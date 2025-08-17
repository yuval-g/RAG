"""
Example demonstrating multi-provider LLM support.
Shows how to use different LLM providers (Google, OpenAI, Anthropic, Local).
Implements requirement 7.2 for supporting multiple LLM backends.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the src directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rag_engine.core.config import PipelineConfig
from rag_engine.generation.llm_providers import (
    LLMProviderFactory,
    get_default_provider_config,
    validate_provider_config
)
from rag_engine.core.exceptions import ConfigurationError


def demonstrate_provider_factory():
    """Demonstrate the LLM provider factory"""
    print("🏭 LLM PROVIDER FACTORY DEMONSTRATION")
    print("=" * 80)
    
    # Show available providers
    providers = LLMProviderFactory.get_available_providers()
    print(f"📋 Available LLM providers: {', '.join(providers)}")
    print()
    
    # Show default configurations
    configs = get_default_provider_config()
    for provider_name, config in configs.items():
        print(f"🔧 {provider_name.upper()} Provider Configuration:")
        print(f"   • Models: {', '.join(config['models'])}")
        print(f"   • Default Model: {config['default_model']}")
        print(f"   • Structured Output: {config['supports_structured_output']}")
        print(f"   • API Key Environment: {config['api_key_env']}")
        print()


def demonstrate_google_provider():
    """Demonstrate Google Gemini provider"""
    print("🤖 GOOGLE GEMINI PROVIDER DEMONSTRATION")
    print("=" * 80)
    
    config = PipelineConfig(
        llm_provider="google",
        llm_model="gemini-2.0-flash-lite",
        temperature=0.7,
        max_tokens=1000
    )
    
    # Validate configuration
    issues = validate_provider_config("google", config)
    if issues:
        print("⚠️  Configuration issues found:")
        for issue in issues:
            print(f"   • {issue}")
        print("   Note: This is expected if GOOGLE_API_KEY is not set")
    else:
        print("✅ Configuration is valid")
    
    print(f"📋 Configuration:")
    print(f"   • Provider: {config.llm_provider}")
    print(f"   • Model: {config.llm_model}")
    print(f"   • Temperature: {config.temperature}")
    print(f"   • Max Tokens: {config.max_tokens}")
    print()
    
    # Try to create provider (will fail without API key, but that's expected)
    try:
        provider = LLMProviderFactory.create_provider("google", config)
        print("✅ Google provider created successfully")
        
        # Get model info
        model_info = provider.get_model_info()
        print(f"📊 Model Info: {model_info}")
        
    except ConfigurationError as e:
        print(f"⚠️  Provider creation failed (expected without API key): {str(e)}")
    
    print()


def demonstrate_openai_provider():
    """Demonstrate OpenAI provider"""
    print("🤖 OPENAI PROVIDER DEMONSTRATION")
    print("=" * 80)
    
    config = PipelineConfig(
        llm_provider="openai",
        llm_model="gpt-3.5-turbo",
        temperature=0.5,
        max_tokens=500
    )
    
    # Validate configuration
    issues = validate_provider_config("openai", config)
    if issues:
        print("⚠️  Configuration issues found:")
        for issue in issues:
            print(f"   • {issue}")
        print("   Note: This is expected if OPENAI_API_KEY is not set")
    else:
        print("✅ Configuration is valid")
    
    print(f"📋 Configuration:")
    print(f"   • Provider: {config.llm_provider}")
    print(f"   • Model: {config.llm_model}")
    print(f"   • Temperature: {config.temperature}")
    print(f"   • Max Tokens: {config.max_tokens}")
    print()
    
    # Try to create provider
    try:
        provider = LLMProviderFactory.create_provider("openai", config)
        print("✅ OpenAI provider created successfully")
        
        # Get model info
        model_info = provider.get_model_info()
        print(f"📊 Model Info: {model_info}")
        
    except ConfigurationError as e:
        print(f"⚠️  Provider creation failed (expected without API key): {str(e)}")
    
    print()


def demonstrate_local_provider():
    """Demonstrate Local/Ollama provider"""
    print("🤖 LOCAL/OLLAMA PROVIDER DEMONSTRATION")
    print("=" * 80)
    
    config = PipelineConfig(
        llm_provider="local",
        llm_model="llama2",
        temperature=0.3,
        ollama_base_url="http://localhost:11434"
    )
    
    # Validate configuration (local provider doesn't require API key)
    issues = validate_provider_config("local", config)
    if issues:
        print("⚠️  Configuration issues found:")
        for issue in issues:
            print(f"   • {issue}")
    else:
        print("✅ Configuration is valid")
    
    print(f"📋 Configuration:")
    print(f"   • Provider: {config.llm_provider}")
    print(f"   • Model: {config.llm_model}")
    print(f"   • Temperature: {config.temperature}")
    print(f"   • Base URL: {config.ollama_base_url}")
    print()
    
    # Try to create provider
    try:
        provider = LLMProviderFactory.create_provider("local", config)
        print("✅ Local provider created successfully")
        
        # Get model info
        model_info = provider.get_model_info()
        print(f"📊 Model Info: {model_info}")
        
    except ConfigurationError as e:
        print(f"⚠️  Provider creation failed: {str(e)}")
        print("   Note: This is expected if langchain-community is not installed")
    
    print()


def demonstrate_provider_switching():
    """Demonstrate switching between providers"""
    print("🔄 PROVIDER SWITCHING DEMONSTRATION")
    print("=" * 80)
    
    providers_to_test = [
        ("google", "gemini-2.0-flash-lite"),
        ("openai", "gpt-3.5-turbo"),
        ("local", "llama2")
    ]
    
    for provider_name, model_name in providers_to_test:
        print(f"🔧 Testing {provider_name} provider with {model_name}...")
        
        config = PipelineConfig(
            llm_provider=provider_name,
            llm_model=model_name,
            temperature=0.5
        )
        
        try:
            provider = LLMProviderFactory.create_provider(provider_name, config)
            model_info = provider.get_model_info()
            print(f"   ✅ Success: {model_info['provider']} - {model_info['model']}")
            
        except ConfigurationError as e:
            print(f"   ⚠️  Failed: {str(e)}")
        
        print()


def demonstrate_custom_provider():
    """Demonstrate registering a custom provider"""
    print("🔧 CUSTOM PROVIDER DEMONSTRATION")
    print("=" * 80)
    
    from rag_engine.core.interfaces import BaseLLMProvider
    
    class MockLLMProvider(BaseLLMProvider):
        """Mock LLM provider for demonstration"""
        
        def __init__(self, config):
            self.config = config
            print(f"   MockLLMProvider initialized with model: {config.llm_model}")
        
        def generate(self, prompt, **kwargs):
            return f"Mock response to: {prompt[:50]}..."
        
        def generate_with_structured_output(self, prompt, schema, **kwargs):
            return {"mock": "structured response", "prompt": prompt[:30]}
        
        def get_model_info(self):
            return {
                "provider": "mock",
                "model": self.config.llm_model,
                "temperature": self.config.temperature,
                "supports_structured_output": True
            }
    
    # Register the custom provider
    LLMProviderFactory.register_provider("mock", MockLLMProvider)
    print("✅ Registered custom 'mock' provider")
    
    # Test the custom provider
    config = PipelineConfig(
        llm_provider="mock",
        llm_model="mock-model-v1",
        temperature=0.8
    )
    
    provider = LLMProviderFactory.create_provider("mock", config)
    print("✅ Created mock provider instance")
    
    # Test generation
    response = provider.generate("What is the meaning of life?")
    print(f"📝 Generated response: {response}")
    
    # Test structured output
    structured = provider.generate_with_structured_output(
        "Generate a summary", 
        {"type": "object", "properties": {"summary": {"type": "string"}}}
    )
    print(f"📊 Structured output: {structured}")
    
    # Get model info
    model_info = provider.get_model_info()
    print(f"📋 Model info: {model_info}")
    
    print()


def main():
    """Main demonstration function"""
    print("🚀 MULTI-PROVIDER LLM SUPPORT DEMONSTRATION")
    print("=" * 80)
    print("This example demonstrates the multi-provider LLM support system")
    print("that allows switching between Google Gemini, OpenAI, Anthropic, and local models.")
    print("=" * 80)
    print()
    
    # Demonstrate provider factory
    demonstrate_provider_factory()
    
    # Demonstrate individual providers
    demonstrate_google_provider()
    demonstrate_openai_provider()
    demonstrate_local_provider()
    
    # Demonstrate provider switching
    demonstrate_provider_switching()
    
    # Demonstrate custom provider
    demonstrate_custom_provider()
    
    print("🎉 MULTI-PROVIDER LLM DEMONSTRATION COMPLETE!")
    print("=" * 80)
    print("Key Features Demonstrated:")
    print("• ✅ Provider factory with multiple LLM backends")
    print("• ✅ Configuration validation and environment variable support")
    print("• ✅ Google Gemini, OpenAI, Anthropic, and Local provider support")
    print("• ✅ Dynamic provider switching")
    print("• ✅ Custom provider registration")
    print("• ✅ Structured output support")
    print("• ✅ Model information and capabilities")
    print()
    print("💡 To use with real API keys, set the following environment variables:")
    print("   • GOOGLE_API_KEY for Google Gemini")
    print("   • OPENAI_API_KEY for OpenAI")
    print("   • ANTHROPIC_API_KEY for Anthropic Claude")
    print("   • OLLAMA_BASE_URL for local Ollama server")


if __name__ == "__main__":
    main()