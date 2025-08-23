"""
Example demonstrating RAG Engine usage without API keys.
Uses local Ollama models for LLM generation.
Perfect for development, testing, and demos.
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
from rag_engine.generation.llm_providers import LLMProviderFactory
from rag_engine.core.exceptions import ConfigurationError


def setup_local_llm_config():
    """Setup configuration for local LLM without API keys"""
    print("🔧 SETTING UP LOCAL LLM CONFIGURATION")
    print("=" * 60)
    
    config = PipelineConfig(
        # Use local provider (Ollama)
        llm_provider="local",
        llm_model="llama2",  # or "mistral", "codellama", etc.
        temperature=0.7,
        max_tokens=500,
        
        # Ollama configuration
        ollama_base_url="http://localhost:11434",
        
        # Other RAG settings
        embedding_provider="sentence_transformers",  # No API key needed
        embedding_model="all-MiniLM-L6-v2",
        vector_store_provider="chroma",  # Local vector store
        chunk_size=500,
        chunk_overlap=50
    )
    
    print(f"✅ Configuration created:")
    print(f"   • LLM Provider: {config.llm_provider}")
    print(f"   • LLM Model: {config.llm_model}")
    print(f"   • Embedding Provider: {config.embedding_provider}")
    print(f"   • Vector Store: {config.vector_store_provider}")
    print(f"   • Ollama URL: {config.ollama_base_url}")
    print()
    
    return config


def test_local_llm_provider(config):
    """Test the local LLM provider"""
    print("🤖 TESTING LOCAL LLM PROVIDER")
    print("=" * 60)
    
    try:
        # Create the provider
        provider = LLMProviderFactory.create_provider("local", config)
        print("✅ Local LLM provider created successfully")
        
        # Get model info
        model_info = provider.get_model_info()
        print(f"📊 Model Info:")
        for key, value in model_info.items():
            print(f"   • {key}: {value}")
        print()
        
        # Test basic generation
        print("🔄 Testing text generation...")
        test_prompt = "What is Retrieval-Augmented Generation (RAG)?"
        
        try:
            response = provider.generate(test_prompt)
            print(f"✅ Generation successful!")
            print(f"📝 Response: {response[:200]}...")
            print()
            
        except Exception as e:
            print(f"⚠️  Generation failed: {str(e)}")
            print("   This is expected if Ollama is not running or model is not available")
            print()
        
        # Test structured output
        print("🔄 Testing structured output...")
        schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "key_points": {"type": "array", "items": {"type": "string"}}
            }
        }
        
        try:
            structured_response = provider.generate_with_structured_output(
                "Summarize the concept of RAG in 2-3 sentences with key points",
                schema
            )
            print(f"✅ Structured generation successful!")
            print(f"📊 Structured Response: {structured_response}")
            print()
            
        except Exception as e:
            print(f"⚠️  Structured generation failed: {str(e)}")
            print()
        
    except ConfigurationError as e:
        print(f"❌ Provider creation failed: {str(e)}")
        print("   Make sure langchain-community is installed: uv add langchain-community")
        print()


def demonstrate_mock_provider():
    """Demonstrate a simple mock provider for testing"""
    print("🎭 MOCK PROVIDER FOR TESTING")
    print("=" * 60)
    
    from rag_engine.core.interfaces import BaseLLMProvider
    
    class MockLLMProvider(BaseLLMProvider):
        """Simple mock LLM provider that doesn't require any external services"""
        
        def __init__(self, config):
            self.config = config
            print(f"   MockLLMProvider initialized with model: {config.llm_model}")
        
        def generate(self, prompt, **kwargs):
            # Simple mock responses based on keywords
            prompt_lower = prompt.lower()
            
            if "rag" in prompt_lower or "retrieval" in prompt_lower:
                return ("Retrieval-Augmented Generation (RAG) is a technique that combines "
                       "information retrieval with text generation. It retrieves relevant "
                       "documents from a knowledge base and uses them to generate more "
                       "accurate and contextual responses.")
            
            elif "hello" in prompt_lower or "hi" in prompt_lower:
                return "Hello! I'm a mock LLM provider. I can help you test the RAG system without needing API keys."
            
            elif "python" in prompt_lower:
                return ("Python is a high-level programming language known for its "
                       "simplicity and readability. It's widely used in data science, "
                       "web development, and AI applications.")
            
            else:
                return f"This is a mock response to your query: '{prompt[:50]}...'. In a real scenario, this would be generated by an actual LLM."
        
        def generate_with_structured_output(self, prompt, schema, **kwargs):
            return {
                "mock_response": True,
                "prompt_received": prompt[:50] + "..." if len(prompt) > 50 else prompt,
                "schema_provided": str(schema),
                "note": "This is a mock structured response for testing purposes"
            }
        
        def get_model_info(self):
            return {
                "provider": "mock",
                "model": self.config.llm_model,
                "temperature": self.config.temperature,
                "supports_structured_output": True,
                "note": "Mock provider for testing without API keys"
            }
    
    # Register and test the mock provider
    LLMProviderFactory.register_provider("mock", MockLLMProvider)
    print("✅ Registered mock provider")
    
    config = PipelineConfig(
        llm_provider="mock",
        llm_model="mock-llm-v1",
        temperature=0.7
    )
    
    provider = LLMProviderFactory.create_provider("mock", config)
    print("✅ Created mock provider instance")
    
    # Test various prompts
    test_prompts = [
        "What is RAG?",
        "Hello, how are you?",
        "Tell me about Python programming",
        "Explain quantum computing"
    ]
    
    for prompt in test_prompts:
        response = provider.generate(prompt)
        print(f"📝 Q: {prompt}")
        print(f"   A: {response}")
        print()


def show_setup_instructions():
    """Show setup instructions for different options"""
    print("📋 SETUP INSTRUCTIONS FOR NO-API-KEY USAGE")
    print("=" * 60)
    
    print("🎯 OPTION 1: Ollama (Recommended for real LLM functionality)")
    print("   1. Install Ollama: https://ollama.ai/")
    print("   2. Pull a model: ollama pull llama2")
    print("   3. Start Ollama server: ollama serve")
    print("   4. Install langchain-community: uv add langchain-community")
    print("   5. Use 'local' provider in your config")
    print()
    
    print("🎯 OPTION 2: Mock Provider (For testing/demos)")
    print("   1. Use the built-in mock provider (no installation needed)")
    print("   2. Set llm_provider='mock' in your config")
    print("   3. Perfect for CI/CD, testing, and demos")
    print()
    
    print("🎯 OPTION 3: Sentence Transformers Only")
    print("   1. Use only embedding and retrieval without generation")
    print("   2. Great for search and similarity tasks")
    print("   3. Install: uv add sentence-transformers")
    print()
    
    print("💡 RECOMMENDED MODELS FOR OLLAMA:")
    print("   • llama2 (7B) - Good balance of speed and quality")
    print("   • mistral (7B) - Fast and efficient")
    print("   • codellama (7B) - Good for code-related tasks")
    print("   • llama2:13b - Better quality, slower")
    print()


def main():
    """Main demonstration function"""
    print("🚀 RAG ENGINE WITHOUT API KEYS")
    print("=" * 60)
    print("This example shows how to use the RAG engine without external API keys")
    print("Perfect for development, testing, and demos!")
    print()
    
    # Show setup instructions
    show_setup_instructions()
    
    # Setup local configuration
    config = setup_local_llm_config()
    
    # Test local LLM provider
    test_local_llm_provider(config)
    
    # Demonstrate mock provider
    demonstrate_mock_provider()
    
    print("🎉 NO-API-KEY DEMONSTRATION COMPLETE!")
    print("=" * 60)
    print("✅ You can now run RAG examples without any API keys using:")
    print("   • Ollama for local LLM inference")
    print("   • Mock providers for testing")
    print("   • Sentence Transformers for embeddings")
    print("   • Chroma for local vector storage")
    print()
    print("💡 Next steps:")
    print("   • Install Ollama and pull a model for real LLM functionality")
    print("   • Use mock provider for quick testing and CI/CD")
    print("   • Check other examples for full RAG pipeline demos")


if __name__ == "__main__":
    main()