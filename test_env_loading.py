#!/usr/bin/env python3
"""
Test script to verify environment variable loading works correctly.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_env_loading():
    """Test that environment variables are loaded correctly"""
    print("🧪 Testing environment variable loading...")
    
    # Test 1: Direct import of env_loader
    print("\n1. Testing direct env_loader import...")
    from rag_engine.core.env_loader import ensure_env_loaded
    ensure_env_loaded()
    print("   ✅ env_loader imported and executed successfully")
    
    # Test 2: Import config (which should load env)
    print("\n2. Testing config import (should load env)...")
    from rag_engine.core.config import ConfigurationManager
    print("   ✅ Config imported successfully")
    
    # Test 3: Import LLM providers (which should load env)
    print("\n3. Testing LLM providers import (should load env)...")
    from rag_engine.generation.llm_providers import LLMProviderFactory
    print("   ✅ LLM providers imported successfully")
    
    # Test 4: Check if some common env vars are accessible
    print("\n4. Testing environment variable access...")
    test_vars = [
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY", 
        "LANGFUSE_SECRET_KEY",
        "RAG_ENVIRONMENT"
    ]
    
    for var in test_vars:
        value = os.getenv(var)
        if value:
            print(f"   ✅ {var}: {'*' * min(len(value), 10)}... (found)")
        else:
            print(f"   ⚪ {var}: (not set)")
    
    # Test 5: Test CLI import
    print("\n5. Testing CLI import (should load env early)...")
    try:
        from rag_engine.cli.main import cli
        print("   ✅ CLI imported successfully")
    except Exception as e:
        print(f"   ❌ CLI import failed: {e}")
    
    # Test 6: Test API import
    print("\n6. Testing API import (should load env early)...")
    try:
        from rag_engine.api.app import create_app
        print("   ✅ API imported successfully")
    except Exception as e:
        print(f"   ❌ API import failed: {e}")
    
    print("\n🎉 Environment loading test completed!")
    print("\n💡 Tips:")
    print("   • Create a .env file in the project root with your API keys")
    print("   • Environment variables are now loaded early in all entry points")
    print("   • The env_loader module ensures consistent loading across the codebase")


if __name__ == "__main__":
    test_env_loading()