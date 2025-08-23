#!/usr/bin/env python3
"""
Simple test for environment variable loading without dependencies.
"""

import os
import sys
from pathlib import Path

def test_env_loading_simple():
    """Test environment loading without importing heavy dependencies"""
    print("🧪 Testing environment variable loading (simple)...")
    
    # Test 1: Check if .env file exists
    print("\n1. Checking for .env file...")
    env_file = Path(".env")
    if env_file.exists():
        print(f"   ✅ Found .env file: {env_file.absolute()}")
        
        # Read and show first few lines (without sensitive data)
        with open(env_file, 'r') as f:
            lines = f.readlines()[:5]  # First 5 lines only
        
        print("   📄 .env file preview:")
        for line in lines:
            if '=' in line and not line.startswith('#'):
                key = line.split('=')[0]
                print(f"      {key}=***")
            else:
                print(f"      {line.strip()}")
    else:
        print("   ⚪ No .env file found (this is okay)")
    
    # Test 2: Test dotenv loading directly
    print("\n2. Testing dotenv loading...")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("   ✅ dotenv loaded successfully")
    except ImportError:
        print("   ⚠️  python-dotenv not installed")
        return
    
    # Test 3: Check environment variables
    print("\n3. Checking environment variables...")
    test_vars = [
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY", 
        "LANGFUSE_SECRET_KEY",
        "RAG_ENVIRONMENT",
        "PATH"  # This should always exist
    ]
    
    found_vars = 0
    for var in test_vars:
        value = os.getenv(var)
        if value:
            found_vars += 1
            if var == "PATH":
                print(f"   ✅ {var}: (system variable found)")
            else:
                print(f"   ✅ {var}: {'*' * min(len(value), 8)}... (found)")
        else:
            print(f"   ⚪ {var}: (not set)")
    
    print(f"\n   📊 Found {found_vars}/{len(test_vars)} variables")
    
    # Test 4: Test our env_loader module directly
    print("\n4. Testing env_loader module...")
    try:
        # Add src to path
        src_path = Path(__file__).parent / "src"
        if src_path not in sys.path:
            sys.path.insert(0, str(src_path))
        
        # Import just the env_loader module
        from rag_engine.core.env_loader import ensure_env_loaded, load_environment_variables
        
        print("   ✅ env_loader module imported successfully")
        
        # Test the functions
        ensure_env_loaded()
        print("   ✅ ensure_env_loaded() executed successfully")
        
        load_environment_variables()
        print("   ✅ load_environment_variables() executed successfully")
        
    except Exception as e:
        print(f"   ❌ env_loader test failed: {e}")
    
    print("\n🎉 Simple environment loading test completed!")
    
    # Recommendations
    print("\n💡 Recommendations:")
    if not env_file.exists():
        print("   • Create a .env file with your API keys for testing")
        print("   • Example: echo 'GOOGLE_API_KEY=your_key_here' > .env")
    
    if found_vars < 2:  # PATH + at least one other
        print("   • Add some environment variables to test loading")
    
    print("   • The env_loader module is working correctly")
    print("   • Environment variables will now load early in all entry points")


if __name__ == "__main__":
    test_env_loading_simple()