#!/usr/bin/env python3
"""
Example demonstrating the configuration management system

This example shows how to:
1. Load configurations from files and environment variables
2. Validate configuration schemas
3. Create environment-specific configurations
4. Save and manage configuration files
"""

import os
import tempfile
import json
from pathlib import Path

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_engine.core.config import (
    ConfigurationManager, 
    PipelineConfig, 
    ConfigurationError
)


def example_basic_configuration():
    """Example: Basic configuration loading"""
    print("=== Basic Configuration Loading ===")
    
    # Create a configuration manager
    config_manager = ConfigurationManager()
    
    # Load default configuration
    config = config_manager.load_config()
    
    print(f"Default LLM Provider: {config.llm_provider}")
    print(f"Default Model: {config.llm_model}")
    print(f"Default Chunk Size: {config.chunk_size}")
    print(f"Default Retrieval K: {config.retrieval_k}")
    print()


def example_file_configuration():
    """Example: Loading configuration from file"""
    print("=== File Configuration Loading ===")
    
    # Use the example configuration file
    config_path = "config/example_config.yaml"
    
    if Path(config_path).exists():
        config_manager = ConfigurationManager(config_path)
        config = config_manager.load_config()
        
        print(f"Loaded from {config_path}")
        print(f"LLM Provider: {config.llm_provider}")
        print(f"Model: {config.llm_model}")
        print(f"Temperature: {config.temperature}")
        print(f"Vector Store: {config.vector_store}")
    else:
        print(f"Configuration file {config_path} not found")
    print()


def example_environment_variables():
    """Example: Configuration with environment variables"""
    print("=== Environment Variable Configuration ===")
    
    # Set some environment variables
    env_vars = {
        'RAG_LLM_PROVIDER': 'openai',
        'RAG_LLM_MODEL': 'gpt-4',
        'RAG_TEMPERATURE': '0.7',
        'RAG_CHUNK_SIZE': '1500',
        'RAG_RETRIEVAL_K': '8',
        'RAG_ENABLE_LOGGING': 'false',
        'OPENAI_API_KEY': 'test-key-123'
    }
    
    # Temporarily set environment variables
    original_env = {}
    for key, value in env_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    try:
        config_manager = ConfigurationManager()
        config = config_manager.load_config()
        
        print("Configuration loaded with environment variables:")
        print(f"LLM Provider: {config.llm_provider}")
        print(f"Model: {config.llm_model}")
        print(f"Temperature: {config.temperature}")
        print(f"Chunk Size: {config.chunk_size}")
        print(f"Retrieval K: {config.retrieval_k}")
        print(f"Enable Logging: {config.enable_logging}")
        print(f"OpenAI API Key: {config.openai_api_key}")
    finally:
        # Restore original environment
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print()


def example_environment_specific_config():
    """Example: Environment-specific configuration"""
    print("=== Environment-Specific Configuration ===")
    
    # Test with different environments
    environments = ["development", "production", "testing"]
    
    for env in environments:
        config_path = f"config/config.{env}.yaml"
        
        if Path(config_path).exists():
            config_manager = ConfigurationManager(config_path, env)
            config = config_manager.load_config()
            
            print(f"{env.upper()} Environment:")
            print(f"  Environment: {config.environment}")
            print(f"  Log Level: {config.log_level}")
            print(f"  Chunk Size: {config.chunk_size}")
            print(f"  Retrieval K: {config.retrieval_k}")
            print(f"  Enable Metrics: {config.enable_metrics}")
            print(f"  Enable Caching: {config.enable_caching}")
        else:
            print(f"{env.upper()} Environment: Configuration file not found")
        print()


def example_configuration_validation():
    """Example: Configuration validation"""
    print("=== Configuration Validation ===")
    
    config_manager = ConfigurationManager()
    
    # Test valid configuration
    valid_config = {
        "llm_provider": "google",
        "temperature": 0.5,
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "retrieval_k": 5
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(valid_config, f)
        valid_config_path = f.name
    
    try:
        is_valid = config_manager.validate_config_file(valid_config_path)
        print(f"Valid configuration validation result: {is_valid}")
    finally:
        os.unlink(valid_config_path)
    
    # Test invalid configuration
    invalid_config = {
        "llm_provider": "invalid_provider",
        "temperature": 5.0,  # Too high
        "chunk_size": -100   # Negative
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(invalid_config, f)
        invalid_config_path = f.name
    
    try:
        is_valid = config_manager.validate_config_file(invalid_config_path)
        print(f"Invalid configuration validation result: {is_valid}")
    finally:
        os.unlink(invalid_config_path)
    print()


def example_save_configuration():
    """Example: Saving configuration"""
    print("=== Saving Configuration ===")
    
    # Create a custom configuration
    config = PipelineConfig(
        llm_provider="openai",
        llm_model="gpt-4",
        temperature=0.3,
        chunk_size=800,
        retrieval_k=7,
        enable_metrics=True,
        openai_api_key="secret-key-123"
    )
    
    config_manager = ConfigurationManager()
    
    # Save with secrets excluded (default)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        safe_config_path = f.name
    
    config_manager.save_config(config, safe_config_path, exclude_secrets=True)
    
    print(f"Configuration saved to {safe_config_path} (secrets excluded)")
    
    # Read and display the saved configuration
    with open(safe_config_path, 'r') as f:
        content = f.read()
        print("Saved configuration content (first 500 chars):")
        print(content[:500])
        if "***REDACTED***" in content:
            print("✓ Secrets were properly redacted")
    
    # Clean up
    os.unlink(safe_config_path)
    print()


def example_merge_configurations():
    """Example: Merging configurations"""
    print("=== Merging Configurations ===")
    
    config_manager = ConfigurationManager()
    
    base_config = {
        "llm_provider": "google",
        "temperature": 0.0,
        "chunk_size": 1000,
        "vector_store_config": {
            "host": "localhost",
            "port": 8000,
            "collection": "base"
        }
    }
    
    override_config = {
        "temperature": 0.5,
        "retrieval_k": 10,
        "vector_store_config": {
            "port": 9000,
            "ssl": True
        }
    }
    
    merged = config_manager.merge_configs(base_config, override_config)
    
    print("Base configuration:")
    print(json.dumps(base_config, indent=2))
    print("\nOverride configuration:")
    print(json.dumps(override_config, indent=2))
    print("\nMerged configuration:")
    print(json.dumps(merged, indent=2))
    print()


def example_configuration_schema():
    """Example: Getting configuration schema"""
    print("=== Configuration Schema ===")
    
    config_manager = ConfigurationManager()
    schema = config_manager.get_config_schema()
    
    print("Configuration schema properties:")
    for prop_name, prop_info in schema.get("properties", {}).items():
        prop_type = prop_info.get("type", "unknown")
        prop_default = prop_info.get("default", "no default")
        print(f"  {prop_name}: {prop_type} (default: {prop_default})")
    
    print(f"\nTotal properties: {len(schema.get('properties', {}))}")
    print()


def example_error_handling():
    """Example: Error handling"""
    print("=== Error Handling ===")
    
    config_manager = ConfigurationManager()
    
    # Test missing file
    try:
        config_manager.load_config("/nonexistent/config.yaml")
    except ConfigurationError as e:
        print(f"✓ Caught expected error for missing file: {e}")
    
    # Test invalid configuration
    invalid_config = {
        "llm_provider": "invalid_provider",
        "temperature": 10.0
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(invalid_config, f)
        invalid_config_path = f.name
    
    try:
        config_manager.load_config(invalid_config_path)
    except ConfigurationError as e:
        print(f"✓ Caught expected error for invalid config: {e}")
    finally:
        os.unlink(invalid_config_path)
    
    print()


def main():
    """Run all configuration management examples"""
    print("Configuration Management System Examples")
    print("=" * 50)
    print()
    
    try:
        example_basic_configuration()
        example_file_configuration()
        example_environment_variables()
        example_environment_specific_config()
        example_configuration_validation()
        example_save_configuration()
        example_merge_configurations()
        example_configuration_schema()
        example_error_handling()
        
        print("✓ All configuration management examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()