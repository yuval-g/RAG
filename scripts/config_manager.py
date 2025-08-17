#!/usr/bin/env python3
"""
Configuration Management CLI Tool

This script provides a command-line interface for managing RAG system configurations.
"""

import argparse
import sys
import json
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_engine.core.config import (
    ConfigurationManager, 
    PipelineConfig, 
    ConfigurationError
)


def validate_config(args):
    """Validate a configuration file"""
    config_manager = ConfigurationManager()
    
    try:
        is_valid = config_manager.validate_config_file(args.config_file)
        if is_valid:
            print(f"✓ Configuration file {args.config_file} is valid")
            return 0
        else:
            print(f"❌ Configuration file {args.config_file} is invalid")
            return 1
    except Exception as e:
        print(f"❌ Error validating configuration: {e}")
        return 1


def show_config(args):
    """Show current configuration"""
    try:
        config_manager = ConfigurationManager(args.config_file, args.environment)
        config = config_manager.load_config()
        
        print(f"Configuration loaded from: {args.config_file or 'defaults'}")
        print(f"Environment: {config.environment}")
        print()
        
        # Display key configuration values
        print("LLM Configuration:")
        print(f"  Provider: {config.llm_provider}")
        print(f"  Model: {config.llm_model}")
        print(f"  Temperature: {config.temperature}")
        print()
        
        print("Embedding Configuration:")
        print(f"  Provider: {config.embedding_provider}")
        print(f"  Model: {config.embedding_model}")
        print()
        
        print("Vector Store Configuration:")
        print(f"  Provider: {config.vector_store}")
        print()
        
        print("Indexing Configuration:")
        print(f"  Strategy: {config.indexing_strategy}")
        print(f"  Chunk Size: {config.chunk_size}")
        print(f"  Chunk Overlap: {config.chunk_overlap}")
        print()
        
        print("Retrieval Configuration:")
        print(f"  Retrieval K: {config.retrieval_k}")
        print(f"  Use Reranking: {config.use_reranking}")
        print(f"  Hybrid Retrieval: {config.enable_hybrid_retrieval}")
        print()
        
        print("Production Configuration:")
        print(f"  Logging: {config.enable_logging}")
        print(f"  Log Level: {config.log_level}")
        print(f"  Metrics: {config.enable_metrics}")
        print(f"  Caching: {config.enable_caching}")
        
        return 0
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return 1


def create_config(args):
    """Create a new configuration file"""
    try:
        # Create default configuration
        config = PipelineConfig()
        
        # Apply any overrides from command line
        if args.llm_provider:
            config.llm_provider = args.llm_provider
        if args.llm_model:
            config.llm_model = args.llm_model
        if args.temperature is not None:
            config.temperature = args.temperature
        if args.chunk_size:
            config.chunk_size = args.chunk_size
        if args.retrieval_k:
            config.retrieval_k = args.retrieval_k
        
        # Save configuration
        config_manager = ConfigurationManager()
        config_manager.save_config(config, args.output_file, exclude_secrets=not args.include_secrets)
        
        print(f"✓ Configuration created: {args.output_file}")
        return 0
    except Exception as e:
        print(f"❌ Error creating configuration: {e}")
        return 1


def list_environments(args):
    """List available environment configurations"""
    try:
        config_manager = ConfigurationManager()
        environments = config_manager.list_available_environments(args.config_dir)
        
        if environments:
            print(f"Available environments in {args.config_dir}:")
            for env in environments:
                print(f"  - {env}")
        else:
            print(f"No environment configurations found in {args.config_dir}")
        
        return 0
    except Exception as e:
        print(f"❌ Error listing environments: {e}")
        return 1


def show_schema(args):
    """Show configuration schema"""
    try:
        config_manager = ConfigurationManager()
        schema = config_manager.get_config_schema()
        
        if args.format == 'json':
            print(json.dumps(schema, indent=2))
        else:
            print("Configuration Schema:")
            print("=" * 50)
            
            properties = schema.get("properties", {})
            for prop_name, prop_info in properties.items():
                prop_type = prop_info.get("type", "unknown")
                prop_default = prop_info.get("default", "no default")
                description = prop_info.get("description", "")
                
                print(f"{prop_name}:")
                print(f"  Type: {prop_type}")
                print(f"  Default: {prop_default}")
                if description:
                    print(f"  Description: {description}")
                print()
        
        return 0
    except Exception as e:
        print(f"❌ Error showing schema: {e}")
        return 1


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="RAG System Configuration Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a configuration file
  python config_manager.py validate config/example_config.yaml
  
  # Show current configuration
  python config_manager.py show --config-file config/example_config.yaml
  
  # Show configuration for specific environment
  python config_manager.py show --config-file config/config.yaml --environment production
  
  # Create a new configuration file
  python config_manager.py create --output config/new_config.yaml --llm-provider openai
  
  # List available environments
  python config_manager.py list-environments --config-dir config/
  
  # Show configuration schema
  python config_manager.py schema
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate a configuration file')
    validate_parser.add_argument('config_file', help='Path to configuration file')
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show configuration')
    show_parser.add_argument('--config-file', help='Path to configuration file')
    show_parser.add_argument('--environment', help='Environment name')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create a new configuration file')
    create_parser.add_argument('--output', dest='output_file', required=True, help='Output file path')
    create_parser.add_argument('--llm-provider', help='LLM provider')
    create_parser.add_argument('--llm-model', help='LLM model')
    create_parser.add_argument('--temperature', type=float, help='Temperature')
    create_parser.add_argument('--chunk-size', type=int, help='Chunk size')
    create_parser.add_argument('--retrieval-k', type=int, help='Retrieval K')
    create_parser.add_argument('--include-secrets', action='store_true', help='Include API keys in output')
    
    # List environments command
    list_parser = subparsers.add_parser('list-environments', help='List available environment configurations')
    list_parser.add_argument('--config-dir', default='config/', help='Configuration directory')
    
    # Schema command
    schema_parser = subparsers.add_parser('schema', help='Show configuration schema')
    schema_parser.add_argument('--format', choices=['text', 'json'], default='text', help='Output format')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    if args.command == 'validate':
        return validate_config(args)
    elif args.command == 'show':
        return show_config(args)
    elif args.command == 'create':
        return create_config(args)
    elif args.command == 'list-environments':
        return list_environments(args)
    elif args.command == 'schema':
        return show_schema(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())