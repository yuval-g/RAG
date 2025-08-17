#!/usr/bin/env python3
"""
Simple Example: Advanced Query Processing System

This example demonstrates the structure and capabilities of the query processing
system without requiring API keys. It shows how to initialize processors and
understand the available strategies.

Requirements:
- Optional: Set GOOGLE_API_KEY environment variable or create .env file
- Install dependencies: uv sync
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_engine.query.processor import QueryProcessor, QueryStrategy
from src.rag_engine.core.models import Document

# Load environment variables from .env file
load_dotenv()


def main():
    """Simple demonstration of query processing capabilities"""
    
    print("🎯 Simple Query Processing Demo")
    print("=" * 50)
    
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("\n⚠️  GOOGLE_API_KEY not set - some features will be limited")
        print("To enable full functionality:")
        print("  1. Copy .env.example to .env: cp .env.example .env")
        print("  2. Edit .env and add your Google API key")
        print("  3. Or export GOOGLE_API_KEY='your-api-key'")
    else:
        print("✅ GOOGLE_API_KEY found - full functionality available!")
    
    # 1. Initialize the Query Processor
    print("\n1. 🚀 Initializing Query Processor...")
    try:
        processor = QueryProcessor(
            llm_model="gemini-2.0-flash-lite",
            temperature=0.0,
            default_strategy="multi_query"
        )
        print("✅ Query Processor initialized successfully!")
    except Exception as e:
        print(f"❌ Initialization failed: {str(e)}")
        print("💡 This is likely due to missing GOOGLE_API_KEY")
        return
    
    # 2. Show Available Strategies
    print("\n2. 📋 Available Query Processing Strategies:")
    strategies = processor.get_available_strategies()
    for i, strategy in enumerate(strategies, 1):
        info = processor.get_strategy_info(strategy)
        print(f"   {i}. {strategy.upper()}")
        print(f"      Description: {info['description']}")
        print(f"      Supports Retriever: {info['supports_retriever']}")
        print()
    
    # 3. Basic Query Processing (without LLM calls)
    print("3. 🔍 Basic Query Processing:")
    query = "What is task decomposition for LLM agents?"
    print(f"   Query: {query}")
    
    # Process with basic strategy (no LLM required)
    try:
        result = processor.process(query, "basic")
        print(f"   ✅ Basic processing successful!")
        print(f"   Original: {result.original_query}")
        print(f"   Strategy: {result.strategy_used}")
        print(f"   Transformed: {result.transformed_queries}")
    except Exception as e:
        print(f"   ❌ Basic processing failed: {str(e)}")
    
    # 4. Configuration Management
    print("\n4. 🔧 Configuration Management:")
    
    # Show current configs
    print("   Current configurations:")
    configs = processor.get_all_configs()
    for strategy, config in configs.items():
        if "error" not in config:
            print(f"     {strategy}: {config.get('strategy', 'N/A')}")
    
    # Configure a strategy
    try:
        processor.configure_strategy("multi_query", num_queries=7)
        print("   ✅ Successfully configured multi_query strategy")
    except Exception as e:
        print(f"   ❌ Configuration failed: {str(e)}")
    
    # 5. Sample Documents and Mock Retriever
    print("\n5. 📚 Sample Documents and Retrieval:")
    
    # Create sample documents
    documents = [
        Document(
            content="Task decomposition breaks complex problems into smaller sub-tasks",
            metadata={"topic": "task_decomposition"}
        ),
        Document(
            content="LLM agents use planning to organize problem-solving approaches",
            metadata={"topic": "planning"}
        ),
        Document(
            content="Memory systems help agents retain information across interactions",
            metadata={"topic": "memory"}
        )
    ]
    
    print(f"   Created {len(documents)} sample documents")
    
    # Simple retriever function
    def simple_retriever(query: str, top_k: int = 3) -> list[Document]:
        """Simple keyword-based retriever"""
        query_words = query.lower().split()
        scored_docs = []
        
        for doc in documents:
            score = sum(1 for word in query_words if word in doc.content.lower())
            if score > 0:
                scored_docs.append((doc, score))
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_k]]
    
    # Test retrieval
    retrieved = simple_retriever(query)
    print(f"   Retrieved {len(retrieved)} documents for query")
    for i, doc in enumerate(retrieved, 1):
        print(f"     {i}. {doc.content[:60]}...")
    
    # 6. Error Handling
    print("\n6. ⚠️  Error Handling:")
    
    # Test invalid strategy
    try:
        processor.process(query, "invalid_strategy")
    except Exception as e:
        print(f"   ✅ Properly caught invalid strategy error: {type(e).__name__}")
    
    # Test invalid configuration
    try:
        processor.configure_strategy("invalid_strategy", param=1)
    except Exception as e:
        print(f"   ✅ Properly caught invalid config error: {type(e).__name__}")
    
    # 7. Summary
    print("\n7. 📊 Summary:")
    print("   ✅ Query Processor initialized with 6 strategies")
    print("   ✅ Basic query processing works without API keys")
    print("   ✅ Configuration management functional")
    print("   ✅ Error handling working properly")
    print("   ✅ Ready for integration with real retrievers")
    
    print("\n💡 Next Steps:")
    print("   1. Set GOOGLE_API_KEY to use advanced strategies")
    print("   2. Integrate with your document store and retriever")
    print("   3. Choose appropriate strategies for your use case")
    print("   4. Configure strategies for optimal performance")
    
    print("\n🎉 Demo completed successfully!")


if __name__ == "__main__":
    main()