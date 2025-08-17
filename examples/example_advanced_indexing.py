#!/usr/bin/env python3
"""
Advanced Indexing Strategies Example

This example demonstrates the advanced indexing capabilities of the RAG engine,
including multi-representation indexing, ColBERT token-level precision, 
hierarchical RAPTOR indexing, and the unified IndexingManager.

Requirements:
- Set GOOGLE_API_KEY environment variable for Gemini access
- Optional: Install ragatouille for ColBERT functionality (uv add ragatouille)
- Install dependencies: uv sync
"""

import os
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_engine.indexing.indexing_manager import IndexingManager, IndexingStrategy
from src.rag_engine.core.models import Document
from src.rag_engine.core.config import PipelineConfig

# Load environment variables from .env file
load_dotenv()


def create_sample_documents() -> List[Document]:
    """Create a comprehensive set of sample documents for indexing"""
    documents = [
        Document(
            content="""
            Task decomposition is a fundamental technique in artificial intelligence where complex problems 
            are systematically broken down into smaller, more manageable sub-tasks. This approach enables 
            AI agents to handle sophisticated challenges through step-by-step problem solving. The process 
            involves identifying the main goal, analyzing its components, and creating a hierarchical 
            structure of sub-goals that can be addressed individually.
            """.strip(),
            metadata={"title": "Task Decomposition in AI", "author": "Dr. Smith", "category": "AI Fundamentals"},
            doc_id="doc_001"
        ),
        Document(
            content="""
            Large Language Model (LLM) agents utilize sophisticated planning mechanisms to organize their 
            approach to complex problems. These systems employ hierarchical planning, where high-level 
            goals are broken down into intermediate objectives and specific actions. The planning process 
            involves goal decomposition, resource allocation, and temporal sequencing to ensure efficient 
            task execution.
            """.strip(),
            metadata={"title": "LLM Agent Planning", "author": "Prof. Johnson", "category": "LLM Systems"},
            doc_id="doc_002"
        ),
        Document(
            content="""
            Memory systems in AI agents serve as crucial components for maintaining context and learning 
            from past interactions. These systems typically include short-term working memory for 
            immediate task processing and long-term episodic memory for experience retention. Advanced 
            memory architectures also incorporate semantic memory for factual knowledge and procedural 
            memory for skill acquisition.
            """.strip(),
            metadata={"title": "AI Agent Memory Systems", "author": "Dr. Chen", "category": "Cognitive Architecture"},
            doc_id="doc_003"
        ),
        Document(
            content="""
            Tool use capabilities enable AI agents to extend their functionality beyond inherent language 
            processing abilities. This includes API interactions, database queries, web searches, file 
            operations, and integration with external systems and services. Tool use requires careful 
            planning, error handling, and security considerations to ensure safe and effective operation.
            """.strip(),
            metadata={"title": "Tool Use in AI Agents", "author": "Dr. Williams", "category": "Agent Capabilities"},
            doc_id="doc_004"
        ),
        Document(
            content="""
            Retrieval-Augmented Generation (RAG) combines the power of large language models with external 
            knowledge retrieval systems. This approach helps ground AI responses in factual, up-to-date 
            information from curated knowledge bases. RAG systems typically involve document indexing, 
            query processing, similarity search, and context integration for enhanced response quality.
            """.strip(),
            metadata={"title": "Retrieval-Augmented Generation", "author": "Prof. Davis", "category": "RAG Systems"},
            doc_id="doc_005"
        ),
        Document(
            content="""
            Vector embeddings represent text as high-dimensional numerical vectors, enabling semantic 
            similarity search and machine learning operations. Modern embedding models capture contextual 
            relationships, semantic meaning, and syntactic patterns in text. These representations are 
            fundamental to information retrieval, recommendation systems, and natural language understanding.
            """.strip(),
            metadata={"title": "Vector Embeddings", "author": "Dr. Brown", "category": "NLP Technology"},
            doc_id="doc_006"
        ),
        Document(
            content="""
            Query transformation techniques improve information retrieval by reformulating user questions 
            to better match relevant documents. Methods include query expansion with synonyms, 
            reformulation for clarity, hypothetical document generation, and multi-query approaches. 
            These techniques help overcome vocabulary mismatch and improve retrieval accuracy.
            """.strip(),
            metadata={"title": "Query Transformation", "author": "Prof. Wilson", "category": "Information Retrieval"},
            doc_id="doc_007"
        ),
        Document(
            content="""
            Evaluation of RAG systems involves measuring both retrieval quality and generation quality. 
            Retrieval metrics include precision, recall, and mean reciprocal rank, while generation 
            metrics assess answer relevance, factual accuracy, and completeness. Comprehensive evaluation 
            requires both automated metrics and human judgment to ensure system effectiveness.
            """.strip(),
            metadata={"title": "RAG System Evaluation", "author": "Dr. Taylor", "category": "System Evaluation"},
            doc_id="doc_008"
        ),
        Document(
            content="""
            Multi-modal AI systems integrate text, images, audio, and other data types for comprehensive 
            understanding and generation. These systems require specialized architectures for cross-modal 
            alignment, fusion techniques for combining different modalities, and evaluation methods that 
            account for multi-modal complexity. Applications include visual question answering and 
            multimedia content generation.
            """.strip(),
            metadata={"title": "Multi-modal AI Systems", "author": "Prof. Anderson", "category": "Multi-modal AI"},
            doc_id="doc_009"
        ),
        Document(
            content="""
            Reinforcement Learning from Human Feedback (RLHF) is a training paradigm that aligns AI 
            systems with human preferences and values. The process involves collecting human feedback on 
            model outputs, training a reward model to predict human preferences, and using reinforcement 
            learning to optimize the model according to the learned reward function. This approach 
            improves model safety and alignment.
            """.strip(),
            metadata={"title": "RLHF Training", "author": "Dr. Martinez", "category": "AI Training"},
            doc_id="doc_010"
        )
    ]
    return documents


def print_separator(title: str, char: str = "="):
    """Print a formatted separator"""
    print(f"\n{char*60}")
    print(f" {title}")
    print(f"{char*60}")


def demonstrate_indexing_manager_basics():
    """Demonstrate basic IndexingManager functionality"""
    print_separator("IndexingManager Basics")
    
    # Create configuration
    config = PipelineConfig(
        llm_model="gemini-2.0-flash-lite",
        embedding_model="models/embedding-001",
        temperature=0.0,
        indexing_strategy="basic",
        chunk_size=500,
        chunk_overlap=100
    )
    
    # Initialize IndexingManager
    print("🚀 Initializing IndexingManager...")
    manager = IndexingManager(config)
    
    # Show available strategies
    strategies = manager.list_strategies()
    print(f"📋 Available Indexing Strategies: {', '.join(strategies)}")
    
    # Show active strategy
    active = manager.get_active_strategy()
    print(f"🎯 Active Strategy: {active}")
    
    # Get strategy information
    print("\n📊 Strategy Information:")
    all_info = manager.get_all_strategies_info()
    for strategy, info in all_info.items():
        status = "✅ Active" if info["is_active"] else "⚪ Available"
        init_status = "🔧 Initialized" if info["is_initialized"] else "💤 Not Initialized"
        print(f"   {strategy}: {status}, {init_status}")
    
    return manager, config


def demonstrate_basic_indexing(manager: IndexingManager, documents: List[Document]):
    """Demonstrate basic indexing strategy"""
    print_separator("Basic Indexing Strategy")
    
    print("📚 Indexing documents with Basic strategy...")
    
    try:
        # Index documents with basic strategy
        success = manager.index_documents(documents, strategy="basic")
        
        if success:
            print("✅ Basic indexing completed successfully!")
            
            # Get document count
            count = manager.get_document_count("basic")
            print(f"📊 Indexed {count} documents")
            
            # Get strategy info
            info = manager.get_strategy_info("basic")
            print(f"📈 Strategy Info: {info}")
            
            # Get retriever for basic strategy
            retriever = manager.get_indexer_for_retrieval("basic")
            if retriever:
                print("🔍 Basic retriever available for search operations")
            else:
                print("❌ No retriever available")
                
        else:
            print("❌ Basic indexing failed")
            
    except Exception as e:
        print(f"❌ Basic indexing error: {str(e)}")


def demonstrate_multi_representation_indexing(manager: IndexingManager, documents: List[Document]):
    """Demonstrate multi-representation indexing strategy"""
    print_separator("Multi-Representation Indexing")
    
    print("🔄 Indexing documents with Multi-Representation strategy...")
    print("This strategy creates summaries for retrieval while storing full documents for generation.")
    
    try:
        # Switch to multi-representation strategy
        manager.set_active_strategy("multi_representation")
        
        # Index documents
        success = manager.index_documents(documents[:5])  # Use fewer docs for demo
        
        if success:
            print("✅ Multi-representation indexing completed!")
            
            # Get indexer for detailed info
            indexer = manager.get_indexer("multi_representation")
            
            # Show summary count vs document count
            if hasattr(indexer, 'get_summary_count'):
                summary_count = indexer.get_summary_count()
                doc_count = indexer.get_docstore_count()
                print(f"📊 Created {summary_count} summaries for {doc_count} documents")
            
            # Get retriever
            retriever = manager.get_indexer_for_retrieval("multi_representation")
            if retriever:
                print("🔍 Multi-vector retriever available")
                print("   → Searches over summaries, returns full documents")
            
        else:
            print("❌ Multi-representation indexing failed")
            
    except Exception as e:
        print(f"❌ Multi-representation indexing error: {str(e)}")
        print("💡 This requires GOOGLE_API_KEY for document summarization")


def demonstrate_colbert_indexing(manager: IndexingManager, documents: List[Document]):
    """Demonstrate ColBERT token-level indexing"""
    print_separator("ColBERT Token-Level Indexing")
    
    print("🎯 Indexing documents with ColBERT strategy...")
    print("This strategy provides token-level precision for fine-grained retrieval.")
    
    try:
        # Check if ColBERT is ready
        is_ready = manager.is_strategy_ready("colbert")
        print(f"🔧 ColBERT readiness: {'✅ Ready' if is_ready else '❌ Not Ready'}")
        
        if not is_ready:
            print("💡 ColBERT requires 'ragatouille' package: uv add ragatouille")
            return
        
        # Switch to ColBERT strategy
        manager.set_active_strategy("colbert")
        
        # Index documents (use fewer for demo due to processing time)
        success = manager.index_documents(documents[:3])
        
        if success:
            print("✅ ColBERT indexing completed!")
            
            # Get indexer info
            indexer = manager.get_indexer("colbert")
            if hasattr(indexer, 'get_index_info'):
                info = indexer.get_index_info()
                print(f"📊 ColBERT Index Info:")
                print(f"   Index Name: {info.get('index_name', 'N/A')}")
                print(f"   Documents: {info.get('document_count', 0)}")
                print(f"   Max Doc Length: {info.get('max_document_length', 'N/A')}")
            
            # Get retriever
            retriever = manager.get_indexer_for_retrieval("colbert")
            if retriever:
                print("🔍 ColBERT retriever available for token-level search")
            
        else:
            print("❌ ColBERT indexing failed")
            
    except Exception as e:
        print(f"❌ ColBERT indexing error: {str(e)}")
        print("💡 Install ragatouille: uv add ragatouille")


def demonstrate_raptor_indexing(manager: IndexingManager, documents: List[Document]):
    """Demonstrate RAPTOR hierarchical indexing"""
    print_separator("RAPTOR Hierarchical Indexing")
    
    print("🌳 Indexing documents with RAPTOR strategy...")
    print("This strategy creates hierarchical tree structures with clustered summaries.")
    
    try:
        # Switch to RAPTOR strategy
        manager.set_active_strategy("raptor")
        
        # Index documents
        success = manager.index_documents(documents)
        
        if success:
            print("✅ RAPTOR indexing completed!")
            
            # Get indexer for tree info
            indexer = manager.get_indexer("raptor")
            if hasattr(indexer, 'get_tree_info'):
                tree_info = indexer.get_tree_info()
                print(f"📊 RAPTOR Tree Structure:")
                print(f"   Total Nodes: {tree_info.get('total_nodes', 0)}")
                print(f"   Tree Levels: {tree_info.get('total_levels', 0)}")
                print(f"   Root Nodes: {tree_info.get('root_nodes', 0)}")
                
                nodes_per_level = tree_info.get('nodes_per_level', {})
                for level, count in nodes_per_level.items():
                    print(f"   Level {level}: {count} nodes")
            
            # Get retriever (RAPTOR returns the indexer itself)
            retriever = manager.get_indexer_for_retrieval("raptor")
            if retriever:
                print("🔍 RAPTOR hierarchical search available")
                print("   → Supports level-specific and hierarchical search")
            
        else:
            print("❌ RAPTOR indexing failed")
            
    except Exception as e:
        print(f"❌ RAPTOR indexing error: {str(e)}")
        print("💡 This requires GOOGLE_API_KEY for summarization and scikit-learn for clustering")


def demonstrate_strategy_comparison(manager: IndexingManager, documents: List[Document]):
    """Compare different indexing strategies"""
    print_separator("Strategy Comparison")
    
    print("📊 Comparing indexing strategies...")
    
    # Test each strategy with a subset of documents
    test_docs = documents[:3]  # Use fewer docs for comparison
    
    strategies_to_test = ["basic", "multi_representation"]  # Start with these
    
    results = {}
    
    for strategy in strategies_to_test:
        print(f"\n🔄 Testing {strategy} strategy...")
        
        try:
            # Clear previous index
            manager.clear_index(strategy)
            
            # Set active strategy
            manager.set_active_strategy(strategy)
            
            # Index documents
            success = manager.index_documents(test_docs, strategy=strategy)
            
            if success:
                # Get strategy info
                info = manager.get_strategy_info(strategy)
                results[strategy] = {
                    "success": True,
                    "document_count": info.get("document_count", 0),
                    "indexer_class": info.get("indexer_class", "Unknown"),
                    "is_ready": manager.is_strategy_ready(strategy)
                }
                print(f"   ✅ Success: {info.get('document_count', 0)} documents indexed")
            else:
                results[strategy] = {"success": False, "error": "Indexing failed"}
                print(f"   ❌ Failed")
                
        except Exception as e:
            results[strategy] = {"success": False, "error": str(e)}
            print(f"   ❌ Error: {str(e)}")
    
    # Print comparison table
    print(f"\n📋 Strategy Comparison Results:")
    print(f"{'Strategy':<20} {'Status':<10} {'Documents':<10} {'Ready':<8}")
    print("-" * 50)
    
    for strategy, result in results.items():
        if result["success"]:
            status = "✅ Success"
            doc_count = result.get("document_count", 0)
            ready = "✅" if result.get("is_ready", False) else "❌"
        else:
            status = "❌ Failed"
            doc_count = "N/A"
            ready = "❌"
        
        print(f"{strategy:<20} {status:<10} {doc_count:<10} {ready:<8}")


def demonstrate_advanced_features(manager: IndexingManager):
    """Demonstrate advanced IndexingManager features"""
    print_separator("Advanced Features")
    
    print("🔧 Advanced IndexingManager capabilities...")
    
    # Strategy readiness check
    print("\n🔍 Strategy Readiness Check:")
    for strategy in manager.list_strategies():
        ready = manager.is_strategy_ready(strategy)
        status = "✅ Ready" if ready else "❌ Not Ready"
        print(f"   {strategy}: {status}")
    
    # Clear all indexes
    print("\n🧹 Clearing all indexes...")
    success = manager.clear_all_indexes()
    if success:
        print("✅ All indexes cleared successfully")
    else:
        print("❌ Failed to clear some indexes")
    
    # Strategy switching
    print("\n🔄 Strategy Switching:")
    original_strategy = manager.get_active_strategy()
    print(f"   Original: {original_strategy}")
    
    # Switch to different strategy
    new_strategy = "multi_representation"
    success = manager.set_active_strategy(new_strategy)
    if success:
        print(f"   ✅ Switched to: {manager.get_active_strategy()}")
    else:
        print(f"   ❌ Failed to switch to: {new_strategy}")
    
    # Switch back
    manager.set_active_strategy(original_strategy)
    print(f"   🔙 Restored to: {manager.get_active_strategy()}")


def demonstrate_error_handling(manager: IndexingManager):
    """Demonstrate error handling capabilities"""
    print_separator("Error Handling")
    
    print("⚠️  Testing error handling...")
    
    # Test invalid strategy
    try:
        manager.get_indexer("invalid_strategy")
        print("❌ Should have failed with invalid strategy")
    except ValueError as e:
        print(f"✅ Properly caught invalid strategy: {type(e).__name__}")
    
    # Test invalid strategy switch
    success = manager.set_active_strategy("nonexistent_strategy")
    if not success:
        print("✅ Properly rejected invalid strategy switch")
    else:
        print("❌ Should have rejected invalid strategy")
    
    # Test empty document list
    success = manager.index_documents([])
    if success:
        print("✅ Properly handled empty document list")
    else:
        print("❌ Should have handled empty document list gracefully")


def main():
    """Main demonstration function"""
    print("🏗️  Advanced Indexing Strategies - Comprehensive Demo")
    print("This demo showcases all advanced indexing capabilities")
    
    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("\n⚠️  Warning: GOOGLE_API_KEY not set!")
        print("Advanced strategies (multi-representation, RAPTOR) require Google API access.")
        print("To set up your API key:")
        print("  1. Copy .env.example to .env: cp .env.example .env")
        print("  2. Edit .env and add your Google API key")
        print("  3. Or export GOOGLE_API_KEY='your-api-key'")
        print("\nContinuing with available functionality...")
    else:
        print("✅ GOOGLE_API_KEY found - full functionality available!")
    
    # Create sample documents
    documents = create_sample_documents()
    print(f"\n📚 Created {len(documents)} sample documents for indexing")
    
    try:
        # Initialize IndexingManager
        manager, config = demonstrate_indexing_manager_basics()
        
        # Demonstrate each indexing strategy
        demonstrate_basic_indexing(manager, documents)
        demonstrate_multi_representation_indexing(manager, documents)
        demonstrate_colbert_indexing(manager, documents)
        demonstrate_raptor_indexing(manager, documents)
        
        # Strategy comparison
        demonstrate_strategy_comparison(manager, documents)
        
        # Advanced features
        demonstrate_advanced_features(manager)
        
        # Error handling
        demonstrate_error_handling(manager)
        
        print_separator("Demo Complete", "=")
        print("✅ All indexing strategy demonstrations completed!")
        
        print("\n💡 Key Takeaways:")
        print("  • 4 advanced indexing strategies available")
        print("  • Unified IndexingManager for strategy coordination")
        print("  • Each strategy optimized for different use cases")
        print("  • Comprehensive error handling and monitoring")
        print("  • Easy integration with existing RAG systems")
        
        print("\n🎯 Strategy Selection Guide:")
        print("  • Basic: Simple chunking for general use")
        print("  • Multi-Representation: Summaries for retrieval, full docs for generation")
        print("  • ColBERT: Token-level precision for fine-grained search")
        print("  • RAPTOR: Hierarchical organization for multi-level retrieval")
        
        print("\n🚀 Next Steps:")
        print("  1. Choose appropriate strategy for your use case")
        print("  2. Configure strategy parameters for optimal performance")
        print("  3. Integrate with your document processing pipeline")
        print("  4. Monitor performance and adjust as needed")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        print("This may be due to missing dependencies or API keys.")
        print("Check the requirements and try again.")


if __name__ == "__main__":
    main()