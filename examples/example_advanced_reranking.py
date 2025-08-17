"""
Example demonstrating advanced document re-ranking functionality.
This example shows how to use the RetrievalEngine with re-ranking capabilities
from workplan/04AdvancedRetrieval-Generation.md
"""

import os
import sys
from typing import List

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag_engine.core.models import Document
from rag_engine.core.config import PipelineConfig
from rag_engine.retrieval.retrieval_engine import RetrievalEngine
from rag_engine.retrieval.vector_retriever import VectorRetriever
from rag_engine.retrieval.reranker import ReRanker


def create_sample_documents() -> List[Document]:
    """Create sample documents for demonstration"""
    return [
        Document(
            content="Python is a high-level programming language known for its simplicity and readability. It's widely used in web development, data science, and machine learning applications.",
            metadata={"source": "python_guide", "topic": "programming", "difficulty": "beginner"},
            doc_id="python_doc"
        ),
        Document(
            content="JavaScript is a dynamic programming language primarily used for web development. It enables interactive web pages and is an essential part of modern web applications.",
            metadata={"source": "js_guide", "topic": "programming", "difficulty": "intermediate"},
            doc_id="js_doc"
        ),
        Document(
            content="Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed for every task.",
            metadata={"source": "ml_intro", "topic": "ai", "difficulty": "advanced"},
            doc_id="ml_doc"
        ),
        Document(
            content="Data structures are ways of organizing and storing data in a computer so that it can be accessed and modified efficiently. Common examples include arrays, linked lists, and trees.",
            metadata={"source": "ds_guide", "topic": "programming", "difficulty": "intermediate"},
            doc_id="ds_doc"
        ),
        Document(
            content="The weather forecast shows sunny skies with temperatures reaching 75 degrees Fahrenheit today. Perfect weather for outdoor activities.",
            metadata={"source": "weather_report", "topic": "weather", "difficulty": "easy"},
            doc_id="weather_doc"
        ),
        Document(
            content="Natural language processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret, and manipulate human language.",
            metadata={"source": "nlp_intro", "topic": "ai", "difficulty": "advanced"},
            doc_id="nlp_doc"
        )
    ]


def demonstrate_basic_retrieval():
    """Demonstrate basic retrieval without re-ranking"""
    print("=== Basic Retrieval (No Re-ranking) ===")
    
    # Create configuration
    config = PipelineConfig(
        retrieval_k=4,
        use_reranking=False
    )
    
    # Create retrieval engine
    engine = RetrievalEngine(config)
    
    # Create a mock retriever with our sample documents
    class MockRetriever:
        def __init__(self, documents):
            self.documents = documents
        
        def retrieve(self, query: str, k: int = 5) -> List[Document]:
            # Simple mock retrieval - return first k documents
            return self.documents[:k]
        
        def retrieve_with_scores(self, query: str, k: int = 5) -> List[tuple]:
            # Return documents with mock scores
            docs = self.documents[:k]
            return [(doc, 0.8 - i * 0.1) for i, doc in enumerate(docs)]
    
    # Add mock retriever
    mock_retriever = MockRetriever(create_sample_documents())
    engine.add_retriever("mock", mock_retriever)
    engine.set_default_retriever("mock")
    
    # Perform retrieval
    query = "Python programming tutorial for beginners"
    results = engine.retrieve(query, k=3)
    
    print(f"Query: {query}")
    print(f"Retrieved {len(results)} documents:")
    for i, doc in enumerate(results, 1):
        print(f"{i}. [{doc.doc_id}] {doc.content[:100]}...")
        print(f"   Topic: {doc.metadata.get('topic', 'N/A')}")
    print()


def demonstrate_llm_reranking():
    """Demonstrate LLM-based re-ranking"""
    print("=== LLM-Based Re-ranking ===")
    
    # Create configuration with re-ranking enabled
    config = PipelineConfig(
        retrieval_k=5,
        use_reranking=True,
        reranker_top_k=3,
        llm_model="gemini-2.0-flash-lite"
    )
    
    # Create sample documents
    documents = create_sample_documents()
    
    # Create LLM re-ranker directly for demonstration
    try:
        reranker = ReRanker(strategy="llm", config=config)
        
        query = "Python programming tutorial for beginners"
        print(f"Query: {query}")
        print("Original document order:")
        for i, doc in enumerate(documents[:4], 1):
            print(f"{i}. [{doc.doc_id}] {doc.content[:80]}...")
        
        print("\nRe-ranking documents...")
        # Note: This would normally call the LLM, but for demo purposes we'll show the structure
        print("(In a real scenario, this would use Google Gemini to score relevance)")
        
        # Simulate re-ranked results
        reranked_docs = [
            documents[0],  # Python doc - most relevant
            documents[3],  # Data structures - programming related
            documents[1],  # JavaScript - also programming
        ]
        
        print("\nRe-ranked results:")
        for i, doc in enumerate(reranked_docs, 1):
            print(f"{i}. [{doc.doc_id}] {doc.content[:80]}...")
            print(f"   Relevance: High (programming-related)")
        
    except Exception as e:
        print(f"Note: LLM re-ranking requires API keys. Error: {e}")
        print("This demonstrates the structure - in production, configure your Google API key.")
    
    print()


def demonstrate_retrieval_engine_workflow():
    """Demonstrate the complete RetrievalEngine workflow"""
    print("=== Complete RetrievalEngine Workflow ===")
    
    # Create configuration
    config = PipelineConfig(
        retrieval_k=6,
        use_reranking=False,  # Start without re-ranking
        reranker_top_k=3
    )
    
    # Create retrieval engine
    engine = RetrievalEngine(config)
    
    # Create mock retriever
    class MockRetriever:
        def __init__(self, documents):
            self.documents = documents
        
        def retrieve(self, query: str, k: int = 5) -> List[Document]:
            return self.documents[:k]
        
        def retrieve_with_scores(self, query: str, k: int = 5) -> List[tuple]:
            docs = self.documents[:k]
            return [(doc, 0.9 - i * 0.1) for i, doc in enumerate(docs)]
    
    # Add multiple retrievers
    all_docs = create_sample_documents()
    programming_docs = [doc for doc in all_docs if doc.metadata.get('topic') == 'programming']
    ai_docs = [doc for doc in all_docs if doc.metadata.get('topic') == 'ai']
    
    engine.add_retriever("all_docs", MockRetriever(all_docs))
    engine.add_retriever("programming", MockRetriever(programming_docs))
    engine.add_retriever("ai", MockRetriever(ai_docs))
    
    # Test different retrievers
    query = "machine learning algorithms"
    
    print(f"Query: {query}")
    print("\n1. Retrieving from all documents:")
    results_all = engine.retrieve(query, k=3, retriever_name="all_docs")
    for i, doc in enumerate(results_all, 1):
        print(f"   {i}. [{doc.doc_id}] Topic: {doc.metadata.get('topic')}")
    
    print("\n2. Retrieving from AI-specific documents:")
    results_ai = engine.retrieve(query, k=2, retriever_name="ai")
    for i, doc in enumerate(results_ai, 1):
        print(f"   {i}. [{doc.doc_id}] Topic: {doc.metadata.get('topic')}")
    
    print("\n3. Retrieving with scores:")
    results_with_scores = engine.retrieve_with_scores(query, k=3, retriever_name="all_docs")
    for i, (doc, score) in enumerate(results_with_scores, 1):
        print(f"   {i}. [{doc.doc_id}] Score: {score:.2f} Topic: {doc.metadata.get('topic')}")
    
    # Enable re-ranking
    print("\n4. Enabling LLM re-ranking:")
    try:
        engine.enable_reranking("llm")
        print("   Re-ranking enabled successfully")
        print(f"   Re-ranking strategy: {engine.reranker.get_strategy() if engine.reranker else 'None'}")
        print("   (Would use Google Gemini for scoring in production)")
    except Exception as e:
        print(f"   Note: Re-ranking requires API configuration: {e}")
    
    # Show engine statistics
    print("\n5. Engine Statistics:")
    stats = engine.get_retrieval_stats()
    print(f"   Available retrievers: {stats['available_retrievers']}")
    print(f"   Default retriever: {stats['default_retriever']}")
    print(f"   Re-ranking enabled: {stats['reranking_enabled']}")
    print(f"   Re-ranking strategy: {stats.get('reranking_strategy', 'None')}")
    
    print()


def demonstrate_configuration_management():
    """Demonstrate configuration management and updates"""
    print("=== Configuration Management ===")
    
    # Initial configuration
    config = PipelineConfig(
        retrieval_k=3,
        use_reranking=False,
        llm_model="gemini-2.0-flash-lite"
    )
    
    engine = RetrievalEngine(config)
    print(f"Initial config - Retrieval K: {engine.config.retrieval_k}, Re-ranking: {engine.config.use_reranking}")
    
    # Update configuration
    new_config = PipelineConfig(
        retrieval_k=5,
        use_reranking=True,
        reranker_top_k=3,
        llm_model="gemini-2.0-flash-lite"
    )
    
    print("Updating configuration...")
    engine.update_config(new_config)
    print(f"Updated config - Retrieval K: {engine.config.retrieval_k}, Re-ranking: {engine.config.use_reranking}")
    
    # Show configuration propagation
    stats = engine.get_retrieval_stats()
    print("Configuration propagated to all components:")
    print(f"  - Retrieval K: {stats['config']['retrieval_k']}")
    print(f"  - Re-ranker Top K: {stats['config']['reranker_top_k']}")
    print(f"  - Use Re-ranking: {stats['config']['use_reranking']}")
    
    print()


def main():
    """Run all demonstrations"""
    print("Advanced Document Re-ranking Demonstration")
    print("=" * 50)
    print()
    
    # Run demonstrations
    demonstrate_basic_retrieval()
    demonstrate_llm_reranking()
    demonstrate_retrieval_engine_workflow()
    demonstrate_configuration_management()
    
    print("Demonstration complete!")
    print("\nKey Features Demonstrated:")
    print("✓ Basic document retrieval")
    print("✓ LLM-based re-ranking with Google Gemini")
    print("✓ Multiple retriever management")
    print("✓ Retrieval with relevance scores")
    print("✓ Dynamic re-ranking enable/disable")
    print("✓ Configuration management and updates")
    print("✓ Engine statistics and monitoring")
    print("\nFor production use:")
    print("- Configure Google API key for Gemini")
    print("- Set up proper vector stores (Chroma, etc.)")
    print("- Configure embedding providers")
    print("- Enable logging and monitoring")


if __name__ == "__main__":
    main()