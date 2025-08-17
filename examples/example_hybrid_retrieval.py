"""
Example demonstrating hybrid retrieval capabilities
"""

import logging
import sys
from pathlib import Path
from typing import List

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_engine.core.models import Document
from rag_engine.core.config import PipelineConfig
from rag_engine.retrieval.retrieval_engine import RetrievalEngine
from rag_engine.retrieval.keyword_retriever import KeywordRetriever


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_documents() -> List[Document]:
    """Create sample documents for demonstration"""
    return [
        Document(
            content="Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models that enable computers to improve their performance on a specific task through experience.",
            metadata={"topic": "ML", "difficulty": "beginner", "type": "definition"},
            doc_id="ml_definition"
        ),
        Document(
            content="Deep learning neural networks use multiple layers of interconnected nodes to process and learn from complex data patterns. These networks can automatically discover representations from raw data.",
            metadata={"topic": "DL", "difficulty": "advanced", "type": "technical"},
            doc_id="dl_networks"
        ),
        Document(
            content="Natural language processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret, and manipulate human language. It combines computational linguistics with machine learning.",
            metadata={"topic": "NLP", "difficulty": "intermediate", "type": "overview"},
            doc_id="nlp_overview"
        ),
        Document(
            content="Computer vision algorithms enable machines to interpret and understand visual information from the world. These systems can identify objects, recognize faces, and analyze scenes in images and videos.",
            metadata={"topic": "CV", "difficulty": "intermediate", "type": "application"},
            doc_id="cv_algorithms"
        ),
        Document(
            content="Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize cumulative reward. It's inspired by behavioral psychology.",
            metadata={"topic": "RL", "difficulty": "advanced", "type": "methodology"},
            doc_id="rl_methodology"
        ),
        Document(
            content="Data preprocessing is a crucial step in machine learning that involves cleaning, transforming, and organizing raw data before feeding it to algorithms. This includes handling missing values and feature scaling.",
            metadata={"topic": "preprocessing", "difficulty": "beginner", "type": "process"},
            doc_id="data_preprocessing"
        ),
        Document(
            content="Supervised learning algorithms learn from labeled training data to make predictions on new, unseen data. Common examples include linear regression, decision trees, and support vector machines.",
            metadata={"topic": "supervised", "difficulty": "intermediate", "type": "category"},
            doc_id="supervised_learning"
        ),
        Document(
            content="Unsupervised learning finds hidden patterns in data without labeled examples. Clustering algorithms like K-means and dimensionality reduction techniques like PCA are common unsupervised methods.",
            metadata={"topic": "unsupervised", "difficulty": "intermediate", "type": "category"},
            doc_id="unsupervised_learning"
        )
    ]


def demonstrate_keyword_retrieval():
    """Demonstrate keyword-based retrieval using BM25"""
    print("\n" + "="*60)
    print("KEYWORD RETRIEVAL DEMONSTRATION")
    print("="*60)
    
    # Initialize keyword retriever
    config = PipelineConfig()
    keyword_retriever = KeywordRetriever(config)
    
    # Add documents
    documents = create_sample_documents()
    keyword_retriever.add_documents(documents)
    
    # Display index statistics
    stats = keyword_retriever.get_index_statistics()
    print(f"\nIndex Statistics:")
    print(f"- Total documents: {stats['total_documents']}")
    print(f"- Vocabulary size: {stats['vocabulary_size']}")
    print(f"- Average document length: {stats['average_document_length']:.1f} tokens")
    
    # Test different queries
    queries = [
        "machine learning algorithms",
        "neural networks deep learning",
        "natural language processing",
        "computer vision images",
        "supervised learning labeled data"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = keyword_retriever.retrieve_with_scores(query, k=3)
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"  {i}. Score: {score:.4f}")
            print(f"     ID: {doc.doc_id}")
            print(f"     Content: {doc.content[:100]}...")
            print()


def demonstrate_hybrid_retrieval():
    """Demonstrate hybrid retrieval combining vector and keyword search"""
    print("\n" + "="*60)
    print("HYBRID RETRIEVAL DEMONSTRATION")
    print("="*60)
    
    # Initialize retrieval engine (mocking vector retriever for demo)
    config = PipelineConfig()
    
    # Create engine with mocked vector retriever
    from unittest.mock import Mock
    
    # Mock vector retriever to simulate vector search results
    mock_vector_retriever = Mock()
    mock_vector_retriever.add_documents.return_value = True
    
    # Create retrieval engine
    engine = RetrievalEngine(config)
    
    # Replace vector retriever with mock for demonstration
    engine.retrievers["vector"] = mock_vector_retriever
    
    # Add documents to all retrievers
    documents = create_sample_documents()
    engine.add_documents(documents)
    
    # Mock vector search results for different queries
    def mock_vector_search(query, k):
        # Simulate vector search returning semantically similar documents
        if "machine learning" in query.lower():
            return [(documents[0], 0.92), (documents[6], 0.85)]  # ML definition, supervised learning
        elif "neural networks" in query.lower():
            return [(documents[1], 0.95), (documents[0], 0.78)]  # Deep learning, ML definition
        elif "language" in query.lower():
            return [(documents[2], 0.90), (documents[0], 0.72)]  # NLP, ML definition
        else:
            return [(documents[0], 0.80), (documents[1], 0.75)]  # Default results
    
    mock_vector_retriever.retrieve_with_scores.side_effect = mock_vector_search
    
    # Test hybrid retrieval with different queries
    queries = [
        "machine learning algorithms for data analysis",
        "deep neural networks and artificial intelligence",
        "natural language processing techniques",
        "computer vision and image recognition"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 50)
        
        # Hybrid retrieval
        hybrid_results = engine.hybrid_retrieve(query, k=4, vector_weight=0.6, keyword_weight=0.4)
        
        print("Hybrid Retrieval Results:")
        for i, doc in enumerate(hybrid_results, 1):
            print(f"  {i}. ID: {doc.doc_id}")
            print(f"     Topic: {doc.metadata.get('topic', 'N/A')}")
            print(f"     Content: {doc.content[:80]}...")
            print()


def demonstrate_long_context_retrieval():
    """Demonstrate long-context retrieval for large context windows"""
    print("\n" + "="*60)
    print("LONG-CONTEXT RETRIEVAL DEMONSTRATION")
    print("="*60)
    
    # Create engine with mocked vector retriever
    config = PipelineConfig()
    from unittest.mock import Mock
    
    mock_vector_retriever = Mock()
    mock_vector_retriever.add_documents.return_value = True
    
    engine = RetrievalEngine(config)
    engine.retrievers["vector"] = mock_vector_retriever
    
    # Add documents
    documents = create_sample_documents()
    engine.add_documents(documents)
    
    # Mock vector search to return all documents
    mock_vector_retriever.retrieve.return_value = documents
    
    # Test long-context retrieval with different context window sizes
    query = "comprehensive overview of machine learning and artificial intelligence"
    
    context_sizes = [500, 1000, 2000, 5000]
    
    for context_size in context_sizes:
        print(f"\nContext Window Size: {context_size} tokens")
        print("-" * 40)
        
        results, metadata = engine.long_context_retrieve(
            query, 
            k=len(documents), 
            context_window_size=context_size
        )
        
        print(f"Documents processed: {metadata['documents_processed']}")
        print(f"Documents returned: {metadata['documents_returned']}")
        print(f"Estimated tokens: {metadata['estimated_tokens']}")
        print(f"Retrieval time: {metadata['retrieval_time']:.3f}s")
        
        print("Included documents:")
        for doc in results:
            truncated = doc.metadata.get('truncated', False)
            status = " (truncated)" if truncated else ""
            print(f"  - {doc.doc_id}: {doc.metadata.get('topic', 'N/A')}{status}")


def demonstrate_adaptive_retrieval():
    """Demonstrate adaptive retrieval strategy selection"""
    print("\n" + "="*60)
    print("ADAPTIVE RETRIEVAL DEMONSTRATION")
    print("="*60)
    
    # Create engine with mocked components
    config = PipelineConfig()
    from unittest.mock import Mock
    
    mock_vector_retriever = Mock()
    mock_vector_retriever.add_documents.return_value = True
    
    engine = RetrievalEngine(config)
    engine.retrievers["vector"] = mock_vector_retriever
    
    # Add documents
    documents = create_sample_documents()
    engine.add_documents(documents)
    
    # Mock different retrieval methods
    engine.hybrid_retrieve = Mock(return_value=documents[:3])
    engine.long_context_retrieve = Mock(return_value=(documents[:5], {"strategy": "long_context"}))
    engine.retrieve_with_rerank = Mock(return_value=documents[:2])
    engine.retrieve = Mock(return_value=documents[:4])
    
    # Test different query types that should trigger different strategies
    test_queries = [
        ("What is AI?", "Simple semantic query"),
        ("Find specific machine learning algorithms for classification", "Keyword-focused query"),
        ("Compare and analyze different approaches to deep learning", "Complex analytical query"),
        ("Provide a comprehensive detailed explanation of the entire machine learning ecosystem including supervised unsupervised reinforcement learning methodologies", "Very long query")
    ]
    
    for query, description in test_queries:
        print(f"\nQuery Type: {description}")
        print(f"Query: '{query}'")
        print("-" * 50)
        
        results, metadata = engine.adaptive_retrieve(query, k=3, strategy="auto")
        
        print(f"Selected Strategy: {metadata['selected_strategy']}")
        print(f"Documents Returned: {metadata['documents_returned']}")
        print(f"Query Length: {metadata['query_length']} characters")
        
        if metadata['selected_strategy'] == 'hybrid':
            print("→ Used hybrid retrieval (vector + keyword)")
        elif metadata['selected_strategy'] == 'long_context':
            print("→ Used long-context retrieval")
        elif metadata['selected_strategy'] == 'rerank':
            print("→ Used retrieval with re-ranking")
        else:
            print("→ Used standard vector retrieval")


def main():
    """Run all hybrid retrieval demonstrations"""
    print("HYBRID RETRIEVAL SYSTEM DEMONSTRATION")
    print("=====================================")
    print("This example demonstrates the hybrid retrieval capabilities")
    print("including keyword search, vector search combination, and")
    print("adaptive strategy selection for different query types.")
    
    try:
        # Run demonstrations
        demonstrate_keyword_retrieval()
        demonstrate_hybrid_retrieval()
        demonstrate_long_context_retrieval()
        demonstrate_adaptive_retrieval()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETE")
        print("="*60)
        print("The hybrid retrieval system successfully demonstrated:")
        print("✓ BM25-based keyword retrieval")
        print("✓ Hybrid vector + keyword search")
        print("✓ Long-context document filtering")
        print("✓ Adaptive strategy selection")
        print("✓ Comprehensive error handling")
        
    except Exception as e:
        logger.error(f"Error in demonstration: {str(e)}")
        raise


if __name__ == "__main__":
    main()