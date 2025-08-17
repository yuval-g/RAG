#!/usr/bin/env python3
"""
Usage Example: Advanced Query Processing System

This example shows practical usage patterns for the query processing system
in different scenarios. It demonstrates real-world integration patterns.

Prerequisites:
- Set GOOGLE_API_KEY environment variable or create .env file
- Install dependencies: uv sync
"""

import os
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_engine.query.processor import QueryProcessor, QueryStrategy
from src.rag_engine.core.models import Document

# Load environment variables from .env file
load_dotenv()


class SimpleRAGSystem:
    """
    Example RAG system using the advanced query processing
    """
    
    def __init__(self):
        """Initialize the RAG system with query processor"""
        # Initialize query processor
        self.query_processor = QueryProcessor(
            llm_model="gemini-2.0-flash-lite",
            temperature=0.0,
            default_strategy="multi_query"
        )
        
        # Sample knowledge base
        self.documents = self._create_knowledge_base()
        
        # Configure strategies for our use case
        self._configure_strategies()
    
    def _create_knowledge_base(self) -> List[Document]:
        """Create a sample knowledge base"""
        return [
            Document(
                content="Task decomposition is a fundamental approach in AI where complex problems are systematically broken down into smaller, more manageable sub-tasks. This technique enables agents to handle sophisticated challenges through step-by-step problem solving.",
                metadata={"source": "ai_fundamentals.pdf", "topic": "task_decomposition", "difficulty": "intermediate"}
            ),
            Document(
                content="Large Language Model (LLM) agents utilize advanced planning mechanisms to organize their approach to complex problems. These systems employ hierarchical planning, goal decomposition, and multi-step reasoning to achieve their objectives.",
                metadata={"source": "llm_planning.pdf", "topic": "planning", "difficulty": "advanced"}
            ),
            Document(
                content="Memory systems in AI agents serve as crucial components for maintaining context and learning from past interactions. They typically include both short-term working memory for immediate tasks and long-term episodic memory for experience retention.",
                metadata={"source": "agent_memory.pdf", "topic": "memory", "difficulty": "intermediate"}
            ),
            Document(
                content="Tool use capabilities enable AI agents to extend their functionality beyond inherent language processing. This includes API interactions, database queries, web searches, and integration with external systems and services.",
                metadata={"source": "tool_usage.pdf", "topic": "tools", "difficulty": "beginner"}
            ),
            Document(
                content="Retrieval-Augmented Generation (RAG) combines the power of large language models with external knowledge retrieval systems. This approach helps ground AI responses in factual, up-to-date information from curated knowledge bases.",
                metadata={"source": "rag_systems.pdf", "topic": "rag", "difficulty": "advanced"}
            )
        ]
    
    def _configure_strategies(self):
        """Configure query processing strategies for our use case"""
        # Configure multi-query for broader coverage
        self.query_processor.configure_strategy(
            "multi_query", 
            num_queries=6,
            temperature=0.1
        )
        
        # Configure RAG-Fusion for better ranking
        self.query_processor.configure_strategy(
            "rag_fusion",
            num_queries=4,
            rrf_k=50
        )
        
        # Configure HyDE for technical documentation style
        self.query_processor.configure_strategy(
            "hyde",
            document_style="technical_documentation"
        )
    
    def retrieve_documents(self, query: str, top_k: int = 3) -> List[Document]:
        """Simple retriever based on keyword matching and metadata"""
        query_words = set(query.lower().split())
        scored_docs = []
        
        for doc in self.documents:
            score = 0
            
            # Content matching
            doc_words = set(doc.content.lower().split())
            content_overlap = len(query_words.intersection(doc_words))
            score += content_overlap * 2
            
            # Topic matching
            if any(word in doc.metadata.get("topic", "") for word in query_words):
                score += 5
            
            # Difficulty preference (prefer intermediate/advanced for complex queries)
            if len(query.split()) > 8:  # Complex query
                if doc.metadata.get("difficulty") in ["intermediate", "advanced"]:
                    score += 1
            
            if score > 0:
                scored_docs.append((doc, score))
        
        # Sort by score and return top_k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_k]]
    
    def answer_question(self, question: str, strategy: str = "multi_query") -> Dict[str, Any]:
        """Answer a question using the specified strategy"""
        
        print(f"🤔 Question: {question}")
        print(f"🔄 Using strategy: {strategy.upper()}")
        
        try:
            if strategy in ["rag_fusion", "multi_query", "hyde"]:
                # Use retriever integration for these strategies
                results = self.query_processor.process_with_retriever(
                    question, strategy, self.retrieve_documents, top_k=3
                )
                
                if strategy == "rag_fusion":
                    # RAG-Fusion returns (document, score) tuples
                    documents = [doc for doc, score in results]
                    scores = [score for doc, score in results]
                    print(f"📊 Retrieved {len(documents)} documents with RRF scores: {[f'{s:.3f}' for s in scores]}")
                else:
                    # Other strategies return document lists
                    documents = results
                    print(f"📚 Retrieved {len(documents)} documents")
                
                # Generate answer from retrieved documents
                context = "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.content}" 
                                     for doc in documents])
                
                return {
                    "question": question,
                    "strategy": strategy,
                    "documents": documents,
                    "context": context,
                    "answer": f"Based on the retrieved documents: {context[:200]}..."
                }
            
            elif strategy == "decomposition":
                # Decomposition returns a synthesized answer
                answer = self.query_processor.process_with_retriever(
                    question, strategy, self.retrieve_documents
                )
                
                return {
                    "question": question,
                    "strategy": strategy,
                    "answer": answer,
                    "documents": []
                }
            
            elif strategy == "step_back":
                # Step-back processing
                normal_docs, step_back_docs, step_back_q = self.query_processor.process_with_retriever(
                    question, strategy, self.retrieve_documents, top_k=3
                )
                
                print(f"🔙 Step-back question: {step_back_q}")
                print(f"📚 Normal context: {len(normal_docs)} docs, Step-back context: {len(step_back_docs)} docs")
                
                return {
                    "question": question,
                    "strategy": strategy,
                    "step_back_question": step_back_q,
                    "normal_documents": normal_docs,
                    "step_back_documents": step_back_docs,
                    "answer": "Combined answer using both normal and step-back context"
                }
            
            else:
                # Basic processing
                result = self.query_processor.process(question, strategy)
                documents = self.retrieve_documents(question)
                
                return {
                    "question": question,
                    "strategy": strategy,
                    "processed_query": result,
                    "documents": documents,
                    "answer": f"Basic answer using {len(documents)} documents"
                }
                
        except Exception as e:
            return {
                "question": question,
                "strategy": strategy,
                "error": str(e),
                "answer": "Error occurred during processing"
            }


def demonstrate_different_strategies():
    """Demonstrate different strategies with the same question"""
    
    print("🎯 Strategy Comparison Demo")
    print("=" * 50)
    
    # Initialize RAG system
    try:
        rag_system = SimpleRAGSystem()
        print("✅ RAG System initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {str(e)}")
        print("💡 Make sure GOOGLE_API_KEY is set for full functionality")
        return
    
    # Test question
    question = "How do LLM agents use task decomposition and planning for complex problem solving?"
    
    # Test different strategies
    strategies = ["basic", "multi_query", "rag_fusion", "step_back", "hyde"]
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Testing {strategy.upper()} Strategy")
        print('='*60)
        
        result = rag_system.answer_question(question, strategy)
        
        print(f"📝 Result:")
        if "error" in result:
            print(f"   ❌ Error: {result['error']}")
        else:
            print(f"   ✅ Strategy: {result['strategy']}")
            if "documents" in result and result["documents"]:
                print(f"   📚 Documents used: {len(result['documents'])}")
                for i, doc in enumerate(result["documents"][:2], 1):
                    print(f"      {i}. {doc.metadata.get('topic', 'Unknown topic')} - {doc.content[:80]}...")
            
            if "step_back_question" in result:
                print(f"   🔙 Step-back question: {result['step_back_question']}")
            
            print(f"   💬 Answer: {result['answer'][:150]}...")


def demonstrate_configuration():
    """Demonstrate strategy configuration"""
    
    print("\n🔧 Configuration Demo")
    print("=" * 30)
    
    try:
        processor = QueryProcessor()
        
        # Show available strategies
        print("📋 Available strategies:")
        for strategy in processor.get_available_strategies():
            info = processor.get_strategy_info(strategy)
            print(f"   • {strategy}: {info['description'][:60]}...")
        
        # Configure strategies
        print("\n⚙️  Configuring strategies:")
        processor.configure_strategy("multi_query", num_queries=8, temperature=0.2)
        processor.configure_strategy("hyde", document_style="encyclopedia")
        print("   ✅ Configured multi_query and hyde strategies")
        
        # Show configurations
        print("\n📊 Current configurations:")
        configs = processor.get_all_configs()
        for strategy, config in configs.items():
            if "error" not in config:
                print(f"   {strategy}: {config.get('strategy', 'basic')}")
        
    except Exception as e:
        print(f"❌ Configuration demo failed: {str(e)}")


def main():
    """Main demonstration"""
    
    print("🚀 Advanced Query Processing - Usage Examples")
    print("This demo shows practical usage patterns")
    
    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("\n⚠️  Warning: GOOGLE_API_KEY not set!")
        print("Some features will not work without proper authentication.")
        print("To set up your API key:")
        print("  1. Copy .env.example to .env: cp .env.example .env")
        print("  2. Edit .env and add your Google API key")
        print("  3. Or export GOOGLE_API_KEY='your-api-key'")
        print("\nContinuing with structure demonstration...")
    else:
        print("✅ GOOGLE_API_KEY found - full functionality available!")
    
    # Demonstrate different strategies
    demonstrate_different_strategies()
    
    # Demonstrate configuration
    demonstrate_configuration()
    
    print("\n" + "="*60)
    print("✅ Usage demonstration completed!")
    
    print("\n💡 Key Usage Patterns:")
    print("  1. Initialize QueryProcessor with your preferred settings")
    print("  2. Configure strategies for your specific use case")
    print("  3. Choose strategy based on query complexity and requirements")
    print("  4. Integrate with your existing retriever and knowledge base")
    print("  5. Handle errors gracefully in production")
    
    print("\n🎯 Strategy Selection Guide:")
    print("  • multi_query: Good for broad topic coverage")
    print("  • rag_fusion: Best for precise ranking and relevance")
    print("  • decomposition: Ideal for complex, multi-part questions")
    print("  • step_back: Useful when queries are too specific")
    print("  • hyde: Excellent for semantic similarity matching")
    print("  • basic: Simple pass-through for basic use cases")


if __name__ == "__main__":
    main()