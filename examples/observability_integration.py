"""
Example of integrating observability with RAG engine operations
"""

import os
from datetime import datetime
from typing import List, Dict, Any

# Set up environment for example
os.environ.setdefault('RAG_OBSERVABILITY_ENABLED', 'true')
os.environ.setdefault('RAG_OBSERVABILITY_PROVIDER', 'langfuse')
os.environ.setdefault('LANGFUSE_SECRET_KEY', 'sk-lf-example')
os.environ.setdefault('LANGFUSE_PUBLIC_KEY', 'pk-lf-example')

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag_engine.core.config import ConfigurationManager
from rag_engine.observability import (
    ObservabilityManager, 
    create_observability_config,
    TraceLevel
)
from rag_engine.observability.decorators import (
    trace_llm_call, 
    trace_retrieval, 
    trace_embedding
)


class MockDocument:
    """Mock document for example"""
    def __init__(self, content: str, metadata: Dict[str, Any] = None):
        self.content = content
        self.metadata = metadata or {}


class ExampleRAGSystem:
    """Example RAG system with observability integration"""
    
    def __init__(self, observability_manager: ObservabilityManager):
        self.obs_manager = observability_manager
    
    def generate_embeddings(self, texts: List[str], trace_context=None) -> List[List[float]]:
        """Generate embeddings for texts"""
        # Mock embedding generation
        embeddings = [[0.1, 0.2, 0.3] for _ in texts]
        return embeddings
    
    def retrieve_documents(self, query: str, k: int = 5, trace_context=None) -> List[MockDocument]:
        """Retrieve relevant documents"""
        # Mock document retrieval
        docs = [
            MockDocument(f"Document {i} content related to: {query}", 
                        {"score": 0.9 - i*0.1, "source": f"doc_{i}.txt"})
            for i in range(k)
        ]
        return docs
    
    def generate_answer(self, query: str, context_docs: List[MockDocument], trace_context=None) -> str:
        """Generate answer using LLM"""
        # Mock answer generation
        context = "\n".join([doc.content for doc in context_docs])
        answer = f"Based on the context, here's the answer to '{query}': [Generated answer]"
        return answer
    
    def process_query(self, query: str, user_id: str = None, session_id: str = None) -> Dict[str, Any]:
        """Process a complete RAG query with full observability"""
        
        # Create main trace for the entire query
        with self.obs_manager.trace(
            name="rag_query_processing",
            user_id=user_id,
            session_id=session_id,
            metadata={
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "system": "example_rag"
            }
        ) as context:
            
            try:
                # Log query start
                self.obs_manager.log_event(
                    context=context,
                    event_name="query_started",
                    data={"query": query, "user_id": user_id}
                )
                
                # Step 1: Generate query embedding
                with self.obs_manager.span(
                    context=context,
                    name="query_embedding",
                    span_type="embedding",
                    input_data={"query": query}
                ) as embedding_span:
                    
                    query_embedding = self.generate_embeddings([query], trace_context=context)[0]
                    
                    self.obs_manager.log_embedding(
                        context=context,
                        name="query_embedding_generation",
                        model="text-embedding-ada-002",
                        provider="openai",
                        input_count=1,
                        embedding_dimension=len(query_embedding),
                        input_data={"query": query},
                        output_data={"embedding_dim": len(query_embedding)}
                    )
                
                # Step 2: Retrieve relevant documents
                with self.obs_manager.span(
                    context=context,
                    name="document_retrieval",
                    span_type="retrieval",
                    input_data={"query": query, "k": 5}
                ) as retrieval_span:
                    
                    documents = self.retrieve_documents(query, k=5, trace_context=context)
                    
                    self.obs_manager.log_retrieval(
                        context=context,
                        name="vector_search",
                        query=query,
                        retrieved_count=len(documents),
                        vector_store="chroma",
                        output_data={
                            "documents": [{"content": doc.content[:100], "metadata": doc.metadata} 
                                        for doc in documents]
                        }
                    )
                
                # Step 3: Generate answer
                with self.obs_manager.span(
                    context=context,
                    name="answer_generation",
                    span_type="llm",
                    input_data={"query": query, "context_docs_count": len(documents)}
                ) as generation_span:
                    
                    answer = self.generate_answer(query, documents, trace_context=context)
                    
                    self.obs_manager.log_llm_call(
                        context=context,
                        name="answer_generation",
                        model="gemini-2.0-flash-lite",
                        provider="google",
                        input_data={
                            "query": query,
                            "context": f"{len(documents)} documents"
                        },
                        output_data={"answer": answer},
                        prompt_tokens=150,  # Mock token counts
                        completion_tokens=75,
                        total_tokens=225,
                        cost=0.001  # Mock cost
                    )
                
                # Log successful completion
                self.obs_manager.log_event(
                    context=context,
                    event_name="query_completed",
                    data={
                        "success": True,
                        "retrieved_docs": len(documents),
                        "answer_length": len(answer)
                    }
                )
                
                return {
                    "answer": answer,
                    "documents": documents,
                    "metadata": {
                        "retrieved_count": len(documents),
                        "trace_id": context.trace_id
                    }
                }
                
            except Exception as e:
                # Log error
                self.obs_manager.log_event(
                    context=context,
                    event_name="query_error",
                    data={"error": str(e), "error_type": type(e).__name__},
                    level=TraceLevel.ERROR
                )
                raise


def main():
    """Main example function"""
    print("🔍 RAG Engine Observability Integration Example")
    print("=" * 50)
    
    # Load configuration
    config_manager = ConfigurationManager()
    config = config_manager.load_config()
    
    # Create observability manager
    obs_config = create_observability_config(config)
    obs_manager = ObservabilityManager(obs_config)
    
    # Initialize observability
    if obs_manager.initialize():
        print(f"✅ Observability initialized with {obs_manager.provider_name} provider")
    else:
        print("⚠️  Observability initialization failed, using no-op provider")
    
    # Create RAG system
    rag_system = ExampleRAGSystem(obs_manager)
    
    # Example queries
    queries = [
        "What is retrieval-augmented generation?",
        "How do vector databases work?",
        "Explain the benefits of RAG systems"
    ]
    
    print(f"\n🚀 Processing {len(queries)} example queries...")
    
    for i, query in enumerate(queries, 1):
        print(f"\n📝 Query {i}: {query}")
        
        try:
            result = rag_system.process_query(
                query=query,
                user_id=f"user_{i}",
                session_id="example_session"
            )
            
            print(f"✅ Answer: {result['answer'][:100]}...")
            print(f"📊 Retrieved {result['metadata']['retrieved_count']} documents")
            print(f"🔗 Trace ID: {result['metadata']['trace_id']}")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
    
    # Flush and cleanup
    print(f"\n🔄 Flushing observability data...")
    obs_manager.flush()
    
    print(f"\n📈 Observability Info:")
    info = obs_manager.get_provider_info()
    for key, value in info.items():
        if key != 'config':
            print(f"  {key}: {value}")
    
    # Shutdown
    obs_manager.shutdown()
    print(f"\n✅ Example completed successfully!")


if __name__ == "__main__":
    main()