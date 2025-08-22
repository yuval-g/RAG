"""
Main RAG engine orchestrator
"""

from typing import List, Dict, Any, Optional
import time
import logging
from .interfaces import BaseRAGEngine
from .models import Document, RAGResponse, EvaluationResult, TestCase
from .config import PipelineConfig, ConfigurationManager
from ..indexing.basic_indexer import BasicIndexer
from ..retrieval.vector_retriever import VectorRetriever
from ..generation.generation_engine import GenerationEngine


logger = logging.getLogger(__name__)


class RAGEngine(BaseRAGEngine):
    """Main RAG engine that orchestrates the entire pipeline"""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the RAG engine with configuration"""
        if config is None:
            config_manager = ConfigurationManager()
            config = config_manager.load_config()
        
        self.config = config
        self._setup_logging()
        
        # Initialize core components
        self._indexer = BasicIndexer(config)
        self._retriever = VectorRetriever(config=config)
        self._generator = GenerationEngine(config)
        
        # Components for later tasks
        self._query_processor = None
        self._router = None
        self._evaluator = None
        
        logger.info("RAG Engine initialized with basic components")
    
    def _setup_logging(self) -> None:
        """Set up logging configuration"""
        if self.config.enable_logging:
            logging.basicConfig(
                level=getattr(logging, self.config.log_level.upper()),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def query(self, question: str, **kwargs) -> RAGResponse:
        """Process a query through the RAG pipeline"""
        start_time = time.time()
        
        try:
            logger.info(f"Processing query: {question[:100]}...")
            
            # Check if retriever is ready (has vector store)
            if not self._retriever.is_ready():
                logger.warning("No documents indexed yet. Please add documents first.")
                return RAGResponse(
                    answer="No documents have been indexed yet. Please add documents to the system first.",
                    source_documents=[],
                    confidence_score=0.0,
                    processing_time=time.time() - start_time,
                    metadata={"query": question, "status": "no_documents"}
                )
            
            # Retrieve relevant documents
            k = kwargs.get('k', self.config.retrieval_k)
            retrieved_docs = self._retriever.retrieve(question, k=k)
            
            if not retrieved_docs:
                logger.warning("No relevant documents found for query")
                return RAGResponse(
                    answer="I couldn't find any relevant information to answer your question.",
                    source_documents=[],
                    confidence_score=0.0,
                    processing_time=time.time() - start_time,
                    metadata={"query": question, "status": "no_relevant_docs"}
                )
            
            # Generate response using retrieved context
            answer = self._generator.generate(question, retrieved_docs)
            
            # Calculate basic confidence score based on number of retrieved docs
            confidence_score = min(len(retrieved_docs) / self.config.retrieval_k, 1.0)
            
            metadata = {
                "query": question,
                "retrieved_count": len(retrieved_docs),
                "config": {
                    "llm_provider": self.config.llm_provider,
                    "embedding_provider": self.config.embedding_provider,
                    "vector_store": self.config.vector_store
                }
            }
            metadata.update(kwargs)

            response = RAGResponse(
                answer=answer,
                source_documents=retrieved_docs,
                confidence_score=confidence_score,
                processing_time=time.time() - start_time,
                metadata=metadata
            )
            
            logger.info(f"Query processed successfully in {response.processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return RAGResponse(
                answer=f"I apologize, but I encountered an error while processing your question: {str(e)}",
                source_documents=[],
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                metadata={"error": str(e), "query": question}
            )
    
    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the RAG system"""
        try:
            logger.info(f"Adding {len(documents)} documents to the system")
            
            if not documents:
                logger.warning("No documents provided")
                return False
            
            # Validate documents
            for i, doc in enumerate(documents):
                if not doc.content:
                    logger.warning(f"Document {i} has empty content")
                    return False
            
            # Index documents using the BasicIndexer
            success = self._indexer.index_documents(documents)
            
            if success:
                # Update retriever with the new vector store
                vectorstore = self._indexer.get_vectorstore()
                if vectorstore is not None:
                    self._retriever.set_vectorstore(vectorstore)
                    logger.info(f"Successfully indexed {len(documents)} documents")
                    return True
                else:
                    logger.error("Indexing succeeded but vector store is None")
                    return False
            else:
                logger.error("Failed to index documents")
                return False
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return False
    
    def evaluate(self, test_cases: List[TestCase]) -> EvaluationResult:
        """Evaluate the RAG system"""
        try:
            logger.info(f"Evaluating system with {len(test_cases)} test cases")

            if self._evaluator is None:
                from ..evaluation.evaluation_manager import EvaluationManager
                self._evaluator = EvaluationManager()

            responses = [self.query(tc.question) for tc in test_cases]

            results_dict = self._evaluator.evaluate_comprehensive(test_cases, responses)
            
            # Convert dict to EvaluationResult object
            return EvaluationResult(
                overall_score=results_dict.get('evaluation_summary', {}).get('overall_score', 0.0),
                metric_scores=results_dict.get('aggregated_metrics', {}),
                test_case_results=results_dict.get('test_case_analysis', {}).get('case_performance', []),
                recommendations=results_dict.get('recommendations', [])
            )
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            return EvaluationResult(
                overall_score=0.0,
                metric_scores={"error": 0.0},
                test_case_results=[],
                recommendations=[f"Fix error: {str(e)}"]
            )
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the RAG system configuration"""
        return {
            "version": "0.1.0",
            "config": {
                "llm_provider": self.config.llm_provider,
                "llm_model": self.config.llm_model,
                "embedding_provider": self.config.embedding_provider,
                "embedding_model": self.config.embedding_model,
                "vector_store": self.config.vector_store,
                "indexing_strategy": self.config.indexing_strategy,
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "retrieval_k": self.config.retrieval_k,
            },
            "components": {
                "indexer": self._indexer is not None,
                "retriever": self._retriever is not None,
                "query_processor": self._query_processor is not None,
                "router": self._router is not None,
                "generator": self._generator is not None,
                "evaluator": self._evaluator is not None,
            },
            "stats": {
                "indexed_documents": self._indexer.get_document_count() if self._indexer else 0,
                "indexed_chunks": self._indexer.get_chunk_count() if self._indexer else 0,
                "retriever_ready": self._retriever.is_ready() if self._retriever else False,
            }
        }
    
    def load_web_documents(self, urls: List[str], **kwargs) -> bool:
        """
        Load documents from web URLs and add them to the system.
        
        Args:
            urls: List of URLs to load
            **kwargs: Additional arguments for web loading
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Loading documents from {len(urls)} URLs")
            
            # Use the indexer's web loading capability
            documents = self._indexer.load_web_documents(urls, **kwargs)
            
            if not documents:
                logger.warning("No documents loaded from URLs")
                return False
            
            # Add the loaded documents to the system
            return self.add_documents(documents)
            
        except Exception as e:
            logger.error(f"Error loading web documents: {str(e)}")
            return False
    
    def clear_documents(self) -> bool:
        """
        Clear all indexed documents from the system.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Clearing all indexed documents")
            
            # Clear the indexer
            success = self._indexer.clear_index()
            
            if success:
                # Reset the retriever by setting vectorstore to None
                self._retriever.vectorstore = None
                self._retriever._retriever = None
                logger.info("Successfully cleared all documents")
                return True
            else:
                logger.error("Failed to clear documents")
                return False
                
        except Exception as e:
            logger.error(f"Error clearing documents: {str(e)}")
            return False
    
    def get_document_count(self) -> int:
        """
        Get the number of indexed documents.
        
        Returns:
            int: Number of indexed documents
        """
        return self._indexer.get_document_count() if self._indexer else 0
    
    def get_chunk_count(self) -> int:
        """
        Get the number of indexed chunks.
        
        Returns:
            int: Number of indexed chunks
        """
        return self._indexer.get_chunk_count() if self._indexer else 0
    
    def is_ready(self) -> bool:
        """
        Check if the RAG engine is ready to process queries.
        
        Returns:
            bool: True if ready, False otherwise
        """
        return (self._indexer is not None and 
                self._retriever is not None and 
                self._generator is not None and
                self._retriever.is_ready())