"""
Basic vector retrieval implementation adapting from workplan/00BasicRAGSystem.md
"""

from typing import List, Optional, Tuple
import logging
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from ..core.interfaces import BaseRetriever
from ..core.models import Document
from ..core.config import PipelineConfig


logger = logging.getLogger(__name__)


class VectorRetriever(BaseRetriever):
    """
    Basic vector retriever that performs similarity search on a vector store.
    Adapts the retrieval implementation from workplan/00BasicRAGSystem.md.
    """
    
    def __init__(self, vectorstore: Optional[Chroma] = None, config: Optional[PipelineConfig] = None):
        """
        Initialize the VectorRetriever.
        
        Args:
            vectorstore: Optional Chroma vector store instance
            config: Optional pipeline configuration
        """
        self.vectorstore = vectorstore
        self.config = config or PipelineConfig()
        self._retriever = None
        
        if self.vectorstore is not None:
            self._setup_retriever()
        
        logger.info("VectorRetriever initialized")
    
    def _setup_retriever(self):
        """Setup the LangChain retriever from the vector store"""
        if self.vectorstore is not None:
            # Create retriever using vectorstore.as_retriever() as in workplan
            self._retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.config.retrieval_k}
            )
            logger.info(f"Retriever setup with k={self.config.retrieval_k}")
    
    def set_vectorstore(self, vectorstore: Chroma) -> None:
        """
        Set the vector store for retrieval.
        
        Args:
            vectorstore: Chroma vector store instance
        """
        self.vectorstore = vectorstore
        self._setup_retriever()
        logger.info("Vector store updated")
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve relevant documents for a query using similarity search.
        
        Args:
            query: Query string to search for
            k: Number of documents to retrieve
            
        Returns:
            List[Document]: Retrieved documents
        """
        try:
            if self._retriever is None:
                logger.warning("No retriever available - vector store not set")
                return []
            
            # Update k parameter if different from config
            if k != self.config.retrieval_k:
                temp_retriever = self.vectorstore.as_retriever(
                    search_kwargs={"k": k}
                )
                langchain_docs = temp_retriever.get_relevant_documents(query)
            else:
                langchain_docs = self._retriever.get_relevant_documents(query)
            
            # Convert LangChain documents to our Document format
            documents = []
            for i, doc in enumerate(langchain_docs):
                documents.append(Document(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    doc_id=doc.metadata.get('doc_id', f'retrieved_doc_{i}')
                ))
            
            logger.info(f"Retrieved {len(documents)} documents for query: '{query[:50]}...'")
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def retrieve_with_scores(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Retrieve documents with relevance scores.
        
        Args:
            query: Query string to search for
            k: Number of documents to retrieve
            
        Returns:
            List[Tuple[Document, float]]: List of (document, score) tuples
        """
        try:
            if self.vectorstore is None:
                logger.warning("No vector store available")
                return []
            
            # Use similarity_search_with_score for scoring
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            # Convert to our Document format with scores
            documents_with_scores = []
            for i, (langchain_doc, score) in enumerate(results):
                document = Document(
                    content=langchain_doc.page_content,
                    metadata=langchain_doc.metadata,
                    doc_id=langchain_doc.metadata.get('doc_id', f'retrieved_doc_{i}')
                )
                documents_with_scores.append((document, score))
            
            logger.info(f"Retrieved {len(documents_with_scores)} documents with scores for query: '{query[:50]}...'")
            return documents_with_scores
            
        except Exception as e:
            logger.error(f"Error retrieving documents with scores: {str(e)}")
            return []
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Perform similarity search (alias for retrieve method).
        
        Args:
            query: Query string to search for
            k: Number of documents to retrieve
            
        Returns:
            List[Document]: Retrieved documents
        """
        return self.retrieve(query, k)
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Get relevant documents using the configured k value.
        Matches the interface used in workplan/00BasicRAGSystem.md
        
        Args:
            query: Query string to search for
            
        Returns:
            List[Document]: Retrieved documents
        """
        return self.retrieve(query, self.config.retrieval_k)
    
    def is_ready(self) -> bool:
        """
        Check if the retriever is ready for use.
        
        Returns:
            bool: True if retriever is ready, False otherwise
        """
        return self._retriever is not None
    
    def get_document_count(self) -> int:
        """
        Get the number of documents in the vector store.
        
        Returns:
            int: Number of documents, or 0 if vector store not available
        """
        try:
            if self.vectorstore is not None:
                collection = self.vectorstore._collection
                return collection.count()
            return 0
        except Exception as e:
            logger.warning(f"Could not get document count: {str(e)}")
            return 0
    
    def update_config(self, config: PipelineConfig) -> None:
        """
        Update the retriever configuration.
        
        Args:
            config: New pipeline configuration
        """
        self.config = config
        if self.vectorstore is not None:
            self._setup_retriever()
        logger.info("Retriever configuration updated")