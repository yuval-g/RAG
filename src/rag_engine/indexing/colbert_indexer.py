"""
ColBERT indexer implementation adapting from workplan/03AdvancedIndexingStrategies.md
Provides token-level precision indexing using RAGatouille
"""

import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from ..core.interfaces import BaseIndexer
from ..core.models import Document
from ..core.config import PipelineConfig


logger = logging.getLogger(__name__)


class ColBERTIndexer(BaseIndexer):
    """
    ColBERT indexer that provides token-level precision using RAGatouille.
    Adapts the implementation from workplan/03AdvancedIndexingStrategies.md.
    """
    
    def __init__(self, config: PipelineConfig, index_name: str = "colbert-index"):
        """
        Initialize the ColBERTIndexer with configuration.
        
        Args:
            config: Pipeline configuration
            index_name: Name for the ColBERT index
        """
        self.config = config
        self.index_name = index_name
        self.max_document_length = 180  # Default from workplan
        self.split_documents = True
        
        # RAGatouille model will be initialized lazily
        self._rag_model = None
        self._document_count = 0
        self._is_indexed = False
        
        logger.info(f"ColBERTIndexer initialized with index_name='{index_name}'")
    
    @property
    def rag_model(self):
        """Lazy initialization of RAGatouille model"""
        if self._rag_model is None:
            try:
                from ragatouille import RAGPretrainedModel
                
                logger.info("Loading ColBERT pre-trained model")
                self._rag_model = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
                logger.info("ColBERT model loaded successfully")
                
            except ImportError:
                raise ImportError(
                    "RAGatouille is required for ColBERT indexing. "
                    "Install it with: uv add ragatouille"
                )
            except Exception as e:
                logger.error(f"Failed to load ColBERT model: {str(e)}")
                raise
        
        return self._rag_model
    
    def index_documents(self, documents: List[Document]) -> bool:
        """
        Index documents using ColBERT's token-level approach.
        
        Args:
            documents: List of Document objects to index
            
        Returns:
            bool: True if indexing was successful, False otherwise
        """
        try:
            if not documents:
                logger.warning("No documents provided for indexing")
                return True
            
            # Convert Document objects to text collection
            document_texts = []
            for doc in documents:
                # Include metadata in the text if available
                text = doc.content
                if doc.metadata:
                    # Add title or other important metadata to the text
                    if 'title' in doc.metadata:
                        text = f"Title: {doc.metadata['title']}\n\n{text}"
                    if 'author' in doc.metadata:
                        text = f"{text}\n\nAuthor: {doc.metadata['author']}"
                
                document_texts.append(text)
            
            logger.info(f"Indexing {len(document_texts)} documents with ColBERT")
            
            # Index documents using RAGatouille
            # This handles chunking and token-level embedding internally
            self.rag_model.index(
                collection=document_texts,
                index_name=self.index_name,
                max_document_length=self.max_document_length,
                split_documents=self.split_documents,
            )
            
            self._document_count += len(documents)
            self._is_indexed = True
            
            logger.info(f"Successfully indexed {len(documents)} documents with ColBERT. "
                       f"Total documents: {self._document_count}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error indexing documents with ColBERT: {str(e)}")
            return False
    
    def get_document_count(self) -> int:
        """
        Get the number of indexed documents.
        
        Returns:
            int: Number of documents that have been indexed
        """
        return self._document_count
    
    def clear_index(self) -> bool:
        """
        Clear the ColBERT index.
        
        Returns:
            bool: True if clearing was successful, False otherwise
        """
        try:
            # Reset internal state
            self._document_count = 0
            self._is_indexed = False
            
            # Note: RAGatouille doesn't provide a direct clear method
            # The index files would need to be manually deleted from disk
            # For now, we'll just reset our internal state
            logger.info("ColBERT index state cleared (index files may remain on disk)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error clearing ColBERT index: {str(e)}")
            return False
    
    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Search the ColBERT index for relevant documents.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List[Dict[str, Any]]: Search results with content, score, rank, etc.
        """
        if not self._is_indexed:
            logger.warning("No documents have been indexed yet")
            return []
        
        try:
            logger.info(f"Searching ColBERT index for: '{query}' (k={k})")
            results = self.rag_model.search(query=query, k=k)
            
            logger.info(f"Found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error searching ColBERT index: {str(e)}")
            return []
    
    def as_langchain_retriever(self, k: int = 3):
        """
        Convert the ColBERT model to a LangChain-compatible retriever.
        
        Args:
            k: Number of documents to retrieve
            
        Returns:
            LangChain retriever object
        """
        if not self._is_indexed:
            logger.warning("No documents have been indexed yet")
            return None
        
        try:
            logger.info(f"Creating LangChain retriever with k={k}")
            return self.rag_model.as_langchain_retriever(k=k)
            
        except Exception as e:
            logger.error(f"Error creating LangChain retriever: {str(e)}")
            return None
    
    def get_index_info(self) -> Dict[str, Any]:
        """
        Get information about the ColBERT index.
        
        Returns:
            Dict[str, Any]: Index information
        """
        return {
            "index_name": self.index_name,
            "document_count": self._document_count,
            "is_indexed": self._is_indexed,
            "max_document_length": self.max_document_length,
            "split_documents": self.split_documents,
            "model_name": "colbert-ir/colbertv2.0"
        }
    
    def set_max_document_length(self, length: int) -> None:
        """
        Set the maximum document length for chunking.
        
        Args:
            length: Maximum document length
        """
        self.max_document_length = length
        logger.info(f"Set max_document_length to {length}")
    
    def set_split_documents(self, split: bool) -> None:
        """
        Set whether to split documents during indexing.
        
        Args:
            split: Whether to split documents
        """
        self.split_documents = split
        logger.info(f"Set split_documents to {split}")
    
    def is_ready(self) -> bool:
        """
        Check if the indexer is ready for operations.
        
        Returns:
            bool: True if ready, False otherwise
        """
        try:
            # This will trigger lazy loading if needed
            _ = self.rag_model
            return True
        except Exception:
            return False