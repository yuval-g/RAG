"""
Chroma vector store provider implementation
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document as LangChainDocument

from ..vector_store_providers import VectorStoreProvider
from ..models import Document
from ..config import PipelineConfig
from ..exceptions import VectorStoreError, ExternalServiceError
from ..resilience import resilient_call, ExternalServiceError
from ..resilience import resilient_c

logger = logging.getLogger(__name__)


class ChromaProvider(VectorStoreProvider):
    """Chroma vector store provider implementation"""
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self._embeddings = None
        self._collection_name = "default"
    
    def initialize(self, **kwargs) -> None:
        """Initialize the Chroma vector store"""
        try:
            # Set up embeddings
            self._setup_embeddings()
            
            # Extract configuration
            self._collection_name = kwargs.get('collection_name', 'default')
            persist_directory = kwargs.get('persist_directory', None)
            
            # Initialize Chroma
            if persist_directory:
                self._store = Chroma(
                    collection_name=self._collection_name,
                    embedding_function=self._embeddings,
                    persist_directory=persist_directory
                )
            else:
                self._store = Chroma(
                    collection_name=self._collection_name,
                    embedding_function=self._embeddings
                )
            
            logger.info(f"Chroma vector store initialized with collection: {self._collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Chroma vector store: {str(e)}")
            raise VectorStoreError(f"Chroma initialization failed: {str(e)}")
    
    def _setup_embeddings(self) -> None:
        """Set up embedding function based on configuration"""
        if self.config.embedding_provider == "google":
            if not self.config.google_api_key:
                raise VectorStoreError("Google API key required for Google embeddings")
            
            self._embeddings = GoogleGenerativeAIEmbeddings(
                model=self.config.embedding_model or "models/embedding-001",
                google_api_key=self.config.google_api_key
            )
        else:
            # Default to OpenAI embeddings for backward compatibility
            from langchain_openai import OpenAIEmbeddings
            self._embeddings = OpenAIEmbeddings(
                model=self.config.embedding_model,
                openai_api_key=self.config.openai_api_key
            )
    
    def add_documents(self, documents: List[Document], embeddings: Optional[List[List[float]]] = None) -> bool:
        """Add documents with embeddings to the Chroma store with resilience patterns"""
        if not self._store:
            raise VectorStoreError("Vector store not initialized")
        
        def _add_documents_internal():
            try:
                # Convert to LangChain documents
                langchain_docs = []
                for doc in documents:
                    langchain_doc = LangChainDocument(
                        page_content=doc.content,
                        metadata=doc.metadata.copy()
                    )
                    if doc.doc_id:
                        langchain_doc.metadata['doc_id'] = doc.doc_id
                    langchain_docs.append(langchain_doc)
                
                # Add documents to Chroma
                if embeddings:
                    # Add with pre-computed embeddings
                    self._store.add_documents(langchain_docs, embeddings=embeddings)
                else:
                    # Let Chroma compute embeddings
                    self._store.add_documents(langchain_docs)
                
                logger.info(f"Added {len(documents)} documents to Chroma")
                return True
                
            except Exception as e:
                logger.error(f"Failed to add documents to Chroma: {str(e)}")
                raise ExternalServiceError(f"Failed to add documents: {str(e)}")
        
        def _fallback_add():
            logger.warning("Add documents operation failed, using fallback")
            return False
        
        try:
            return resilient_call(
                _add_documents_internal,
                circuit_breaker_name="chroma_vectorstore_add",
                retry_name="chroma_vectorstore_retry",
                fallback_operation="chroma_add_documents",
                fallback_func=_fallback_add
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to add documents after all resilience attempts: {str(e)}")
    
    def similarity_search(self, query_embedding: List[float], k: int = 5) -> List[Document]:
        """Search for similar documents using embedding with resilience patterns"""
        if not self._store:
            raise VectorStoreError("Vector store not initialized")
        
        def _search_internal():
            try:
                # Chroma doesn't directly support embedding search, so we'll use query text
                # This is a limitation we'll note in the implementation
                results = self._store.similarity_search_by_vector(query_embedding, k=k)
                
                # Convert back to our Document format
                documents = []
                for result in results:
                    doc = Document(
                        content=result.page_content,
                        metadata=result.metadata,
                        doc_id=result.metadata.get('doc_id')
                    )
                    documents.append(doc)
                
                return documents
                
            except Exception as e:
                logger.error(f"Similarity search failed: {str(e)}")
                raise ExternalServiceError(f"Similarity search failed: {str(e)}")
        
        def _fallback_search():
            logger.warning("Similarity search failed, returning empty results")
            return []
        
        try:
            return resilient_call(
                _search_internal,
                circuit_breaker_name="chroma_vectorstore_search",
                retry_name="chroma_vectorstore_retry",
                fallback_operation="chroma_similarity_search",
                fallback_func=_fallback_search
            )
        except Exception as e:
            raise VectorStoreError(f"Similarity search failed after all resilience attempts: {str(e)}")
    
    def similarity_search_with_scores(self, query_embedding: List[float], k: int = 5) -> List[Tuple[Document, float]]:
        """Search for similar documents with scores"""
        if not self._store:
            raise VectorStoreError("Vector store not initialized")
        
        try:
            results = self._store.similarity_search_with_score_by_vector(query_embedding, k=k)
            
            # Convert to our format
            documents_with_scores = []
            for result, score in results:
                doc = Document(
                    content=result.page_content,
                    metadata=result.metadata,
                    doc_id=result.metadata.get('doc_id')
                )
                documents_with_scores.append((doc, score))
            
            return documents_with_scores
            
        except Exception as e:
            logger.error(f"Similarity search with scores failed: {str(e)}")
            raise VectorStoreError(f"Similarity search with scores failed: {str(e)}")
    
    def delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete documents by IDs with resilience patterns"""
        if not self._store:
            raise VectorStoreError("Vector store not initialized")
        
        def _delete_internal():
            try:
                # Chroma uses delete method with IDs
                self._store.delete(ids=doc_ids)
                logger.info(f"Deleted {len(doc_ids)} documents from Chroma")
                return True
                
            except Exception as e:
                logger.error(f"Failed to delete documents: {str(e)}")
                raise ExternalServiceError(f"Failed to delete documents: {str(e)}")
        
        def _fallback_delete():
            logger.warning("Delete documents operation failed, using fallback")
            return False
        
        try:
            return resilient_call(
                _delete_internal,
                circuit_breaker_name="chroma_vectorstore_delete",
                retry_name="chroma_vectorstore_retry",
                fallback_operation="chroma_delete_documents",
                fallback_func=_fallback_delete
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to delete documents after all resilience attempts: {str(e)}")
    
    def get_document_count(self) -> int:
        """Get the number of documents in the store"""
        if not self._store:
            return 0
        
        try:
            collection = self._store._collection
            return collection.count()
        except Exception as e:
            logger.error(f"Failed to get document count: {str(e)}")
            return 0
    
    def clear_store(self) -> bool:
        """Clear all documents from the store"""
        if not self._store:
            return True
        
        try:
            # Get all document IDs and delete them
            collection = self._store._collection
            all_data = collection.get()
            if all_data['ids']:
                self._store.delete(ids=all_data['ids'])
            
            logger.info("Cleared all documents from Chroma store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear store: {str(e)}")
            raise VectorStoreError(f"Failed to clear store: {str(e)}")
    
    def get_store_info(self) -> Dict[str, Any]:
        """Get information about the Chroma vector store"""
        info = {
            'provider': 'chroma',
            'collection_name': self._collection_name,
            'document_count': self.get_document_count(),
            'embedding_model': self.config.embedding_model,
            'embedding_provider': self.config.embedding_provider
        }
        
        if self._store:
            try:
                collection = self._store._collection
                info['collection_metadata'] = collection.metadata
            except Exception:
                pass
        
        return info