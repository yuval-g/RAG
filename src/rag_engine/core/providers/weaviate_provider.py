"""
Weaviate vector store provider implementation
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import uuid
from langchain_weaviate import WeaviateVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document as LangChainDocument
import weaviate
from weaviate.classes.init import Auth

from ..vector_store_providers import VectorStoreProvider
from ..models import Document
from ..config import PipelineConfig
from ..exceptions import VectorStoreError

logger = logging.getLogger(__name__)


class WeaviateProvider(VectorStoreProvider):
    """Weaviate vector store provider implementation"""
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self._embeddings = None
        self._class_name = "Document"
        self._weaviate_client = None
        self._url = "http://localhost:8080"
    
    def initialize(self, **kwargs) -> None:
        """Initialize the Weaviate vector store"""
        try:
            # Extract configuration
            self._url = kwargs.get('url', 'http://localhost:8080')
            api_key = kwargs.get('api_key')
            self._class_name = kwargs.get('class_name', 'Document')
            
            # Set up embeddings
            self._setup_embeddings()
            
            # Initialize Weaviate client (v4 API)
            # Parse URL to extract host and port
            from urllib.parse import urlparse
            parsed_url = urlparse(self._url)
            host = parsed_url.hostname or 'localhost'
            port = parsed_url.port or 8080
            
            if api_key:
                auth_config = Auth.api_key(api_key)
                self._weaviate_client = weaviate.connect_to_custom(
                    http_host=host,
                    http_port=port,
                    http_secure=parsed_url.scheme == 'https',
                    auth_credentials=auth_config
                )
            else:
                self._weaviate_client = weaviate.connect_to_local(
                    host=host,
                    port=port
                )
            
            # Ensure class schema exists
            self._ensure_class_exists()
            
            # Initialize LangChain Weaviate vector store
            self._store = WeaviateVectorStore(
                client=self._weaviate_client,
                index_name=self._class_name,
                text_key="content",
                embedding=self._embeddings
            )
            
            logger.info(f"Weaviate vector store initialized with class: {self._class_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate vector store: {str(e)}")
            raise VectorStoreError(f"Weaviate initialization failed: {str(e)}")
    
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
            # Default to OpenAI embeddings
            from langchain_openai import OpenAIEmbeddings
            self._embeddings = OpenAIEmbeddings(
                model=self.config.embedding_model,
                openai_api_key=self.config.openai_api_key
            )
    
    def _ensure_class_exists(self) -> None:
        """Ensure the Weaviate class schema exists"""
        try:
            # Check if collection exists (v4 API uses collections instead of classes)
            if self._weaviate_client.collections.exists(self._class_name):
                logger.info(f"Using existing Weaviate collection: {self._class_name}")
                return
            
            logger.info(f"Creating Weaviate collection: {self._class_name}")
            
            # Create collection with v4 API
            from weaviate.classes.config import Configure, Property, DataType
            
            self._weaviate_client.collections.create(
                name=self._class_name,
                description="Document collection for RAG system",
                properties=[
                    Property(name="content", data_type=DataType.TEXT, description="Document content"),
                    Property(name="doc_id", data_type=DataType.TEXT, description="Document ID"),
                    Property(name="metadata", data_type=DataType.OBJECT, description="Document metadata")
                ],
                vectorizer_config=Configure.Vectorizer.none()  # We'll provide our own vectors
            )
            logger.info(f"Created Weaviate collection: {self._class_name}")
                
        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {str(e)}")
            raise VectorStoreError(f"Collection creation failed: {str(e)}")
    
    def add_documents(self, documents: List[Document], embeddings: Optional[List[List[float]]] = None) -> bool:
        """Add documents with embeddings to the Weaviate store"""
        if not self._store:
            raise VectorStoreError("Vector store not initialized")
        
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
                else:
                    # Generate ID if not provided
                    langchain_doc.metadata['doc_id'] = str(uuid.uuid4())
                langchain_docs.append(langchain_doc)
            
            # Add documents to Weaviate
            if embeddings:
                # Add with pre-computed embeddings
                ids = [doc.metadata['doc_id'] for doc in langchain_docs]
                self._store.add_documents(langchain_docs, embeddings=embeddings, ids=ids)
            else:
                # Let Weaviate compute embeddings
                self._store.add_documents(langchain_docs)
            
            logger.info(f"Added {len(documents)} documents to Weaviate")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to Weaviate: {str(e)}")
            raise VectorStoreError(f"Failed to add documents: {str(e)}")
    
    def similarity_search(self, query_embedding: List[float], k: int = 5) -> List[Document]:
        """Search for similar documents using embedding"""
        if not self._store:
            raise VectorStoreError("Vector store not initialized")
        
        try:
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
            raise VectorStoreError(f"Similarity search failed: {str(e)}")
    
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
        """Delete documents by IDs"""
        if not self._weaviate_client:
            raise VectorStoreError("Vector store not initialized")
        
        try:
            # Get collection and delete documents by doc_id property (v4 API)
            collection = self._weaviate_client.collections.get(self._class_name)
            
            for doc_id in doc_ids:
                from weaviate.classes.query import Filter
                collection.data.delete_many(
                    where=Filter.by_property("doc_id").equal(doc_id)
                )
            
            logger.info(f"Deleted {len(doc_ids)} documents from Weaviate")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {str(e)}")
            raise VectorStoreError(f"Failed to delete documents: {str(e)}")
    
    def get_document_count(self) -> int:
        """Get the number of documents in the store"""
        if not self._weaviate_client:
            return 0
        
        try:
            # Use v4 API to get document count
            collection = self._weaviate_client.collections.get(self._class_name)
            result = collection.aggregate.over_all(total_count=True)
            return result.total_count or 0
        except Exception as e:
            logger.error(f"Failed to get document count: {str(e)}")
            return 0
    
    def clear_store(self) -> bool:
        """Clear all documents from the store"""
        if not self._weaviate_client:
            return True
        
        try:
            # Delete all objects in the collection (v4 API)
            collection = self._weaviate_client.collections.get(self._class_name)
            collection.data.delete_many(where=None)  # Delete all objects
            
            logger.info("Cleared all documents from Weaviate store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear store: {str(e)}")
            raise VectorStoreError(f"Failed to clear store: {str(e)}")
    
    def get_store_info(self) -> Dict[str, Any]:
        """Get information about the Weaviate vector store"""
        info = {
            'provider': 'weaviate',
            'class_name': self._class_name,
            'url': self._url,
            'document_count': self.get_document_count(),
            'embedding_model': self.config.embedding_model,
            'embedding_provider': self.config.embedding_provider
        }
        
        if self._weaviate_client:
            try:
                # Get cluster info (v4 API)
                cluster_info = self._weaviate_client.cluster.nodes(collection=self._class_name)
                info['cluster_status'] = cluster_info
            except Exception:
                pass
        
        return info