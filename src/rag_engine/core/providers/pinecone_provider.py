"""
Pinecone vector store provider implementation
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import uuid
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document as LangChainDocument
from pinecone import Pinecone, ServerlessSpec

from ..vector_store_providers import VectorStoreProvider
from ..models import Document
from ..config import PipelineConfig
from ..exceptions import VectorStoreError

logger = logging.getLogger(__name__)


class PineconeProvider(VectorStoreProvider):
    """Pinecone vector store provider implementation"""
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self._embeddings = None
        self._index_name = "default"
        self._pinecone_client = None
        self._dimension = 1536  # Default for OpenAI embeddings
    
    def initialize(self, **kwargs) -> None:
        """Initialize the Pinecone vector store"""
        try:
            # Extract configuration
            api_key = kwargs.get('api_key') or self.config.vector_store_config.get('api_key')
            if not api_key:
                raise VectorStoreError("Pinecone API key is required")
            
            self._index_name = kwargs.get('index_name', 'default')
            environment = kwargs.get('environment', 'us-east-1-aws')
            self._dimension = kwargs.get('dimension', 1536)
            
            # Set up embeddings first to get dimension
            self._setup_embeddings()
            
            # Initialize Pinecone client
            self._pinecone_client = Pinecone(api_key=api_key)
            
            # Create index if it doesn't exist
            self._ensure_index_exists(environment)
            
            # Initialize LangChain Pinecone vector store
            self._store = PineconeVectorStore(
                index_name=self._index_name,
                embedding=self._embeddings,
                pinecone_api_key=api_key
            )
            
            logger.info(f"Pinecone vector store initialized with index: {self._index_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone vector store: {str(e)}")
            raise VectorStoreError(f"Pinecone initialization failed: {str(e)}")
    
    def _setup_embeddings(self) -> None:
        """Set up embedding function based on configuration"""
        if self.config.embedding_provider == "google":
            if not self.config.google_api_key:
                raise VectorStoreError("Google API key required for Google embeddings")
            
            self._embeddings = GoogleGenerativeAIEmbeddings(
                model=self.config.embedding_model or "models/embedding-001",
                google_api_key=self.config.google_api_key
            )
            # Google embeddings have different dimensions
            self._dimension = 768  # Default for Google embeddings
        else:
            # Default to OpenAI embeddings
            from langchain_openai import OpenAIEmbeddings
            self._embeddings = OpenAIEmbeddings(
                model=self.config.embedding_model,
                openai_api_key=self.config.openai_api_key
            )
            self._dimension = 1536  # OpenAI embedding dimension
    
    def _ensure_index_exists(self, environment: str) -> None:
        """Ensure the Pinecone index exists"""
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self._pinecone_client.list_indexes()]
            
            if self._index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self._index_name}")
                
                # Create index with serverless spec
                self._pinecone_client.create_index(
                    name=self._index_name,
                    dimension=self._dimension,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region=environment
                    )
                )
                logger.info(f"Created Pinecone index: {self._index_name}")
            else:
                logger.info(f"Using existing Pinecone index: {self._index_name}")
                
        except Exception as e:
            logger.error(f"Failed to ensure index exists: {str(e)}")
            raise VectorStoreError(f"Index creation failed: {str(e)}")
    
    def add_documents(self, documents: List[Document], embeddings: Optional[List[List[float]]] = None) -> bool:
        """Add documents with embeddings to the Pinecone store"""
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
            
            # Add documents to Pinecone
            if embeddings:
                # Add with pre-computed embeddings
                ids = [doc.metadata['doc_id'] for doc in langchain_docs]
                self._store.add_documents(langchain_docs, embeddings=embeddings, ids=ids)
            else:
                # Let Pinecone compute embeddings
                self._store.add_documents(langchain_docs)
            
            logger.info(f"Added {len(documents)} documents to Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to Pinecone: {str(e)}")
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
        if not self._store:
            raise VectorStoreError("Vector store not initialized")
        
        try:
            # Pinecone uses delete method with IDs
            self._store.delete(ids=doc_ids)
            logger.info(f"Deleted {len(doc_ids)} documents from Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {str(e)}")
            raise VectorStoreError(f"Failed to delete documents: {str(e)}")
    
    def get_document_count(self) -> int:
        """Get the number of documents in the store"""
        if not self._pinecone_client:
            return 0
        
        try:
            index = self._pinecone_client.Index(self._index_name)
            stats = index.describe_index_stats()
            return stats.total_vector_count
        except Exception as e:
            logger.error(f"Failed to get document count: {str(e)}")
            return 0
    
    def clear_store(self) -> bool:
        """Clear all documents from the store"""
        if not self._pinecone_client:
            return True
        
        try:
            index = self._pinecone_client.Index(self._index_name)
            # Delete all vectors in the index
            index.delete(delete_all=True)
            
            logger.info("Cleared all documents from Pinecone store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear store: {str(e)}")
            raise VectorStoreError(f"Failed to clear store: {str(e)}")
    
    def get_store_info(self) -> Dict[str, Any]:
        """Get information about the Pinecone vector store"""
        info = {
            'provider': 'pinecone',
            'index_name': self._index_name,
            'document_count': self.get_document_count(),
            'embedding_model': self.config.embedding_model,
            'embedding_provider': self.config.embedding_provider,
            'dimension': self._dimension
        }
        
        if self._pinecone_client:
            try:
                index = self._pinecone_client.Index(self._index_name)
                stats = index.describe_index_stats()
                info['index_stats'] = {
                    'total_vector_count': stats.total_vector_count,
                    'dimension': stats.dimension
                }
            except Exception:
                pass
        
        return info