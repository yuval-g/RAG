"""
Base interfaces and abstract classes for extensibility
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from .models import Document, ProcessedQuery, RAGResponse, EvaluationResult, TestCase, RoutingDecision, RouteConfig


class BaseIndexer(ABC):
    """Abstract base class for document indexers"""
    
    @abstractmethod
    def index_documents(self, documents: List[Document]) -> bool:
        """Index a list of documents"""
        pass
    
    @abstractmethod
    def get_document_count(self) -> int:
        """Get the number of indexed documents"""
        pass
    
    @abstractmethod
    def clear_index(self) -> bool:
        """Clear all indexed documents"""
        pass


class BaseRetriever(ABC):
    """Abstract base class for document retrievers"""
    
    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant documents for a query"""
        pass
    
    @abstractmethod
    def retrieve_with_scores(self, query: str, k: int = 5) -> List[tuple[Document, float]]:
        """Retrieve documents with relevance scores"""
        pass


class BaseQueryProcessor(ABC):
    """Abstract base class for query processors"""
    
    @abstractmethod
    def process(self, query: str, strategy: str = "basic") -> ProcessedQuery:
        """Process a query using specified strategy"""
        pass
    
    @abstractmethod
    def get_available_strategies(self) -> List[str]:
        """Get list of available processing strategies"""
        pass


class BaseRouter(ABC):
    """Abstract base class for query routers"""
    
    @abstractmethod
    def route(self, query: str) -> RoutingDecision:
        """Route a query to appropriate destination"""
        pass
    
    @abstractmethod
    def add_route(self, route_config: RouteConfig) -> bool:
        """Add a new routing destination"""
        pass
    
    @abstractmethod
    def get_available_routes(self) -> List[str]:
        """Get list of available routes"""
        pass


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using the LLM"""
        pass
    
    @abstractmethod
    def generate_with_structured_output(self, prompt: str, schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate structured output using the LLM"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        pass


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents"""
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query"""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        pass


class BaseVectorStore(ABC):
    """Abstract base class for vector stores"""
    
    @abstractmethod
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> bool:
        """Add documents with embeddings to the store"""
        pass
    
    @abstractmethod
    def similarity_search(self, query_embedding: List[float], k: int = 5) -> List[Document]:
        """Search for similar documents"""
        pass
    
    @abstractmethod
    def similarity_search_with_scores(self, query_embedding: List[float], k: int = 5) -> List[tuple[Document, float]]:
        """Search for similar documents with scores"""
        pass
    
    @abstractmethod
    def delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete documents by IDs"""
        pass


class BaseVectorStoreProvider(ABC):
    """Abstract base class for vector store providers"""
    
    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """Initialize the vector store"""
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[Document], embeddings: Optional[List[List[float]]] = None) -> bool:
        """Add documents with embeddings to the store"""
        pass
    
    @abstractmethod
    def similarity_search(self, query_embedding: List[float], k: int = 5) -> List[Document]:
        """Search for similar documents"""
        pass
    
    @abstractmethod
    def similarity_search_with_scores(self, query_embedding: List[float], k: int = 5) -> List[tuple[Document, float]]:
        """Search for similar documents with scores"""
        pass
    
    @abstractmethod
    def delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete documents by IDs"""
        pass
    
    @abstractmethod
    def get_document_count(self) -> int:
        """Get the number of documents in the store"""
        pass
    
    @abstractmethod
    def clear_store(self) -> bool:
        """Clear all documents from the store"""
        pass
    
    @abstractmethod
    def get_store_info(self) -> Dict[str, Any]:
        """Get information about the vector store"""
        pass


class BaseEvaluator(ABC):
    """Abstract base class for evaluators"""
    
    @abstractmethod
    def evaluate(self, test_cases: List[TestCase], responses: List[RAGResponse]) -> EvaluationResult:
        """Evaluate RAG responses against test cases"""
        pass
    
    @abstractmethod
    def get_supported_metrics(self) -> List[str]:
        """Get list of supported evaluation metrics"""
        pass


class BaseReRanker(ABC):
    """Abstract base class for document re-rankers"""
    
    @abstractmethod
    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """Re-rank documents based on relevance to query"""
        pass
    
    @abstractmethod
    def rerank_with_scores(self, query: str, documents: List[Document], top_k: int = 5) -> List[tuple[Document, float]]:
        """Re-rank documents with relevance scores"""
        pass


class BaseRAGEngine(ABC):
    """Abstract base class for RAG engines"""
    
    @abstractmethod
    def query(self, question: str, **kwargs) -> RAGResponse:
        """Process a query through the RAG pipeline"""
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the RAG system"""
        pass
    
    @abstractmethod
    def evaluate(self, test_cases: List[TestCase]) -> EvaluationResult:
        """Evaluate the RAG system"""
        pass
    
    @abstractmethod
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the RAG system configuration"""
        pass