"""
Async interfaces for the RAG system to support concurrent operations.
Implements requirement 8.4 for async/await support.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncIterator
import asyncio
import logging
from functools import wraps
from .models import Document, ProcessedQuery, RAGResponse, EvaluationResult, TestCase, RoutingDecision, RouteConfig

logger = logging.getLogger(__name__)


class AsyncBaseIndexer(ABC):
    """Async abstract base class for document indexers"""
    
    @abstractmethod
    async def index_documents(self, documents: List[Document]) -> bool:
        """Async index a list of documents"""
        pass
    
    @abstractmethod
    async def get_document_count(self) -> int:
        """Async get the number of indexed documents"""
        pass
    
    @abstractmethod
    async def clear_index(self) -> bool:
        """Async clear all indexed documents"""
        pass


class AsyncBaseRetriever(ABC):
    """Async abstract base class for document retrievers"""
    
    @abstractmethod
    async def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Async retrieve relevant documents for a query"""
        pass
    
    @abstractmethod
    async def retrieve_with_scores(self, query: str, k: int = 5) -> List[tuple[Document, float]]:
        """Async retrieve documents with relevance scores"""
        pass


class AsyncBaseQueryProcessor(ABC):
    """Async abstract base class for query processors"""
    
    @abstractmethod
    async def process(self, query: str, strategy: str = "basic") -> ProcessedQuery:
        """Async process a query using specified strategy"""
        pass
    
    @abstractmethod
    async def get_available_strategies(self) -> List[str]:
        """Async get list of available processing strategies"""
        pass


class AsyncBaseRouter(ABC):
    """Async abstract base class for query routers"""
    
    @abstractmethod
    async def route(self, query: str) -> RoutingDecision:
        """Async route a query to appropriate destination"""
        pass
    
    @abstractmethod
    async def add_route(self, route_config: RouteConfig) -> bool:
        """Async add a new routing destination"""
        pass
    
    @abstractmethod
    async def get_available_routes(self) -> List[str]:
        """Async get list of available routes"""
        pass


class AsyncBaseLLMProvider(ABC):
    """Async abstract base class for LLM providers"""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Async generate text using the LLM"""
        pass
    
    @abstractmethod
    async def generate_with_structured_output(self, prompt: str, schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Async generate structured output using the LLM"""
        pass
    
    @abstractmethod
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Async generate text with streaming"""
        pass
    
    @abstractmethod
    async def get_model_info(self) -> Dict[str, Any]:
        """Async get information about the model"""
        pass


class AsyncBaseEmbeddingProvider(ABC):
    """Async abstract base class for embedding providers"""
    
    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async generate embeddings for a list of documents"""
        pass
    
    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """Async generate embedding for a single query"""
        pass
    
    @abstractmethod
    async def get_embedding_dimension(self) -> int:
        """Async get the dimension of embeddings"""
        pass


class AsyncBaseVectorStoreProvider(ABC):
    """Async abstract base class for vector store providers"""
    
    @abstractmethod
    async def initialize(self, **kwargs) -> None:
        """Async initialize the vector store"""
        pass
    
    @abstractmethod
    async def add_documents(self, documents: List[Document], embeddings: Optional[List[List[float]]] = None) -> bool:
        """Async add documents with embeddings to the store"""
        pass
    
    @abstractmethod
    async def similarity_search(self, query_embedding: List[float], k: int = 5) -> List[Document]:
        """Async search for similar documents"""
        pass
    
    @abstractmethod
    async def similarity_search_with_scores(self, query_embedding: List[float], k: int = 5) -> List[tuple[Document, float]]:
        """Async search for similar documents with scores"""
        pass
    
    @abstractmethod
    async def delete_documents(self, doc_ids: List[str]) -> bool:
        """Async delete documents by IDs"""
        pass
    
    @abstractmethod
    async def get_document_count(self) -> int:
        """Async get the number of documents in the store"""
        pass
    
    @abstractmethod
    async def clear_store(self) -> bool:
        """Async clear all documents from the store"""
        pass
    
    @abstractmethod
    async def get_store_info(self) -> Dict[str, Any]:
        """Async get information about the vector store"""
        pass


class AsyncBaseEvaluator(ABC):
    """Async abstract base class for evaluators"""
    
    @abstractmethod
    async def evaluate(self, test_cases: List[TestCase], responses: List[RAGResponse]) -> EvaluationResult:
        """Async evaluate RAG responses against test cases"""
        pass
    
    @abstractmethod
    async def get_supported_metrics(self) -> List[str]:
        """Async get list of supported evaluation metrics"""
        pass


class AsyncBaseReRanker(ABC):
    """Async abstract base class for document re-rankers"""
    
    @abstractmethod
    async def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """Async re-rank documents based on relevance to query"""
        pass
    
    @abstractmethod
    async def rerank_with_scores(self, query: str, documents: List[Document], top_k: int = 5) -> List[tuple[Document, float]]:
        """Async re-rank documents with relevance scores"""
        pass


class AsyncBaseRAGEngine(ABC):
    """Async abstract base class for RAG engines"""
    
    @abstractmethod
    async def query(self, question: str, **kwargs) -> RAGResponse:
        """Async process a query through the RAG pipeline"""
        pass
    
    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> bool:
        """Async add documents to the RAG system"""
        pass
    
    @abstractmethod
    async def evaluate(self, test_cases: List[TestCase]) -> EvaluationResult:
        """Async evaluate the RAG system"""
        pass
    
    @abstractmethod
    async def get_system_info(self) -> Dict[str, Any]:
        """Async get information about the RAG system configuration"""
        pass


class AsyncBatchProcessor:
    """Async batch processor for improved performance"""
    
    def __init__(self, 
                 batch_size: int = 10,
                 max_wait_time: float = 1.0,
                 processor_func: Optional[callable] = None):
        """
        Initialize async batch processor
        
        Args:
            batch_size: Maximum batch size
            max_wait_time: Maximum time to wait for batch to fill
            processor_func: Async function to process batches
        """
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.processor_func = processor_func
        
        self._queue: asyncio.Queue = asyncio.Queue()
        self._results: Dict[str, Any] = {}
        self._processing_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
    
    async def start(self) -> None:
        """Start the batch processor"""
        if self._processing_task is None:
            self._processing_task = asyncio.create_task(self._process_loop())
    
    async def stop(self) -> None:
        """Stop the batch processor"""
        self._shutdown_event.set()
        if self._processing_task:
            await self._processing_task
    
    async def submit(self, item: Any, item_id: str) -> Any:
        """
        Submit item for batch processing
        
        Args:
            item: Item to process
            item_id: Unique identifier for the item
            
        Returns:
            Processing result
        """
        await self._queue.put((item, item_id))
        
        # Wait for result
        while item_id not in self._results:
            await asyncio.sleep(0.01)
        
        result = self._results.pop(item_id)
        if isinstance(result, Exception):
            raise result
        return result
    
    async def _process_loop(self) -> None:
        """Main async processing loop"""
        while not self._shutdown_event.is_set():
            try:
                batch = await self._get_batch()
                if batch and self.processor_func:
                    await self._process_batch(batch)
            except Exception as e:
                logger.error(f"Error in async batch processing: {e}")
    
    async def _get_batch(self) -> List[tuple]:
        """Get a batch of items to process"""
        batch = []
        
        try:
            # Wait for first item or timeout
            item = await asyncio.wait_for(self._queue.get(), timeout=self.max_wait_time)
            batch.append(item)
            
            # Collect additional items up to batch size
            for _ in range(self.batch_size - 1):
                try:
                    item = self._queue.get_nowait()
                    batch.append(item)
                except asyncio.QueueEmpty:
                    break
                    
        except asyncio.TimeoutError:
            pass
        
        return batch
    
    async def _process_batch(self, batch: List[tuple]) -> None:
        """Process a batch of items"""
        try:
            items = [item for item, _ in batch]
            item_ids = [item_id for _, item_id in batch]
            
            # Process the batch
            results = await self.processor_func(items)
            
            # Store results
            for item_id, result in zip(item_ids, results):
                self._results[item_id] = result
                
        except Exception as e:
            logger.error(f"Error processing async batch: {e}")
            # Store error for all items in batch
            for _, item_id in batch:
                self._results[item_id] = Exception(f"Async batch processing failed: {e}")


def async_cached(ttl: Optional[float] = None, key_func: Optional[callable] = None):
    """Decorator for caching async function results"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            from .performance import get_cache_manager
            cache = get_cache_manager()
            
            # Generate cache key
            if key_func:
                cache_key = await key_func(*args, **kwargs) if asyncio.iscoroutinefunction(key_func) else key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{cache._generate_key(*args, **kwargs)}"
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator


class AsyncConnectionPool:
    """Async connection pool for managing async connections"""
    
    def __init__(self, 
                 connection_factory: callable,
                 max_connections: int = 10,
                 connection_timeout: float = 30.0):
        """
        Initialize async connection pool
        
        Args:
            connection_factory: Async function to create new connections
            max_connections: Maximum number of connections
            connection_timeout: Timeout for getting connections
        """
        self.connection_factory = connection_factory
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        
        self._pool: asyncio.Queue = asyncio.Queue(maxsize=max_connections)
        self._active_connections = set()
        self._lock = asyncio.Lock()
    
    async def get_connection(self) -> Any:
        """Get a connection from the pool"""
        try:
            # Try to get from pool first
            connection = self._pool.get_nowait()
            return connection
        except asyncio.QueueEmpty:
            pass
        
        async with self._lock:
            # Check if we can create a new connection
            if len(self._active_connections) < self.max_connections:
                connection = await self.connection_factory()
                self._active_connections.add(connection)
                return connection
        
        # Wait for a connection to become available
        return await asyncio.wait_for(self._pool.get(), timeout=self.connection_timeout)
    
    async def return_connection(self, connection: Any) -> None:
        """Return a connection to the pool"""
        try:
            self._pool.put_nowait(connection)
        except asyncio.QueueFull:
            # Pool is full, close the connection
            if hasattr(connection, 'close'):
                await connection.close()
            async with self._lock:
                self._active_connections.discard(connection)
    
    async def close_all(self) -> None:
        """Close all connections"""
        # Close pooled connections
        while not self._pool.empty():
            try:
                connection = self._pool.get_nowait()
                if hasattr(connection, 'close'):
                    await connection.close()
            except asyncio.QueueEmpty:
                break
        
        # Close active connections
        async with self._lock:
            for connection in self._active_connections.copy():
                if hasattr(connection, 'close'):
                    await connection.close()
            self._active_connections.clear()