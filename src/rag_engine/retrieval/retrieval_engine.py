"""
Enhanced retrieval engine with hybrid retrieval capabilities
"""

from typing import List, Optional, Tuple, Dict, Any, Union
import logging
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core.interfaces import BaseRetriever
from ..core.models import Document
from ..core.config import PipelineConfig
from .vector_retriever import VectorRetriever
from .reranker import ReRanker
from .self_correction import SelfCorrectionEngine


logger = logging.getLogger(__name__)


class RetrievalEngine:
    """
    Enhanced retrieval engine that supports multiple retrieval strategies and re-ranking.
    Implements the advanced retrieval concepts from workplan/04AdvancedRetrieval-Generation.md
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the retrieval engine.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self.retrievers: Dict[str, BaseRetriever] = {}
        self.reranker: Optional[ReRanker] = None
        self.self_correction_engine: Optional[SelfCorrectionEngine] = None
        self.default_retriever_name = "vector"
        
        # Initialize default retrievers
        self.add_retriever("vector", VectorRetriever(config=self.config))
        
        # Add keyword retriever for hybrid search
        try:
            from .keyword_retriever import KeywordRetriever
            self.add_retriever("keyword", KeywordRetriever(config=self.config))
        except ImportError as e:
            logger.warning(f"Could not initialize keyword retriever: {str(e)}")
        
        # Initialize re-ranker if enabled
        if self.config.use_reranking:
            self._setup_reranker()
        
        # Initialize self-correction engine if enabled
        if self.config.enable_self_correction:
            self._setup_self_correction()
        
        logger.info("RetrievalEngine initialized")
    
    def _setup_reranker(self):
        """Setup the re-ranker based on configuration"""
        try:
            # Get the base retriever for contextual compression
            base_retriever = self.retrievers.get(self.default_retriever_name)
            
            if hasattr(base_retriever, '_retriever') and base_retriever._retriever is not None:
                # Use contextual compression strategy with LangChain retriever
                self.reranker = ReRanker(
                    strategy="contextual",
                    base_retriever=base_retriever._retriever,
                    config=self.config
                )
            else:
                # Use LLM-based re-ranking strategy
                self.reranker = ReRanker(
                    strategy="llm",
                    config=self.config
                )
            
            logger.info(f"Re-ranker setup complete with strategy: {self.reranker.get_strategy()}")
            
        except Exception as e:
            logger.error(f"Failed to setup re-ranker: {str(e)}")
            self.reranker = None
    
    def _setup_self_correction(self):
        """Setup the self-correction engine based on configuration"""
        try:
            self.self_correction_engine = SelfCorrectionEngine(self.config)
            logger.info("Self-correction engine setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup self-correction engine: {str(e)}")
            self.self_correction_engine = None
    
    def add_retriever(self, name: str, retriever: BaseRetriever) -> None:
        """
        Add a retriever to the engine.
        
        Args:
            name: Name for the retriever
            retriever: Retriever instance
        """
        self.retrievers[name] = retriever
        logger.info(f"Added retriever: {name}")
    
    def set_default_retriever(self, name: str) -> None:
        """
        Set the default retriever.
        
        Args:
            name: Name of the retriever to set as default
        """
        if name not in self.retrievers:
            raise ValueError(f"Retriever '{name}' not found")
        
        self.default_retriever_name = name
        logger.info(f"Default retriever set to: {name}")
    
    def get_retriever(self, name: Optional[str] = None) -> BaseRetriever:
        """
        Get a retriever by name.
        
        Args:
            name: Name of the retriever (uses default if None)
            
        Returns:
            BaseRetriever: The requested retriever
        """
        retriever_name = name or self.default_retriever_name
        
        if retriever_name not in self.retrievers:
            raise ValueError(f"Retriever '{retriever_name}' not found")
        
        return self.retrievers[retriever_name]
    
    def retrieve(self, query: str, k: int = 5, retriever_name: Optional[str] = None) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query string to search for
            k: Number of documents to retrieve
            retriever_name: Name of retriever to use (uses default if None)
            
        Returns:
            List[Document]: Retrieved documents
        """
        start_time = time.time()
        
        try:
            retriever = self.get_retriever(retriever_name)
            documents = retriever.retrieve(query, k)
            
            retrieval_time = time.time() - start_time
            logger.info(f"Retrieved {len(documents)} documents in {retrieval_time:.3f}s")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def retrieve_with_rerank(self, query: str, k: int = 10, top_k: int = 5, 
                           retriever_name: Optional[str] = None) -> List[Document]:
        """
        Retrieve documents and apply re-ranking.
        Implements the re-ranking approach from workplan/04AdvancedRetrieval-Generation.md
        
        Args:
            query: Query string to search for
            k: Number of documents to retrieve initially
            top_k: Number of documents to return after re-ranking
            retriever_name: Name of retriever to use (uses default if None)
            
        Returns:
            List[Document]: Re-ranked documents
        """
        start_time = time.time()
        
        try:
            # First-pass retrieval: get more documents than needed
            retriever = self.get_retriever(retriever_name)
            initial_docs = retriever.retrieve(query, k)
            
            if not initial_docs:
                logger.warning("No documents retrieved for re-ranking")
                return []
            
            # Apply re-ranking if available
            if self.reranker is not None:
                reranked_docs = self.reranker.rerank(query, initial_docs, top_k)
                
                retrieval_time = time.time() - start_time
                logger.info(f"Retrieved and re-ranked {len(reranked_docs)} documents in {retrieval_time:.3f}s")
                
                return reranked_docs
            else:
                # No re-ranker available, return top_k from initial retrieval
                logger.warning("Re-ranking requested but no re-ranker available")
                return initial_docs[:top_k]
                
        except Exception as e:
            logger.error(f"Error in retrieve_with_rerank: {str(e)}")
            return []
    
    def retrieve_with_scores(self, query: str, k: int = 5, 
                           retriever_name: Optional[str] = None) -> List[Tuple[Document, float]]:
        """
        Retrieve documents with relevance scores.
        
        Args:
            query: Query string to search for
            k: Number of documents to retrieve
            retriever_name: Name of retriever to use (uses default if None)
            
        Returns:
            List[Tuple[Document, float]]: List of (document, score) tuples
        """
        start_time = time.time()
        
        try:
            retriever = self.get_retriever(retriever_name)
            
            if hasattr(retriever, 'retrieve_with_scores'):
                documents_with_scores = retriever.retrieve_with_scores(query, k)
            else:
                # Fallback: retrieve without scores and assign default scores
                documents = retriever.retrieve(query, k)
                documents_with_scores = [(doc, 0.5) for doc in documents]
            
            retrieval_time = time.time() - start_time
            logger.info(f"Retrieved {len(documents_with_scores)} documents with scores in {retrieval_time:.3f}s")
            
            return documents_with_scores
            
        except Exception as e:
            logger.error(f"Error retrieving documents with scores: {str(e)}")
            return []
    
    def retrieve_with_rerank_and_scores(self, query: str, k: int = 10, top_k: int = 5,
                                      retriever_name: Optional[str] = None) -> List[Tuple[Document, float]]:
        """
        Retrieve documents, apply re-ranking, and return with scores.
        
        Args:
            query: Query string to search for
            k: Number of documents to retrieve initially
            top_k: Number of documents to return after re-ranking
            retriever_name: Name of retriever to use (uses default if None)
            
        Returns:
            List[Tuple[Document, float]]: List of (document, score) tuples
        """
        start_time = time.time()
        
        try:
            # First-pass retrieval
            retriever = self.get_retriever(retriever_name)
            initial_docs = retriever.retrieve(query, k)
            
            if not initial_docs:
                logger.warning("No documents retrieved for re-ranking")
                return []
            
            # Apply re-ranking with scores if available
            if self.reranker is not None:
                reranked_docs_with_scores = self.reranker.rerank_with_scores(query, initial_docs, top_k)
                
                retrieval_time = time.time() - start_time
                logger.info(f"Retrieved and re-ranked {len(reranked_docs_with_scores)} documents with scores in {retrieval_time:.3f}s")
                
                return reranked_docs_with_scores
            else:
                # No re-ranker available, return with default scores
                logger.warning("Re-ranking requested but no re-ranker available")
                return [(doc, 0.5) for doc in initial_docs[:top_k]]
                
        except Exception as e:
            logger.error(f"Error in retrieve_with_rerank_and_scores: {str(e)}")
            return []
    
    def enable_reranking(self, strategy: str = "llm") -> None:
        """
        Enable re-ranking with specified strategy.
        
        Args:
            strategy: Re-ranking strategy ("llm" or "contextual")
        """
        try:
            base_retriever = None
            if strategy == "contextual":
                base_retriever_obj = self.retrievers.get(self.default_retriever_name)
                if hasattr(base_retriever_obj, '_retriever'):
                    base_retriever = base_retriever_obj._retriever
            
            self.reranker = ReRanker(
                strategy=strategy,
                base_retriever=base_retriever,
                config=self.config
            )
            
            self.config.use_reranking = True
            logger.info(f"Re-ranking enabled with strategy: {strategy}")
            
        except Exception as e:
            logger.error(f"Failed to enable re-ranking: {str(e)}")
            self.reranker = None
    
    def disable_reranking(self) -> None:
        """Disable re-ranking"""
        self.reranker = None
        self.config.use_reranking = False
        logger.info("Re-ranking disabled")
    
    def is_reranking_enabled(self) -> bool:
        """Check if re-ranking is enabled"""
        return self.reranker is not None
    
    def retrieve_with_correction(self, query: str, k: int = 10, top_k: int = 5,
                               retriever_name: Optional[str] = None) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Retrieve documents with self-correction applied.
        Implements CRAG-style relevance checking from workplan/04AdvancedRetrieval-Generation.md
        
        Args:
            query: Query string to search for
            k: Number of documents to retrieve initially
            top_k: Number of documents to return after correction
            retriever_name: Name of retriever to use (uses default if None)
            
        Returns:
            Tuple[List[Document], Dict[str, Any]]: Corrected documents and metadata
        """
        start_time = time.time()
        
        try:
            # First-pass retrieval
            retriever = self.get_retriever(retriever_name)
            initial_docs = retriever.retrieve(query, k)
            
            if not initial_docs:
                logger.warning("No documents retrieved for correction")
                return [], {"correction_applied": False, "reason": "no_documents"}
            
            # Apply self-correction if available
            if self.self_correction_engine is not None:
                corrected_docs, correction_metadata = self.self_correction_engine.correct_retrieval(
                    query, initial_docs
                )
                
                # Limit to top_k documents
                final_docs = corrected_docs[:top_k]
                
                retrieval_time = time.time() - start_time
                correction_metadata["retrieval_time"] = retrieval_time
                
                logger.info(f"Retrieved and corrected {len(final_docs)} documents in {retrieval_time:.3f}s")
                return final_docs, correction_metadata
            else:
                # No self-correction available, return top_k from initial retrieval
                logger.warning("Self-correction requested but not available")
                return initial_docs[:top_k], {"correction_applied": False, "reason": "not_enabled"}
                
        except Exception as e:
            logger.error(f"Error in retrieve_with_correction: {str(e)}")
            return [], {"correction_applied": False, "error": str(e)}
    
    def enable_self_correction(self) -> None:
        """Enable self-correction mechanisms"""
        try:
            self.self_correction_engine = SelfCorrectionEngine(self.config)
            self.config.enable_self_correction = True
            logger.info("Self-correction enabled")
            
        except Exception as e:
            logger.error(f"Failed to enable self-correction: {str(e)}")
            self.self_correction_engine = None
    
    def disable_self_correction(self) -> None:
        """Disable self-correction mechanisms"""
        self.self_correction_engine = None
        self.config.enable_self_correction = False
        logger.info("Self-correction disabled")
    
    def is_self_correction_enabled(self) -> bool:
        """Check if self-correction is enabled"""
        return self.self_correction_engine is not None
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        Add documents to all retrievers in the engine.
        
        Args:
            documents: List of documents to add
            
        Returns:
            bool: True if successful for all retrievers
        """
        success = True
        
        for name, retriever in self.retrievers.items():
            try:
                if hasattr(retriever, 'add_documents'):
                    result = retriever.add_documents(documents)
                    if not result:
                        logger.warning(f"Failed to add documents to retriever: {name}")
                        success = False
                else:
                    logger.warning(f"Retriever {name} does not support adding documents")
            except Exception as e:
                logger.error(f"Error adding documents to retriever {name}: {str(e)}")
                success = False
        
        logger.info(f"Added {len(documents)} documents to {len(self.retrievers)} retrievers")
        return success
    
    def get_available_retrievers(self) -> List[str]:
        """Get list of available retriever names"""
        return list(self.retrievers.keys())
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval engine statistics"""
        stats = {
            "available_retrievers": self.get_available_retrievers(),
            "default_retriever": self.default_retriever_name,
            "reranking_enabled": self.is_reranking_enabled(),
            "reranking_strategy": self.reranker.get_strategy() if self.reranker else None,
            "self_correction_enabled": self.is_self_correction_enabled(),
            "config": {
                "retrieval_k": self.config.retrieval_k,
                "reranker_top_k": self.config.reranker_top_k,
                "use_reranking": self.config.use_reranking,
                "enable_self_correction": self.config.enable_self_correction,
                "relevance_threshold": getattr(self.config, 'relevance_threshold', 0.7),
                "factuality_threshold": getattr(self.config, 'factuality_threshold', 0.7),
                "min_relevant_docs": getattr(self.config, 'min_relevant_docs', 2)
            }
        }
        
        # Add self-correction stats if available
        if self.self_correction_engine is not None:
            stats["self_correction_stats"] = self.self_correction_engine.get_correction_stats()
        
        # Add retriever-specific stats
        for name, retriever in self.retrievers.items():
            if hasattr(retriever, 'get_document_count'):
                stats[f"{name}_document_count"] = retriever.get_document_count()
            if hasattr(retriever, 'is_ready'):
                stats[f"{name}_ready"] = retriever.is_ready()
        
        return stats
    
    def hybrid_retrieve(self, query: str, k: int = 5, 
                       vector_weight: float = 0.7, keyword_weight: float = 0.3,
                       retriever_names: Optional[List[str]] = None) -> List[Document]:
        """
        Perform hybrid retrieval combining multiple retrieval strategies.
        Implements the hybrid approach from workplan/04AdvancedRetrieval-Generation.md
        
        Args:
            query: Query string to search for
            k: Number of documents to retrieve
            vector_weight: Weight for vector search results (0.0 to 1.0)
            keyword_weight: Weight for keyword search results (0.0 to 1.0)
            retriever_names: List of retriever names to combine (uses all if None)
            
        Returns:
            List[Document]: Hybrid retrieved documents
        """
        start_time = time.time()
        
        try:
            # Determine which retrievers to use
            if retriever_names is None:
                retriever_names = list(self.retrievers.keys())
            
            if len(retriever_names) < 2:
                logger.warning("Hybrid retrieval requires at least 2 retrievers, falling back to single retriever")
                return self.retrieve(query, k, retriever_names[0] if retriever_names else None)
            
            # Retrieve from multiple sources concurrently
            retrieval_results = {}
            with ThreadPoolExecutor(max_workers=len(retriever_names)) as executor:
                future_to_retriever = {
                    executor.submit(self._retrieve_with_scores_safe, query, k * 2, name): name
                    for name in retriever_names
                }
                
                for future in as_completed(future_to_retriever):
                    retriever_name = future_to_retriever[future]
                    try:
                        docs_with_scores = future.result()
                        retrieval_results[retriever_name] = docs_with_scores
                    except Exception as e:
                        logger.error(f"Error retrieving from {retriever_name}: {str(e)}")
                        retrieval_results[retriever_name] = []
            
            # Combine results using weighted scoring
            combined_docs = self._combine_retrieval_results(
                retrieval_results, 
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
                top_k=k
            )
            
            retrieval_time = time.time() - start_time
            logger.info(f"Hybrid retrieved {len(combined_docs)} documents in {retrieval_time:.3f}s")
            
            return combined_docs
            
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {str(e)}")
            return []
    
    def _retrieve_with_scores_safe(self, query: str, k: int, retriever_name: str) -> List[Tuple[Document, float]]:
        """
        Safely retrieve documents with scores from a specific retriever.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            retriever_name: Name of the retriever
            
        Returns:
            List[Tuple[Document, float]]: Documents with scores
        """
        try:
            retriever = self.get_retriever(retriever_name)
            
            if hasattr(retriever, 'retrieve_with_scores'):
                return retriever.retrieve_with_scores(query, k)
            else:
                # Fallback: retrieve without scores and assign default scores
                documents = retriever.retrieve(query, k)
                return [(doc, 0.5) for doc in documents]
                
        except Exception as e:
            logger.error(f"Error retrieving from {retriever_name}: {str(e)}")
            return []
    
    def _combine_retrieval_results(self, retrieval_results: Dict[str, List[Tuple[Document, float]]],
                                 vector_weight: float = 0.7, keyword_weight: float = 0.3,
                                 top_k: int = 5) -> List[Document]:
        """
        Combine retrieval results from multiple sources using weighted scoring.
        
        Args:
            retrieval_results: Dictionary mapping retriever names to (document, score) lists
            vector_weight: Weight for vector-based retrievers
            keyword_weight: Weight for keyword-based retrievers
            top_k: Number of top documents to return
            
        Returns:
            List[Document]: Combined and ranked documents
        """
        # Document scoring dictionary: doc_id -> (document, combined_score)
        doc_scores: Dict[str, Tuple[Document, float]] = {}
        
        for retriever_name, docs_with_scores in retrieval_results.items():
            # Determine weight based on retriever type
            weight = vector_weight if 'vector' in retriever_name.lower() else keyword_weight
            
            for doc, score in docs_with_scores:
                doc_id = doc.doc_id or f"{hash(doc.content[:100])}"
                
                if doc_id in doc_scores:
                    # Combine scores using weighted average
                    existing_doc, existing_score = doc_scores[doc_id]
                    combined_score = (existing_score + (score * weight)) / 2
                    doc_scores[doc_id] = (existing_doc, combined_score)
                else:
                    # New document
                    doc_scores[doc_id] = (doc, score * weight)
        
        # Sort by combined score and return top_k
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in sorted_docs[:top_k]]
    
    def long_context_retrieve(self, query: str, k: int = 20, 
                            context_window_size: int = 100000,
                            retriever_name: Optional[str] = None) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Retrieve documents optimized for long-context models.
        Implements the long-context hybrid approach from workplan/04AdvancedRetrieval-Generation.md
        
        Args:
            query: Query string to search for
            k: Number of documents to retrieve initially
            context_window_size: Maximum context window size in tokens (approximate)
            retriever_name: Name of retriever to use (uses default if None)
            
        Returns:
            Tuple[List[Document], Dict[str, Any]]: Retrieved documents and metadata
        """
        start_time = time.time()
        
        try:
            # First-pass retrieval: get more documents for long context
            retriever = self.get_retriever(retriever_name)
            initial_docs = retriever.retrieve(query, k)
            
            if not initial_docs:
                logger.warning("No documents retrieved for long-context processing")
                return [], {"strategy": "long_context", "documents_processed": 0}
            
            # Estimate token count and filter documents to fit context window
            filtered_docs, total_tokens = self._filter_docs_for_context_window(
                initial_docs, context_window_size
            )
            
            # Apply re-ranking if available to prioritize most relevant documents
            if self.reranker is not None:
                filtered_docs = self.reranker.rerank(query, filtered_docs, len(filtered_docs))
            
            retrieval_time = time.time() - start_time
            metadata = {
                "strategy": "long_context",
                "documents_processed": len(initial_docs),
                "documents_returned": len(filtered_docs),
                "estimated_tokens": total_tokens,
                "context_window_size": context_window_size,
                "retrieval_time": retrieval_time
            }
            
            logger.info(f"Long-context retrieved {len(filtered_docs)} documents "
                       f"({total_tokens} tokens) in {retrieval_time:.3f}s")
            
            return filtered_docs, metadata
            
        except Exception as e:
            logger.error(f"Error in long-context retrieval: {str(e)}")
            return [], {"strategy": "long_context", "error": str(e)}
    
    def _filter_docs_for_context_window(self, documents: List[Document], 
                                      max_tokens: int) -> Tuple[List[Document], int]:
        """
        Filter documents to fit within a context window size.
        
        Args:
            documents: List of documents to filter
            max_tokens: Maximum number of tokens allowed
            
        Returns:
            Tuple[List[Document], int]: Filtered documents and total token count
        """
        filtered_docs = []
        total_tokens = 0
        
        for doc in documents:
            # Rough token estimation: ~4 characters per token
            doc_tokens = len(doc.content) // 4
            
            if total_tokens + doc_tokens <= max_tokens:
                filtered_docs.append(doc)
                total_tokens += doc_tokens
            else:
                # Try to fit partial document if there's remaining space
                remaining_tokens = max_tokens - total_tokens
                if remaining_tokens > 100:  # Only if significant space remains
                    # Truncate document content to fit
                    truncated_content = doc.content[:remaining_tokens * 4]
                    truncated_doc = Document(
                        content=truncated_content,
                        metadata={**doc.metadata, "truncated": True},
                        doc_id=doc.doc_id,
                        embedding=doc.embedding
                    )
                    filtered_docs.append(truncated_doc)
                    total_tokens = max_tokens
                break
        
        return filtered_docs, total_tokens
    
    def adaptive_retrieve(self, query: str, k: int = 5, 
                         strategy: str = "auto") -> Tuple[List[Document], Dict[str, Any]]:
        """
        Adaptively choose the best retrieval strategy based on query characteristics.
        
        Args:
            query: Query string to search for
            k: Number of documents to retrieve
            strategy: Retrieval strategy ("auto", "vector", "keyword", "hybrid", "long_context")
            
        Returns:
            Tuple[List[Document], Dict[str, Any]]: Retrieved documents and metadata
        """
        start_time = time.time()
        
        try:
            # Auto-select strategy based on query characteristics
            if strategy == "auto":
                strategy = self._select_optimal_strategy(query)
            
            metadata = {"selected_strategy": strategy, "query_length": len(query)}
            
            # Execute retrieval based on selected strategy
            if strategy == "hybrid":
                documents = self.hybrid_retrieve(query, k)
                metadata["retrieval_type"] = "hybrid"
                
            elif strategy == "long_context":
                documents, long_context_metadata = self.long_context_retrieve(query, k * 2)
                metadata.update(long_context_metadata)
                
            elif strategy == "rerank":
                documents = self.retrieve_with_rerank(query, k * 2, k)
                metadata["retrieval_type"] = "rerank"
                
            else:
                # Default to single retriever strategy
                documents = self.retrieve(query, k, strategy if strategy in self.retrievers else None)
                metadata["retrieval_type"] = "single"
            
            retrieval_time = time.time() - start_time
            metadata["retrieval_time"] = retrieval_time
            metadata["documents_returned"] = len(documents)
            
            logger.info(f"Adaptive retrieval using {strategy} returned {len(documents)} documents "
                       f"in {retrieval_time:.3f}s")
            
            return documents, metadata
            
        except Exception as e:
            logger.error(f"Error in adaptive retrieval: {str(e)}")
            return [], {"selected_strategy": strategy, "error": str(e)}
    
    def _select_optimal_strategy(self, query: str) -> str:
        """
        Select the optimal retrieval strategy based on query characteristics.
        
        Args:
            query: Query string to analyze
            
        Returns:
            str: Selected strategy name
        """
        query_lower = query.lower()
        query_length = len(query.split())
        
        # Long queries benefit from long-context approach
        if query_length > 20:
            return "long_context"
        
        # Queries with specific keywords benefit from hybrid approach
        keyword_indicators = ["specific", "exact", "precise", "find", "search", "locate"]
        if any(keyword in query_lower for keyword in keyword_indicators):
            return "hybrid"
        
        # Complex queries benefit from re-ranking
        complexity_indicators = ["compare", "analyze", "explain", "describe", "summarize"]
        if any(indicator in query_lower for indicator in complexity_indicators):
            return "rerank"
        
        # Default to vector search for semantic queries
        return "vector"
    
    def update_config(self, config: PipelineConfig) -> None:
        """
        Update the configuration for all components.
        
        Args:
            config: New pipeline configuration
        """
        self.config = config
        
        # Update all retrievers
        for retriever in self.retrievers.values():
            if hasattr(retriever, 'update_config'):
                retriever.update_config(config)
        
        # Update re-ranker
        if self.reranker is not None:
            self.reranker.set_config(config)
        
        # Update self-correction engine
        if self.self_correction_engine is not None:
            # Update thresholds
            self.self_correction_engine.update_thresholds(
                relevance_threshold=getattr(config, 'relevance_threshold', None),
                factuality_threshold=getattr(config, 'factuality_threshold', None),
                min_relevant_docs=getattr(config, 'min_relevant_docs', None)
            )
        
        # Re-setup re-ranker if configuration changed
        if config.use_reranking and self.reranker is None:
            self._setup_reranker()
        elif not config.use_reranking and self.reranker is not None:
            self.disable_reranking()
        
        # Re-setup self-correction if configuration changed
        if config.enable_self_correction and self.self_correction_engine is None:
            self._setup_self_correction()
        elif not config.enable_self_correction and self.self_correction_engine is not None:
            self.disable_self_correction()
        
        logger.info("RetrievalEngine configuration updated")