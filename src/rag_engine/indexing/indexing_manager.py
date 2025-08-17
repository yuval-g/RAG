"""
Unified indexing manager that coordinates all indexing strategies
"""

import logging
from typing import List, Dict, Any, Optional, Type
from enum import Enum

from ..core.interfaces import BaseIndexer
from ..core.models import Document
from ..core.config import PipelineConfig
from .basic_indexer import BasicIndexer
from .multi_representation_indexer import MultiRepresentationIndexer
from .colbert_indexer import ColBERTIndexer
from .raptor_indexer import RAPTORIndexer


logger = logging.getLogger(__name__)


class IndexingStrategy(Enum):
    """Available indexing strategies"""
    BASIC = "basic"
    MULTI_REPRESENTATION = "multi_representation"
    COLBERT = "colbert"
    RAPTOR = "raptor"


class IndexingManager:
    """
    Unified manager that coordinates all indexing strategies.
    Provides a single interface for document indexing with different approaches.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the IndexingManager with configuration.
        
        Args:
            config: Pipeline configuration containing indexing settings
        """
        self.config = config
        self._indexers: Dict[str, BaseIndexer] = {}
        self._active_strategy = config.indexing_strategy
        
        # Registry of available indexer classes
        self._indexer_registry: Dict[str, Type[BaseIndexer]] = {
            IndexingStrategy.BASIC.value: BasicIndexer,
            IndexingStrategy.MULTI_REPRESENTATION.value: MultiRepresentationIndexer,
            IndexingStrategy.COLBERT.value: ColBERTIndexer,
            IndexingStrategy.RAPTOR.value: RAPTORIndexer,
        }
        
        logger.info(f"IndexingManager initialized with active strategy: {self._active_strategy}")
    
    def get_indexer(self, strategy: str) -> BaseIndexer:
        """
        Get an indexer for the specified strategy, creating it if necessary.
        
        Args:
            strategy: Indexing strategy name
            
        Returns:
            BaseIndexer: The indexer instance for the strategy
            
        Raises:
            ValueError: If the strategy is not supported
        """
        if strategy not in self._indexer_registry:
            available_strategies = list(self._indexer_registry.keys())
            raise ValueError(f"Unsupported indexing strategy: {strategy}. "
                           f"Available strategies: {available_strategies}")
        
        # Create indexer if it doesn't exist
        if strategy not in self._indexers:
            indexer_class = self._indexer_registry[strategy]
            
            # Handle special initialization for different indexers
            if strategy == IndexingStrategy.COLBERT.value:
                # ColBERT indexer needs an index name
                index_name = f"colbert-{strategy}"
                self._indexers[strategy] = indexer_class(self.config, index_name=index_name)
            elif strategy == IndexingStrategy.RAPTOR.value:
                # RAPTOR indexer can have custom parameters
                max_levels = getattr(self.config, 'raptor_max_levels', 3)
                cluster_threshold = getattr(self.config, 'raptor_cluster_threshold', 5)
                self._indexers[strategy] = indexer_class(
                    self.config, 
                    max_levels=max_levels, 
                    cluster_size_threshold=cluster_threshold
                )
            else:
                # Basic and multi-representation indexers use standard initialization
                self._indexers[strategy] = indexer_class(self.config)
            
            logger.info(f"Created {strategy} indexer")
        
        return self._indexers[strategy]
    
    def index_documents(self, documents: List[Document], strategy: Optional[str] = None) -> bool:
        """
        Index documents using the specified or active strategy.
        
        Args:
            documents: List of Document objects to index
            strategy: Indexing strategy to use (defaults to active strategy)
            
        Returns:
            bool: True if indexing was successful, False otherwise
        """
        if not documents:
            logger.warning("No documents provided for indexing")
            return True
        
        strategy = strategy or self._active_strategy
        
        try:
            indexer = self.get_indexer(strategy)
            logger.info(f"Indexing {len(documents)} documents using {strategy} strategy")
            
            result = indexer.index_documents(documents)
            
            if result:
                logger.info(f"Successfully indexed {len(documents)} documents with {strategy}")
            else:
                logger.error(f"Failed to index documents with {strategy}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error indexing documents with {strategy}: {str(e)}")
            return False
    
    def get_document_count(self, strategy: Optional[str] = None) -> int:
        """
        Get the number of indexed documents for a strategy.
        
        Args:
            strategy: Indexing strategy (defaults to active strategy)
            
        Returns:
            int: Number of documents indexed
        """
        strategy = strategy or self._active_strategy
        
        try:
            if strategy in self._indexers:
                return self._indexers[strategy].get_document_count()
            else:
                return 0
        except Exception as e:
            logger.error(f"Error getting document count for {strategy}: {str(e)}")
            return 0
    
    def clear_index(self, strategy: Optional[str] = None) -> bool:
        """
        Clear the index for a specific strategy.
        
        Args:
            strategy: Indexing strategy to clear (defaults to active strategy)
            
        Returns:
            bool: True if clearing was successful, False otherwise
        """
        strategy = strategy or self._active_strategy
        
        try:
            if strategy in self._indexers:
                result = self._indexers[strategy].clear_index()
                if result:
                    logger.info(f"Cleared index for {strategy}")
                else:
                    logger.error(f"Failed to clear index for {strategy}")
                return result
            else:
                logger.info(f"No index to clear for {strategy}")
                return True
                
        except Exception as e:
            logger.error(f"Error clearing index for {strategy}: {str(e)}")
            return False
    
    def clear_all_indexes(self) -> bool:
        """
        Clear all indexes for all strategies.
        
        Returns:
            bool: True if all indexes were cleared successfully, False otherwise
        """
        success = True
        
        for strategy in list(self._indexers.keys()):
            try:
                result = self.clear_index(strategy)
                if not result:
                    success = False
            except Exception as e:
                logger.error(f"Error clearing index for {strategy}: {str(e)}")
                success = False
        
        if success:
            logger.info("All indexes cleared successfully")
        else:
            logger.warning("Some indexes failed to clear")
        
        return success
    
    def set_active_strategy(self, strategy: str) -> bool:
        """
        Set the active indexing strategy.
        
        Args:
            strategy: Strategy name to set as active
            
        Returns:
            bool: True if strategy was set successfully, False otherwise
        """
        if strategy not in self._indexer_registry:
            available_strategies = list(self._indexer_registry.keys())
            logger.error(f"Invalid strategy: {strategy}. Available: {available_strategies}")
            return False
        
        self._active_strategy = strategy
        logger.info(f"Active indexing strategy set to: {strategy}")
        return True
    
    def get_active_strategy(self) -> str:
        """
        Get the currently active indexing strategy.
        
        Returns:
            str: Active strategy name
        """
        return self._active_strategy
    
    def list_strategies(self) -> List[str]:
        """
        Get list of available indexing strategies.
        
        Returns:
            List[str]: List of available strategy names
        """
        return list(self._indexer_registry.keys())
    
    def get_strategy_info(self, strategy: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a specific indexing strategy.
        
        Args:
            strategy: Strategy to get info for (defaults to active strategy)
            
        Returns:
            Dict[str, Any]: Strategy information
        """
        strategy = strategy or self._active_strategy
        
        indexer_class = self._indexer_registry.get(strategy)
        class_name = indexer_class.__name__ if indexer_class else "Unknown"
        
        info = {
            "strategy": strategy,
            "is_active": strategy == self._active_strategy,
            "is_initialized": strategy in self._indexers,
            "document_count": 0,
            "indexer_class": class_name
        }
        
        if strategy in self._indexers:
            try:
                info["document_count"] = self._indexers[strategy].get_document_count()
                
                # Add strategy-specific information
                indexer = self._indexers[strategy]
                
                if strategy == IndexingStrategy.BASIC.value:
                    if hasattr(indexer, 'get_chunk_count'):
                        info["chunk_count"] = indexer.get_chunk_count()
                
                elif strategy == IndexingStrategy.MULTI_REPRESENTATION.value:
                    if hasattr(indexer, 'get_summary_count'):
                        info["summary_count"] = indexer.get_summary_count()
                    if hasattr(indexer, 'get_docstore_count'):
                        info["docstore_count"] = indexer.get_docstore_count()
                
                elif strategy == IndexingStrategy.COLBERT.value:
                    if hasattr(indexer, 'get_index_info'):
                        colbert_info = indexer.get_index_info()
                        info.update(colbert_info)
                
                elif strategy == IndexingStrategy.RAPTOR.value:
                    if hasattr(indexer, 'get_tree_info'):
                        tree_info = indexer.get_tree_info()
                        info.update(tree_info)
                
            except Exception as e:
                logger.error(f"Error getting info for {strategy}: {str(e)}")
                info["error"] = str(e)
        
        return info
    
    def get_all_strategies_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all available indexing strategies.
        
        Returns:
            Dict[str, Dict[str, Any]]: Information for all strategies
        """
        all_info = {}
        
        for strategy in self.list_strategies():
            all_info[strategy] = self.get_strategy_info(strategy)
        
        return all_info
    
    def is_strategy_ready(self, strategy: str) -> bool:
        """
        Check if a strategy is ready for use.
        
        Args:
            strategy: Strategy name to check
            
        Returns:
            bool: True if strategy is ready, False otherwise
        """
        if strategy not in self._indexer_registry:
            return False
        
        try:
            indexer = self.get_indexer(strategy)
            
            # Check if indexer has a ready method
            if hasattr(indexer, 'is_ready'):
                return indexer.is_ready()
            
            # For strategies without is_ready method, assume ready if initialized
            return True
            
        except Exception as e:
            logger.error(f"Error checking readiness for {strategy}: {str(e)}")
            return False
    
    def get_indexer_for_retrieval(self, strategy: Optional[str] = None):
        """
        Get the appropriate retriever/vectorstore for a strategy.
        
        Args:
            strategy: Strategy to get retriever for (defaults to active strategy)
            
        Returns:
            Retriever object or None if not available
        """
        strategy = strategy or self._active_strategy
        
        try:
            if strategy not in self._indexers:
                return None
            
            indexer = self._indexers[strategy]
            
            # Return strategy-specific retriever
            if strategy == IndexingStrategy.BASIC.value:
                if hasattr(indexer, 'get_vectorstore'):
                    return indexer.get_vectorstore()
            
            elif strategy == IndexingStrategy.MULTI_REPRESENTATION.value:
                if hasattr(indexer, 'get_retriever'):
                    return indexer.get_retriever()
            
            elif strategy == IndexingStrategy.COLBERT.value:
                if hasattr(indexer, 'as_langchain_retriever'):
                    return indexer.as_langchain_retriever()
            
            elif strategy == IndexingStrategy.RAPTOR.value:
                # RAPTOR doesn't have a single retriever, return the indexer itself
                return indexer
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting retriever for {strategy}: {str(e)}")
            return None