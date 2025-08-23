"""
Unified Query Processor that coordinates all transformation strategies
"""

from typing import List, Dict, Any, Optional, Callable, Union
from enum import Enum

from ..core.models import Document, ProcessedQuery
from ..core.exceptions import QueryProcessingError
from ..core.interfaces import BaseQueryProcessor

from .multi_query import MultiQueryGenerator
from .rag_fusion import RAGFusionProcessor
from .decomposition import QueryDecomposer
from .step_back import StepBackProcessor
from .hyde import HyDEProcessor


class QueryStrategy(Enum):
    """Available query processing strategies"""
    MULTI_QUERY = "multi_query"
    RAG_FUSION = "rag_fusion"
    DECOMPOSITION = "decomposition"
    STEP_BACK = "step_back"
    HYDE = "hyde"
    BASIC = "basic"


class QueryProcessor(BaseQueryProcessor):
    """
    Unified Query Processor that coordinates all transformation strategies.
    
    This class provides a single interface to access all query transformation
    techniques including multi-query generation, RAG-Fusion, decomposition,
    step-back prompting, and HyDE. It supports strategy selection, configuration
    management, and integration testing for all query processing techniques.
    """
    
    def __init__(
        self,
        llm_model: str = "gemini-2.0-flash-lite",
        temperature: float = 0.0,
        default_strategy: str = "multi_query",
        **llm_kwargs
    ):
        """
        Initialize the QueryProcessor with all transformation strategies.
        
        Args:
            llm_model: The Google Gemini model to use for all processors
            temperature: Temperature for generation (0.0 for deterministic)
            default_strategy: Default strategy to use when none specified
            **llm_kwargs: Additional arguments for the LLM
        """
        # Handle case where llm_model is actually a PipelineConfig object
        self.llm_model = llm_model
        self.temperature = temperature
        
        self.default_strategy = default_strategy
        self.llm_kwargs = llm_kwargs
        
        # Initialize all processors
        self._initialize_processors()
        
        # Strategy configuration
        self.strategy_configs = {}
    
    def _initialize_processors(self):
        """Initialize all query processing strategies."""
        try:
            # Multi-Query Generator
            self.multi_query_generator = MultiQueryGenerator(
                llm_model=self.llm_model,
                temperature=self.temperature,
                **self.llm_kwargs
            )
            
            # RAG-Fusion Processor
            self.rag_fusion_processor = RAGFusionProcessor(
                llm_model=self.llm_model,
                temperature=self.temperature,
                **self.llm_kwargs
            )
            
            # Query Decomposer
            self.query_decomposer = QueryDecomposer(
                llm_model=self.llm_model,
                temperature=self.temperature,
                **self.llm_kwargs
            )
            
            # Step-Back Processor
            self.step_back_processor = StepBackProcessor(
                llm_model=self.llm_model,
                temperature=self.temperature,
                **self.llm_kwargs
            )
            
            # HyDE Processor
            self.hyde_processor = HyDEProcessor(
                llm_model=self.llm_model,
                temperature=self.temperature,
                **self.llm_kwargs
            )
            
        except Exception as e:
            raise QueryProcessingError(f"Failed to initialize query processors: {str(e)}")
    
    def process(self, query: str, strategy: str = "basic") -> ProcessedQuery:
        """
        Process a query using the specified strategy.
        
        Args:
            query: The input query to process
            strategy: The processing strategy to use
            
        Returns:
            ProcessedQuery object with transformed queries
            
        Raises:
            QueryProcessingError: If processing fails
        """
        try:
            # Handle basic strategy first
            if strategy == "basic":
                return ProcessedQuery(
                    original_query=query,
                    transformed_queries=[query],
                    strategy_used="basic",
                    metadata={"processor": "unified"}
                )
            
            # Use default strategy if basic was requested but we want to use default
            if strategy == "default":
                strategy = self.default_strategy
            
            # Validate strategy
            if not self._is_valid_strategy(strategy):
                raise QueryProcessingError(f"Invalid strategy: {strategy}")
            
            # Get strategy configuration
            config = self.strategy_configs.get(strategy, {})
            
            # Process based on strategy
            if strategy == QueryStrategy.MULTI_QUERY.value:
                return self.multi_query_generator.process_query(query, **config)
            
            elif strategy == QueryStrategy.RAG_FUSION.value:
                return self.rag_fusion_processor.process_query(query, **config)
            
            elif strategy == QueryStrategy.DECOMPOSITION.value:
                return self.query_decomposer.process_query(query, **config)
            
            elif strategy == QueryStrategy.STEP_BACK.value:
                return self.step_back_processor.process_query(query, **config)
            
            elif strategy == QueryStrategy.HYDE.value:
                return self.hyde_processor.process_query(query, **config)
            
            else:
                # Fallback to basic processing
                return ProcessedQuery(
                    original_query=query,
                    transformed_queries=[query],
                    strategy_used="basic",
                    metadata={"processor": "unified"}
                )
                
        except Exception as e:
            raise QueryProcessingError(f"Failed to process query with strategy '{strategy}': {str(e)}")
    
    def get_available_strategies(self) -> List[str]:
        """
        Get list of available processing strategies.
        
        Returns:
            List of strategy names
        """
        return [strategy.value for strategy in QueryStrategy]
    
    def _is_valid_strategy(self, strategy: str) -> bool:
        """
        Check if a strategy is valid.
        
        Args:
            strategy: Strategy name to validate
            
        Returns:
            True if strategy is valid
        """
        return strategy in self.get_available_strategies()
    
    def configure_strategy(self, strategy: str, **config) -> None:
        """
        Configure a specific strategy with custom parameters.
        
        Args:
            strategy: Strategy name to configure
            **config: Configuration parameters
            
        Raises:
            QueryProcessingError: If strategy is invalid
        """
        if not self._is_valid_strategy(strategy):
            raise QueryProcessingError(f"Invalid strategy: {strategy}")
        
        self.strategy_configs[strategy] = config
    
    def multi_query(self, query: str, **kwargs) -> List[str]:
        """
        Generate multiple query perspectives.
        
        Args:
            query: Input query
            **kwargs: Additional parameters
            
        Returns:
            List of alternative queries
        """
        try:
            return self.multi_query_generator.generate_queries(query)
        except Exception as e:
            raise QueryProcessingError(f"Multi-query generation failed: {str(e)}")
    
    def decompose(self, query: str, **kwargs) -> List[str]:
        """
        Decompose a complex query into sub-questions.
        
        Args:
            query: Input query to decompose
            **kwargs: Additional parameters
            
        Returns:
            List of sub-questions
        """
        try:
            return self.query_decomposer.decompose_query(query)
        except Exception as e:
            raise QueryProcessingError(f"Query decomposition failed: {str(e)}")
    
    def step_back(self, query: str, **kwargs) -> str:
        """
        Generate a step-back (more general) version of the query.
        
        Args:
            query: Input query
            **kwargs: Additional parameters
            
        Returns:
            Step-back question
        """
        try:
            return self.step_back_processor.generate_step_back_question(query)
        except Exception as e:
            raise QueryProcessingError(f"Step-back generation failed: {str(e)}")
    
    def hyde(self, query: str, **kwargs) -> str:
        """
        Generate a hypothetical document for the query.
        
        Args:
            query: Input query
            **kwargs: Additional parameters
            
        Returns:
            Hypothetical document content
        """
        try:
            return self.hyde_processor.generate_hypothetical_document(query)
        except Exception as e:
            raise QueryProcessingError(f"HyDE generation failed: {str(e)}")
    
    def process_with_retriever(
        self,
        query: str,
        strategy: str,
        retriever_func: Callable[[str], List[Document]],
        **kwargs
    ) -> Union[List[Document], List[tuple], str]:
        """
        Process a query with a retriever function using the specified strategy.
        
        Args:
            query: Input query
            strategy: Processing strategy
            retriever_func: Function that retrieves documents
            **kwargs: Additional parameters
            
        Returns:
            Strategy-specific results (documents, ranked documents, or response)
            
        Raises:
            QueryProcessingError: If processing fails
        """
        try:
            if strategy == QueryStrategy.RAG_FUSION.value:
                return self.rag_fusion_processor.process_with_retriever(
                    query, retriever_func, **kwargs
                )
            
            elif strategy == QueryStrategy.STEP_BACK.value:
                return self.step_back_processor.process_with_retriever(
                    query, retriever_func, **kwargs
                )
            
            elif strategy == QueryStrategy.HYDE.value:
                return self.hyde_processor.process_with_retriever(
                    query, retriever_func, **kwargs
                )
            
            elif strategy == QueryStrategy.DECOMPOSITION.value:
                # For decomposition, we need an answerer function too
                def simple_answerer(question: str, documents: List[Document]) -> str:
                    if not documents:
                        return "No relevant information found."
                    return f"Based on the available documents: {documents[0].content[:200]}..."
                
                return self.query_decomposer.process_query_with_decomposition(
                    query, retriever_func, simple_answerer
                )
            
            elif strategy == QueryStrategy.MULTI_QUERY.value:
                # For multi-query, retrieve with all queries and combine
                queries = self.multi_query_generator.generate_queries(query)
                all_docs = []
                for q in queries:
                    docs = retriever_func(q)
                    all_docs.extend(docs)
                
                # Remove duplicates based on content
                unique_docs = []
                seen_content = set()
                for doc in all_docs:
                    if doc.content not in seen_content:
                        unique_docs.append(doc)
                        seen_content.add(doc.content)
                
                return unique_docs[:kwargs.get('top_k', 10)]
            
            else:
                # Basic retrieval
                return retriever_func(query)
                
        except Exception as e:
            raise QueryProcessingError(f"Failed to process query with retriever using strategy '{strategy}': {str(e)}")
    
    def get_processor_config(self, strategy: str) -> Dict[str, Any]:
        """
        Get configuration for a specific processor.
        
        Args:
            strategy: Strategy name
            
        Returns:
            Configuration dictionary
            
        Raises:
            QueryProcessingError: If strategy is invalid
        """
        if not self._is_valid_strategy(strategy):
            raise QueryProcessingError(f"Invalid strategy: {strategy}")
        
        try:
            if strategy == QueryStrategy.MULTI_QUERY.value:
                return self.multi_query_generator.get_config()
            elif strategy == QueryStrategy.RAG_FUSION.value:
                return self.rag_fusion_processor.get_config()
            elif strategy == QueryStrategy.DECOMPOSITION.value:
                return self.query_decomposer.get_config()
            elif strategy == QueryStrategy.STEP_BACK.value:
                return self.step_back_processor.get_config()
            elif strategy == QueryStrategy.HYDE.value:
                return self.hyde_processor.get_config()
            else:
                return {"strategy": strategy, "processor": "unified"}
                
        except Exception as e:
            raise QueryProcessingError(f"Failed to get config for strategy '{strategy}': {str(e)}")
    
    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get configurations for all processors.
        
        Returns:
            Dictionary mapping strategy names to their configurations
        """
        configs = {}
        
        for strategy in self.get_available_strategies():
            try:
                configs[strategy] = self.get_processor_config(strategy)
            except Exception:
                configs[strategy] = {"error": "Failed to get config"}
        
        return configs
    
    def set_default_strategy(self, strategy: str) -> None:
        """
        Set the default processing strategy.
        
        Args:
            strategy: Strategy name to set as default
            
        Raises:
            QueryProcessingError: If strategy is invalid
        """
        if not self._is_valid_strategy(strategy):
            raise QueryProcessingError(f"Invalid strategy: {strategy}")
        
        self.default_strategy = strategy
    
    def get_strategy_info(self, strategy: str) -> Dict[str, Any]:
        """
        Get detailed information about a strategy.
        
        Args:
            strategy: Strategy name
            
        Returns:
            Dictionary with strategy information
            
        Raises:
            QueryProcessingError: If strategy is invalid
        """
        if not self._is_valid_strategy(strategy):
            raise QueryProcessingError(f"Invalid strategy: {strategy}")
        
        info = {
            "name": strategy,
            "description": self._get_strategy_description(strategy),
            "config": self.get_processor_config(strategy),
            "custom_config": self.strategy_configs.get(strategy, {}),
            "supports_retriever": self._supports_retriever(strategy)
        }
        
        return info
    
    def _get_strategy_description(self, strategy: str) -> str:
        """Get description for a strategy."""
        descriptions = {
            QueryStrategy.MULTI_QUERY.value: "Generates multiple alternative versions of the query for broader retrieval",
            QueryStrategy.RAG_FUSION.value: "Combines multiple queries with Reciprocal Rank Fusion for intelligent re-ranking",
            QueryStrategy.DECOMPOSITION.value: "Breaks complex queries into simpler sub-questions for systematic answering",
            QueryStrategy.STEP_BACK.value: "Generates broader, more general questions to provide richer context",
            QueryStrategy.HYDE.value: "Creates hypothetical documents to improve semantic matching during retrieval",
            QueryStrategy.BASIC.value: "Basic query processing without transformation"
        }
        
        return descriptions.get(strategy, "Unknown strategy")
    
    def _supports_retriever(self, strategy: str) -> bool:
        """Check if strategy supports retriever integration."""
        retriever_strategies = {
            QueryStrategy.RAG_FUSION.value,
            QueryStrategy.STEP_BACK.value,
            QueryStrategy.HYDE.value,
            QueryStrategy.DECOMPOSITION.value,
            QueryStrategy.MULTI_QUERY.value
        }
        
        return strategy in retriever_strategies