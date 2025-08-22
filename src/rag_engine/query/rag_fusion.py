"""
RAG-Fusion with Reciprocal Rank Fusion for intelligent document re-ranking
"""

from typing import List, Dict, Any, Tuple, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI


from ..core.models import Document, ProcessedQuery
from ..core.exceptions import QueryProcessingError
from ..common.utils import get_llm


class RAGFusionProcessor:
    """
    Implements RAG-Fusion with Reciprocal Rank Fusion (RRF) for improved retrieval.
    
    Based on the RAG-Fusion approach from workplan/01AdvancedQueryTransformations.md,
    this class generates multiple queries and then re-ranks documents using RRF,
    which intelligently combines results from multiple searches by boosting documents
    that appear consistently high across different result lists.
    """
    
    def __init__(
        self,
        llm_model: str = "gemini-2.0-flash-lite",
        temperature: float = 0.0,
        num_queries: int = 4,
        rrf_k: int = 60,
        **llm_kwargs
    ):
        """
        Initialize the RAGFusionProcessor.
        
        Args:
            llm_model: The Google Gemini model to use
            temperature: Temperature for generation (0.0 for deterministic)
            num_queries: Number of queries to generate for fusion
            rrf_k: RRF parameter (higher values reduce the impact of rank differences)
            **llm_kwargs: Additional arguments for the LLM
        """
        self.llm_model = llm_model
        self.temperature = temperature
        self.num_queries = num_queries
        self.rrf_k = rrf_k
        self.llm_kwargs = llm_kwargs

        # Initialize the LLM
        self.llm = get_llm(llm_model, temperature, **llm_kwargs)
        
        # Create the prompt template for RAG-Fusion
        self.prompt_template = self._create_prompt_template()
        
        # Build the generation chain
        self.generation_chain = (
            self.prompt_template
            | self.llm
            | StrOutputParser()
            | self._parse_queries
        )
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the prompt template for RAG-Fusion query generation."""
        template = f"""You are a helpful assistant that generates multiple search queries based on a single input query.
Generate multiple search queries related to: {{question}}
Output ({self.num_queries} queries):"""
        
        return ChatPromptTemplate.from_template(template)
    
    def _parse_queries(self, output: str) -> List[str]:
        """
        Parse the LLM output into a list of queries.
        
        Args:
            output: Raw output from the LLM
            
        Returns:
            List of parsed queries
        """
        # Split by newlines and clean up
        queries = [q.strip() for q in output.split("\n") if q.strip()]
        
        # Remove numbering if present (e.g., "1. ", "2. ")
        cleaned_queries = []
        for query in queries:
            # Remove leading numbers and dots
            import re
            cleaned_query = re.sub(r'^\d+\.\s*', '', query)
            if cleaned_query:
                cleaned_queries.append(cleaned_query)
        
        return cleaned_queries[:self.num_queries]  # Limit to requested number
    
    def reciprocal_rank_fusion(
        self, 
        results: List[List[Document]], 
        k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """
        Reciprocal Rank Fusion that intelligently combines multiple ranked lists.
        
        Based on the RRF implementation from workplan/01AdvancedQueryTransformations.md,
        this function re-ranks documents by giving higher scores to documents that
        appear consistently high across different result lists.
        
        Args:
            results: List of document lists from different queries
            k: RRF parameter (uses instance default if None)
            
        Returns:
            List of (document, score) tuples sorted by fused score
        """
        if k is None:
            k = self.rrf_k
            
        fused_scores = {}
        doc_map = {}  # Map from doc_key to actual document
        
        # Iterate through each list of ranked documents
        for docs in results:
            for rank, doc in enumerate(docs):
                # Create a unique key for the document using content and metadata
                doc_key = self._create_doc_key(doc)
                
                if doc_key not in fused_scores:
                    fused_scores[doc_key] = 0
                    doc_map[doc_key] = doc
                    
                # The core of RRF: documents ranked higher (lower rank value) get a larger score
                fused_scores[doc_key] += 1 / (rank + k)
        
        # Sort documents by their new fused scores in descending order
        reranked_results = [
            (doc_map[doc_key], score)
            for doc_key, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return reranked_results
    
    def _create_doc_key(self, doc: Document) -> str:
        """
        Create a unique key for a document based on content and metadata.
        
        Args:
            doc: Document to create key for
            
        Returns:
            Unique string key for the document
        """
        # Use content hash and doc_id if available, otherwise use content + metadata
        import hashlib
        
        if doc.doc_id:
            return f"doc_id:{doc.doc_id}"
        
        # Create hash from content and key metadata
        content_hash = hashlib.md5(doc.content.encode()).hexdigest()
        metadata_str = str(sorted(doc.metadata.items())) if doc.metadata else ""
        metadata_hash = hashlib.md5(metadata_str.encode()).hexdigest()
        
        return f"content:{content_hash}:meta:{metadata_hash}"
    
    def generate_fusion_queries(self, question: str) -> List[str]:
        """
        Generate multiple queries for RAG-Fusion.
        
        Args:
            question: The original user question
            
        Returns:
            List of queries for fusion (including original)
            
        Raises:
            QueryProcessingError: If query generation fails
        """
        try:
            # Generate alternative queries
            alternative_queries = self.generation_chain.invoke({"question": question})
            
            # Validate and filter queries
            validated_queries = self._validate_queries(alternative_queries, question)
            
            # Always include the original question first
            all_queries = [question] + validated_queries
            
            return all_queries
            
        except Exception as e:
            raise QueryProcessingError(f"Failed to generate fusion queries: {str(e)}")
    
    def _validate_queries(self, queries: List[str], original_question: str) -> List[str]:
        """
        Validate and filter generated queries.
        
        Args:
            queries: List of generated queries
            original_question: The original question for comparison
            
        Returns:
            List of validated queries
        """
        validated = []
        
        for query in queries:
            # Skip empty queries
            if not query or len(query.strip()) < 5:
                continue
                
            # Skip queries that are too similar to the original
            if self._is_too_similar(query, original_question):
                continue
                
            # Skip queries that are too long (likely hallucinated)
            if len(query) > 500:
                continue
                
            validated.append(query)
        
        # Ensure we have at least some alternative queries
        if not validated:
            # If no valid alternatives, create simple variations
            validated = [
                f"How to {original_question.lower()}?",
                f"Explain {original_question.lower()}"
            ]
        
        return validated[:self.num_queries - 1]  # Reserve space for original
    
    def _is_too_similar(self, query1: str, query2: str, threshold: float = 0.8) -> bool:
        """
        Check if two queries are too similar using simple word overlap.
        
        Args:
            query1: First query
            query2: Second query
            threshold: Similarity threshold (0-1)
            
        Returns:
            True if queries are too similar
        """
        # Simple word-based similarity check
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        if not words1 or not words2:
            return False
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0
        return similarity > threshold
    
    def process_with_retriever(
        self, 
        question: str, 
        retriever_func, 
        top_k: int = 10
    ) -> List[Tuple[Document, float]]:
        """
        Process a query with RAG-Fusion using a provided retriever function.
        
        Args:
            question: The original user question
            retriever_func: Function that takes a query and returns List[Document]
            top_k: Number of top documents to return after fusion
            
        Returns:
            List of (document, score) tuples re-ranked by RRF
            
        Raises:
            QueryProcessingError: If processing fails
        """
        try:
            # Generate fusion queries
            fusion_queries = self.generate_fusion_queries(question)
            
            # Retrieve documents for each query
            all_results = []
            for query in fusion_queries:
                docs = retriever_func(query)
                all_results.append(docs)
            
            # Apply reciprocal rank fusion
            fused_results = self.reciprocal_rank_fusion(all_results)
            
            # Return top-k results
            return fused_results[:top_k]
            
        except Exception as e:
            raise QueryProcessingError(f"Failed to process query with RAG-Fusion: {str(e)}")
    
    def process_query(self, question: str, **kwargs) -> ProcessedQuery:
        """
        Process a query and return a ProcessedQuery object with fusion queries.
        
        Args:
            question: The original user question
            **kwargs: Additional processing parameters
            
        Returns:
            ProcessedQuery object with fusion queries
        """
        try:
            # Generate fusion queries
            fusion_queries = self.generate_fusion_queries(question)
            
            # Create metadata
            metadata = {
                "num_fusion_queries": len(fusion_queries),
                "rrf_k": self.rrf_k,
                "model_used": self.llm_model,
                "temperature": self.temperature,
                **kwargs
            }
            
            return ProcessedQuery(
                original_query=question,
                transformed_queries=fusion_queries,
                strategy_used="rag_fusion",
                metadata=metadata
            )
            
        except Exception as e:
            raise QueryProcessingError(f"Failed to process query with RAG-Fusion: {str(e)}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return {
            "llm_model": self.llm_model,
            "temperature": self.temperature,
            "num_queries": self.num_queries,
            "rrf_k": self.rrf_k,
            "strategy": "rag_fusion"
        }