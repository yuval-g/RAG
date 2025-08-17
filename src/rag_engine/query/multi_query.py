"""
Multi-Query Generation for improved retrieval through query expansion
"""

from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from ..core.models import ProcessedQuery
from ..core.exceptions import QueryProcessingError


class MultiQueryGenerator:
    """
    Generates multiple alternative versions of a user query to improve retrieval.
    
    Based on the Multi-Query approach from workplan/01AdvancedQueryTransformations.md,
    this class uses an LLM to generate several different versions of the user's question,
    effectively searching from multiple angles to overcome limitations of distance-based
    similarity search.
    """
    
    def __init__(
        self,
        llm_model: str = "gemini-1.5-flash",
        temperature: float = 0.0,
        num_queries: int = 5,
        **llm_kwargs
    ):
        """
        Initialize the MultiQueryGenerator.
        
        Args:
            llm_model: The Google Gemini model to use
            temperature: Temperature for generation (0.0 for deterministic)
            num_queries: Number of alternative queries to generate
            **llm_kwargs: Additional arguments for the LLM
        """
        self.llm_model = llm_model
        self.temperature = temperature
        self.num_queries = num_queries
        self.llm_kwargs = llm_kwargs
        
        # Initialize the LLM
        self.llm = ChatGoogleGenerativeAI(
            model=llm_model,
            temperature=temperature,
            **llm_kwargs
        )
        
        # Create the prompt template based on workplan implementation
        self.prompt_template = self._create_prompt_template()
        
        # Build the generation chain
        self.generation_chain = (
            self.prompt_template
            | self.llm
            | StrOutputParser()
            | self._parse_queries
        )
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the prompt template for multi-query generation."""
        template = f"""You are an AI language model assistant. Your task is to generate {self.num_queries} 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {{question}}"""
        
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
        
        return cleaned_queries
    
    def generate_queries(self, question: str) -> List[str]:
        """
        Generate multiple alternative versions of the input question.
        
        Args:
            question: The original user question
            
        Returns:
            List of alternative queries including the original
            
        Raises:
            QueryProcessingError: If query generation fails
        """
        try:
            # Generate alternative queries
            alternative_queries = self.generation_chain.invoke({"question": question})
            
            # Validate queries
            validated_queries = self._validate_queries(alternative_queries, question)
            
            # Always include the original question
            all_queries = [question] + validated_queries
            
            return all_queries
            
        except Exception as e:
            raise QueryProcessingError(f"Failed to generate multi-queries: {str(e)}")
    
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
            # If no valid alternatives, create a simple rephrasing
            validated = [f"What is {original_question.lower()}?"]
        
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
    
    def process_query(self, question: str, **kwargs) -> ProcessedQuery:
        """
        Process a query and return a ProcessedQuery object.
        
        Args:
            question: The original user question
            **kwargs: Additional processing parameters
            
        Returns:
            ProcessedQuery object with generated alternatives
        """
        try:
            # Generate alternative queries
            transformed_queries = self.generate_queries(question)
            
            # Create metadata
            metadata = {
                "num_generated": len(transformed_queries) - 1,  # Exclude original
                "model_used": self.llm_model,
                "temperature": self.temperature,
                **kwargs
            }
            
            return ProcessedQuery(
                original_query=question,
                transformed_queries=transformed_queries,
                strategy_used="multi_query",
                metadata=metadata
            )
            
        except Exception as e:
            raise QueryProcessingError(f"Failed to process query with multi-query: {str(e)}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return {
            "llm_model": self.llm_model,
            "temperature": self.temperature,
            "num_queries": self.num_queries,
            "strategy": "multi_query"
        }