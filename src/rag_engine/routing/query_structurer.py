"""
Query structuring implementation for converting natural language to structured queries
"""

import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from ..core.models import RoutingDecision


class TutorialSearch(BaseModel):
    """A data model for searching over a database of tutorial videos."""
    
    # The main query for a similarity search over the video's transcript.
    content_search: str = Field(
        ..., 
        description="Similarity search query applied to video transcripts."
    )
    
    # A more succinct query for searching just the video's title.
    title_search: str = Field(
        ..., 
        description="Alternate version of the content search query to apply to video titles."
    )
    
    # Optional metadata filters
    min_view_count: Optional[int] = Field(
        None, 
        description="Minimum view count filter, inclusive."
    )
    max_view_count: Optional[int] = Field(
        None, 
        description="Maximum view count filter, exclusive."
    )
    earliest_publish_date: Optional[datetime.date] = Field(
        None, 
        description="Earliest publish date filter, inclusive."
    )
    latest_publish_date: Optional[datetime.date] = Field(
        None, 
        description="Latest publish date filter, exclusive."
    )
    min_length_sec: Optional[int] = Field(
        None, 
        description="Minimum video length in seconds, inclusive."
    )
    max_length_sec: Optional[int] = Field(
        None, 
        description="Maximum video length in seconds, exclusive."
    )
    
    def pretty_print(self) -> None:
        """A helper function to print the populated fields of the model."""
        for field in self.__class__.model_fields:
            value = getattr(self, field)
            if value is not None:
                print(f"{field}: {value}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values"""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class DocumentSearch(BaseModel):
    """A data model for searching over a database of documents."""
    
    content_search: str = Field(
        ..., 
        description="Main search query for document content."
    )
    
    title_search: str = Field(
        ..., 
        description="Search query for document titles."
    )
    
    # Document metadata filters
    author: Optional[str] = Field(
        None, 
        description="Filter by document author."
    )
    category: Optional[str] = Field(
        None, 
        description="Filter by document category."
    )
    min_word_count: Optional[int] = Field(
        None, 
        description="Minimum word count filter, inclusive."
    )
    max_word_count: Optional[int] = Field(
        None, 
        description="Maximum word count filter, exclusive."
    )
    earliest_created_date: Optional[datetime.date] = Field(
        None, 
        description="Earliest creation date filter, inclusive."
    )
    latest_created_date: Optional[datetime.date] = Field(
        None, 
        description="Latest creation date filter, exclusive."
    )
    tags: Optional[List[str]] = Field(
        None, 
        description="Filter by document tags."
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values"""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class QueryStructurer:
    """
    Query structurer that converts natural language questions into structured queries
    that can leverage metadata filters for precise retrieval.
    """
    
    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.0
    ):
        """
        Initialize the query structurer.
        
        Args:
            model_name: Name of the Google Gemini model to use
            temperature: Temperature for LLM generation
        """
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature
        )
        
        # Build the query analyzer chains
        self._build_analyzers()
    
    def _build_analyzers(self):
        """Build the query analyzer chains for different search types"""
        
        # Tutorial/Video search analyzer
        tutorial_system_prompt = """You are an expert at converting user questions into database queries. \
You have access to a database of tutorial videos about a software library for building LLM-powered applications. \
Given a question, return a database query optimized to retrieve the most relevant results.

If there are acronyms or words you are not familiar with, do not try to rephrase them."""
        
        tutorial_prompt = ChatPromptTemplate.from_messages([
            ("system", tutorial_system_prompt), 
            ("human", "{question}")
        ])
        
        tutorial_structured_llm = self.llm.with_structured_output(TutorialSearch)
        self.tutorial_analyzer = tutorial_prompt | tutorial_structured_llm
        
        # Document search analyzer
        document_system_prompt = """You are an expert at converting user questions into database queries. \
You have access to a database of documents and articles. \
Given a question, return a database query optimized to retrieve the most relevant results.

Pay attention to any metadata filters the user mentions like author, category, date ranges, or document length."""
        
        document_prompt = ChatPromptTemplate.from_messages([
            ("system", document_system_prompt), 
            ("human", "{question}")
        ])
        
        document_structured_llm = self.llm.with_structured_output(DocumentSearch)
        self.document_analyzer = document_prompt | document_structured_llm
    
    def structure_tutorial_query(self, question: str) -> TutorialSearch:
        """
        Convert a natural language question into a structured tutorial/video search query.
        
        Args:
            question: Natural language question
            
        Returns:
            TutorialSearch: Structured search query
        """
        try:
            result = self.tutorial_analyzer.invoke({"question": question})
            return result
        except Exception as e:
            # Return a basic search query if structuring fails
            return TutorialSearch(
                content_search=question,
                title_search=question
            )
    
    def structure_document_query(self, question: str) -> DocumentSearch:
        """
        Convert a natural language question into a structured document search query.
        
        Args:
            question: Natural language question
            
        Returns:
            DocumentSearch: Structured search query
        """
        try:
            result = self.document_analyzer.invoke({"question": question})
            return result
        except Exception as e:
            # Return a basic search query if structuring fails
            return DocumentSearch(
                content_search=question,
                title_search=question
            )
    
    def structure_query(
        self, 
        question: str, 
        search_type: str = "tutorial"
    ) -> Dict[str, Any]:
        """
        Convert a natural language question into a structured query.
        
        Args:
            question: Natural language question
            search_type: Type of search ("tutorial" or "document")
            
        Returns:
            Dict[str, Any]: Structured query as dictionary
        """
        if search_type.lower() == "tutorial":
            structured_query = self.structure_tutorial_query(question)
        elif search_type.lower() == "document":
            structured_query = self.structure_document_query(question)
        else:
            raise ValueError(f"Unsupported search type: {search_type}")
        
        return structured_query.to_dict()
    
    def analyze_query_complexity(self, question: str) -> Dict[str, Any]:
        """
        Analyze the complexity of a query and suggest the best structuring approach.
        
        Args:
            question: Natural language question
            
        Returns:
            Dict[str, Any]: Analysis results with recommendations
        """
        # Simple heuristics to determine query complexity
        analysis = {
            "has_date_filter": any(word in question.lower() for word in [
                "published", "created", "after", "before", "in 2023", "in 2024", 
                "recent", "latest", "old", "new"
            ]),
            "has_length_filter": any(word in question.lower() for word in [
                "short", "long", "under", "over", "minutes", "seconds", "brief"
            ]),
            "has_view_filter": any(word in question.lower() for word in [
                "popular", "views", "watched", "trending"
            ]),
            "has_author_filter": any(word in question.lower() for word in [
                "by", "author", "written by", "created by"
            ]),
            "has_category_filter": any(word in question.lower() for word in [
                "category", "type", "about", "on"
            ]),
            "complexity_score": 0
        }
        
        # Calculate complexity score
        analysis["complexity_score"] = sum([
            analysis["has_date_filter"],
            analysis["has_length_filter"], 
            analysis["has_view_filter"],
            analysis["has_author_filter"],
            analysis["has_category_filter"]
        ])
        
        # Recommend search type based on content
        if any(word in question.lower() for word in [
            "video", "tutorial", "watch", "minutes", "seconds", "views"
        ]):
            analysis["recommended_search_type"] = "tutorial"
        else:
            analysis["recommended_search_type"] = "document"
        
        return analysis
    
    def get_supported_filters(self, search_type: str = "tutorial") -> List[str]:
        """
        Get list of supported filters for a search type.
        
        Args:
            search_type: Type of search ("tutorial" or "document")
            
        Returns:
            List[str]: List of supported filter names
        """
        if search_type.lower() == "tutorial":
            return list(TutorialSearch.model_fields.keys())
        elif search_type.lower() == "document":
            return list(DocumentSearch.model_fields.keys())
        else:
            return []


# Legacy function for compatibility with workplan examples
def query_analyzer(question: str) -> TutorialSearch:
    """
    Legacy function to maintain compatibility with workplan examples.
    Converts a natural language question into a TutorialSearch query.
    
    Args:
        question: Natural language question
        
    Returns:
        TutorialSearch: Structured search query
    """
    structurer = QueryStructurer()
    return structurer.structure_tutorial_query(question)