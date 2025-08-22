"""
HyDE (Hypothetical Document Embeddings) for improved retrieval through hypothetical document generation
"""

from typing import List, Dict, Any, Optional, Callable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from ..core.models import Document, ProcessedQuery
from ..core.exceptions import QueryProcessingError
from ..common.utils import get_llm


class HyDEProcessor:
    """
    Implements HyDE (Hypothetical Document Embeddings) for improved retrieval.
    
    Based on the HyDE approach from workplan/01AdvancedQueryTransformations.md,
    this class generates hypothetical documents that answer the user's question.
    These fake documents, while not factually correct, are semantically rich and
    use the kind of language expected in real answers. The embeddings of these
    hypothetical documents are then used to find real documents that are
    semantically similar to an ideal answer.
    """
    
    def __init__(
        self,
        llm_model: str = "gemini-2.0-flash-lite",
        temperature: float = 0.0,
        document_style: str = "scientific_paper",
        **llm_kwargs
    ):
        """
        Initialize the HyDEProcessor.
        
        Args:
            llm_model: The Google Gemini model to use
            temperature: Temperature for generation (0.0 for deterministic)
            document_style: Style of hypothetical document to generate
            **llm_kwargs: Additional arguments for the LLM
        """
        self.llm_model = llm_model
        self.temperature = temperature
        self.document_style = document_style
        self.llm_kwargs = llm_kwargs

        # Initialize the LLM
        self.llm = get_llm(llm_model, temperature, **llm_kwargs)
        
        # Create the prompt template for hypothetical document generation
        self.hyde_prompt = self._create_hyde_prompt()
        
        # Build the generation chain
        self.generation_chain = (
            self.hyde_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _create_hyde_prompt(self) -> ChatPromptTemplate:
        """Create the prompt template for hypothetical document generation."""
        # Different prompt templates based on document style
        if self.document_style == "scientific_paper":
            template = """Please write a scientific paper passage to answer the question
Question: {question}
Passage:"""
        elif self.document_style == "technical_documentation":
            template = """Please write a technical documentation section to answer the question
Question: {question}
Documentation:"""
        elif self.document_style == "tutorial":
            template = """Please write a tutorial section to answer the question
Question: {question}
Tutorial:"""
        elif self.document_style == "encyclopedia":
            template = """Please write an encyclopedia entry to answer the question
Question: {question}
Entry:"""
        else:
            # Default to scientific paper style
            template = """Please write a scientific paper passage to answer the question
Question: {question}
Passage:"""
        
        return ChatPromptTemplate.from_template(template)
    
    def generate_hypothetical_document(self, question: str) -> str:
        """
        Generate a hypothetical document that answers the question.
        
        Args:
            question: The user's question
            
        Returns:
            Hypothetical document content
            
        Raises:
            QueryProcessingError: If document generation fails
        """
        try:
            # Generate hypothetical document
            hypothetical_doc = self.generation_chain.invoke({"question": question})
            
            # Validate and clean the document
            validated_doc = self._validate_hypothetical_document(hypothetical_doc, question)
            
            return validated_doc
            
        except Exception as e:
            raise QueryProcessingError(f"Failed to generate hypothetical document: {str(e)}")
    
    def _validate_hypothetical_document(self, document: str, question: str) -> str:
        """
        Validate and potentially fix the generated hypothetical document.
        
        Args:
            document: Generated hypothetical document
            question: Original question for context
            
        Returns:
            Validated hypothetical document
        """
        # Clean up the document
        document = document.strip()
        
        # If empty or too short, create a fallback
        if not document or len(document) < 20:
            return self._create_fallback_document(question)
        
        # If too long, truncate (but keep it meaningful)
        if len(document) > 2000:
            # Find a good truncation point (end of sentence)
            truncated = document[:1800]
            last_period = truncated.rfind('.')
            if last_period > 1000:  # Ensure we don't truncate too much
                document = truncated[:last_period + 1]
            else:
                document = truncated + "..."
        
        # Ensure the document is coherent and relevant
        if not self._is_relevant_document(document, question):
            return self._create_fallback_document(question)
        
        return document
    
    def _is_relevant_document(self, document: str, question: str, threshold: float = 0.3) -> bool:
        """
        Check if the generated document is relevant to the question.
        
        Args:
            document: Generated document
            question: Original question
            threshold: Relevance threshold (0-1)
            
        Returns:
            True if document appears relevant
        """
        # Simple relevance check based on word overlap
        import re
        
        # Extract key terms from question (remove stop words)
        stop_words = {'what', 'how', 'why', 'when', 'where', 'who', 'which', 'is', 'are', 'do', 'does', 'did', 'can', 'could', 'would', 'should', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        question_words = question_words - stop_words
        
        document_words = set(re.findall(r'\b\w+\b', document.lower()))
        
        if not question_words:
            return True  # If no key terms, assume relevant
        
        # Calculate overlap
        overlap = len(question_words.intersection(document_words))
        relevance = overlap / len(question_words)
        
        return relevance >= threshold
    
    def _create_fallback_document(self, question: str) -> str:
        """
        Create a fallback hypothetical document when generation fails.
        
        Args:
            question: The original question
            
        Returns:
            Fallback hypothetical document
        """
        # Extract key terms from the question
        import re
        
        # Remove question words and extract key terms
        clean_question = re.sub(r'\b(what|how|why|when|where|who|which|is|are|do|does|did|can|could|would|should)\b', '', question.lower())
        key_terms = [term.strip() for term in clean_question.split() if len(term.strip()) > 2]
        
        if key_terms:
            main_terms = ' '.join(key_terms[:3])  # Use first 3 key terms
            
            if self.document_style == "scientific_paper":
                return f"This paper examines {main_terms} and their applications. The research demonstrates that {main_terms} play a crucial role in modern systems. Through comprehensive analysis, we show that understanding {main_terms} is essential for effective implementation. The methodology involves systematic evaluation of {main_terms} across different contexts. Results indicate that {main_terms} provide significant benefits when properly applied."
            elif self.document_style == "technical_documentation":
                return f"This section covers {main_terms} and their implementation details. {main_terms.title()} are fundamental components that enable system functionality. To work with {main_terms}, developers need to understand their core principles and best practices. The following guidelines help ensure proper usage of {main_terms} in production environments."
            else:
                return f"The concept of {main_terms} encompasses several important aspects. {main_terms.title()} represent key elements in this domain. Understanding {main_terms} requires examining their properties, relationships, and practical applications. This comprehensive overview provides insights into how {main_terms} function and their significance in the broader context."
        else:
            # Very basic fallback
            return "This document provides comprehensive information about the topic in question. The subject matter involves multiple interconnected concepts that work together to form a complete understanding. Through detailed analysis and explanation, this content addresses the key aspects and provides valuable insights for readers seeking to understand the fundamental principles involved."
    
    def process_with_retriever(
        self,
        question: str,
        retriever_func: Callable[[str], List[Document]],
        embedding_func: Optional[Callable[[str], List[float]]] = None,
        top_k: int = 5
    ) -> List[Document]:
        """
        Process a query with HyDE using a provided retriever function.
        
        Args:
            question: The original user question
            retriever_func: Function that takes a query and returns List[Document]
            embedding_func: Optional function to generate embeddings for hypothetical doc
            top_k: Number of documents to retrieve
            
        Returns:
            List of documents retrieved using hypothetical document
            
        Raises:
            QueryProcessingError: If processing fails
        """
        try:
            # Generate hypothetical document
            hypothetical_doc = self.generate_hypothetical_document(question)
            
            # Use hypothetical document for retrieval
            # If embedding_func is provided, we could use it for more sophisticated retrieval
            # For now, we'll use the hypothetical document text directly with the retriever
            retrieved_docs = retriever_func(hypothetical_doc)
            
            return retrieved_docs[:top_k]
            
        except Exception as e:
            raise QueryProcessingError(f"Failed to process query with HyDE: {str(e)}")
    
    def process_with_embedding_retriever(
        self,
        question: str,
        embedding_func: Callable[[str], List[float]],
        vector_search_func: Callable[[List[float], int], List[Document]],
        top_k: int = 5
    ) -> List[Document]:
        """
        Process a query with HyDE using embedding-based retrieval.
        
        Args:
            question: The original user question
            embedding_func: Function to generate embeddings
            vector_search_func: Function that takes embedding and returns documents
            top_k: Number of documents to retrieve
            
        Returns:
            List of documents retrieved using hypothetical document embedding
            
        Raises:
            QueryProcessingError: If processing fails
        """
        try:
            # Generate hypothetical document
            hypothetical_doc = self.generate_hypothetical_document(question)
            
            # Generate embedding for hypothetical document
            hypothetical_embedding = embedding_func(hypothetical_doc)
            
            # Use hypothetical document embedding for retrieval
            retrieved_docs = vector_search_func(hypothetical_embedding, top_k)
            
            return retrieved_docs
            
        except Exception as e:
            raise QueryProcessingError(f"Failed to process query with HyDE embedding retrieval: {str(e)}")
    
    def generate_multiple_hypothetical_documents(
        self,
        question: str,
        num_documents: int = 3
    ) -> List[str]:
        """
        Generate multiple hypothetical documents for the same question.
        
        Args:
            question: The user's question
            num_documents: Number of hypothetical documents to generate
            
        Returns:
            List of hypothetical documents
            
        Raises:
            QueryProcessingError: If generation fails
        """
        try:
            documents = []
            
            # Generate multiple documents with slight temperature variation
            original_temp = self.temperature
            
            for i in range(num_documents):
                # Slightly vary temperature for diversity
                self.llm.temperature = original_temp + (i * 0.1)
                
                # Generate document
                doc = self.generate_hypothetical_document(question)
                documents.append(doc)
            
            # Restore original temperature
            self.llm.temperature = original_temp
            
            return documents
            
        except Exception as e:
            raise QueryProcessingError(f"Failed to generate multiple hypothetical documents: {str(e)}")
    
    def process_query(self, question: str, **kwargs) -> ProcessedQuery:
        """
        Process a query and return a ProcessedQuery object with hypothetical document.
        
        Args:
            question: The original user question
            **kwargs: Additional processing parameters
            
        Returns:
            ProcessedQuery object with hypothetical document
        """
        try:
            # Generate hypothetical document
            hypothetical_doc = self.generate_hypothetical_document(question)
            
            # Create metadata
            metadata = {
                "hypothetical_document": hypothetical_doc,
                "document_style": self.document_style,
                "model_used": self.llm_model,
                "temperature": self.temperature,
                **kwargs
            }
            
            return ProcessedQuery(
                original_query=question,
                transformed_queries=[hypothetical_doc],  # Use hypothetical doc as transformed query
                strategy_used="hyde",
                metadata=metadata
            )
            
        except Exception as e:
            raise QueryProcessingError(f"Failed to process query with HyDE: {str(e)}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return {
            "llm_model": self.llm_model,
            "temperature": self.temperature,
            "document_style": self.document_style,
            "strategy": "hyde"
        }