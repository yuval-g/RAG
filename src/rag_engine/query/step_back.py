"""
Step-Back Prompting for generating broader context questions
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from ..core.models import Document, ProcessedQuery
from ..core.exceptions import QueryProcessingError


class StepBackProcessor:
    """
    Implements Step-Back Prompting for broader context generation.
    
    Based on the Step-Back approach from workplan/01AdvancedQueryTransformations.md,
    this class uses an LLM to take a "step back" and form more general questions
    when the user's query is too specific. It retrieves context for both the
    specific and general questions, providing richer context for the final answer.
    """
    
    def __init__(
        self,
        llm_model: str = "gemini-1.5-flash",
        temperature: float = 0.0,
        **llm_kwargs
    ):
        """
        Initialize the StepBackProcessor.
        
        Args:
            llm_model: The Google Gemini model to use
            temperature: Temperature for generation (0.0 for deterministic)
            **llm_kwargs: Additional arguments for the LLM
        """
        self.llm_model = llm_model
        self.temperature = temperature
        self.llm_kwargs = llm_kwargs
        
        # Initialize the LLM
        self.llm = ChatGoogleGenerativeAI(
            model=llm_model,
            temperature=temperature,
            **llm_kwargs
        )
        
        # Create the step-back prompt with few-shot examples
        self.step_back_prompt = self._create_step_back_prompt()
        self.response_prompt = self._create_response_prompt()
        
        # Build the step-back generation chain
        self.step_back_chain = (
            self.step_back_prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Build the response generation chain
        self.response_chain = (
            self.response_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _create_step_back_prompt(self) -> ChatPromptTemplate:
        """Create the step-back prompt with few-shot examples."""
        # Few-shot examples to teach the model how to generate step-back questions
        examples = [
            {
                "input": "Could the members of The Police perform lawful arrests?",
                "output": "what can the members of The Police do?",
            },
            {
                "input": "Jan Sindel's was born in what country?",
                "output": "what is Jan Sindel's personal history?",
            },
            {
                "input": "What is the specific implementation of task decomposition in GPT-4?",
                "output": "what are the general approaches to task decomposition in AI systems?",
            },
            {
                "input": "How does the attention mechanism work in transformer models?",
                "output": "what are the key components of neural network architectures?",
            },
            {
                "input": "What is the exact algorithm used in BERT's masked language modeling?",
                "output": "what are the different approaches to language model training?",
            }
        ]
        
        # Define how each example is formatted in the prompt
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{input}"),  # User input
            ("ai", "{output}")     # Model's response
        ])
        
        # Wrap the few-shot examples into a reusable prompt template
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples,
        )
        
        # Full prompt includes system instruction, few-shot examples, and the user question
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are an expert at world knowledge. Your task is to step back and paraphrase a question "
             "to a more generic step-back question, which is easier to answer. Here are a few examples:"),
            few_shot_prompt,
            ("user", "{question}"),
        ])
        
        return prompt
    
    def _create_response_prompt(self) -> ChatPromptTemplate:
        """Create the response prompt that uses both normal and step-back context."""
        template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

# Normal Context
{normal_context}

# Step-Back Context
{step_back_context}

# Original Question: {question}
# Answer:"""
        
        return ChatPromptTemplate.from_template(template)
    
    def generate_step_back_question(self, question: str) -> str:
        """
        Generate a step-back (more general) version of the input question.
        
        Args:
            question: The original specific question
            
        Returns:
            Step-back question that is more general
            
        Raises:
            QueryProcessingError: If step-back generation fails
        """
        try:
            step_back_question = self.step_back_chain.invoke({"question": question})
            
            # Validate the step-back question
            validated_question = self._validate_step_back_question(step_back_question, question)
            
            return validated_question
            
        except Exception as e:
            raise QueryProcessingError(f"Failed to generate step-back question: {str(e)}")
    
    def _validate_step_back_question(self, step_back_question: str, original_question: str) -> str:
        """
        Validate and potentially fix the generated step-back question.
        
        Args:
            step_back_question: Generated step-back question
            original_question: Original question for comparison
            
        Returns:
            Validated step-back question
        """
        # Clean up the step-back question
        step_back_question = step_back_question.strip()
        
        # If empty or too short, create a fallback
        if not step_back_question or len(step_back_question) < 5:
            return self._create_fallback_step_back(original_question)
        
        # If too similar to original, create a more general version
        if self._is_too_similar(step_back_question, original_question):
            return self._create_fallback_step_back(original_question)
        
        # Ensure it's properly formatted as a question
        if not step_back_question.endswith('?'):
            step_back_question += '?'
        
        return step_back_question
    
    def _is_too_similar(self, question1: str, question2: str, threshold: float = 0.7) -> bool:
        """
        Check if two questions are too similar using simple word overlap.
        
        Args:
            question1: First question
            question2: Second question
            threshold: Similarity threshold (0-1)
            
        Returns:
            True if questions are too similar
        """
        # Simple word-based similarity check
        words1 = set(question1.lower().split())
        words2 = set(question2.lower().split())
        
        if not words1 or not words2:
            return False
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0
        return similarity > threshold
    
    def _create_fallback_step_back(self, original_question: str) -> str:
        """
        Create a fallback step-back question when generation fails.
        
        Args:
            original_question: The original question
            
        Returns:
            Fallback step-back question
        """
        # Extract key terms and create a more general question
        import re
        
        # Remove common question words and extract key terms
        clean_question = re.sub(r'\b(what|how|why|when|where|who|which|is|are|do|does|did|can|could|would|should|the|a|an)\b', '', original_question.lower())
        key_terms = [term.strip() for term in clean_question.split() if len(term.strip()) > 2]
        
        if key_terms:
            # Create a general question about the domain
            main_term = key_terms[0]
            return f"What are the general principles and approaches related to {main_term}?"
        else:
            # Very basic fallback
            return "What are the fundamental concepts and principles involved?"
    
    def process_with_retriever(
        self,
        question: str,
        retriever_func: Callable[[str], List[Document]],
        top_k: int = 5
    ) -> Tuple[List[Document], List[Document], str]:
        """
        Process a query with step-back prompting using a provided retriever function.
        
        Args:
            question: The original specific question
            retriever_func: Function that takes a query and returns List[Document]
            top_k: Number of documents to retrieve for each query
            
        Returns:
            Tuple of (normal_context_docs, step_back_context_docs, step_back_question)
            
        Raises:
            QueryProcessingError: If processing fails
        """
        try:
            # Generate step-back question
            step_back_question = self.generate_step_back_question(question)
            
            # Retrieve documents for both questions
            normal_docs = retriever_func(question)[:top_k]
            step_back_docs = retriever_func(step_back_question)[:top_k]
            
            return normal_docs, step_back_docs, step_back_question
            
        except Exception as e:
            raise QueryProcessingError(f"Failed to process query with step-back prompting: {str(e)}")
    
    def generate_response_with_step_back(
        self,
        question: str,
        normal_context: List[Document],
        step_back_context: List[Document]
    ) -> str:
        """
        Generate a response using both normal and step-back context.
        
        Args:
            question: The original question
            normal_context: Documents retrieved for the original question
            step_back_context: Documents retrieved for the step-back question
            
        Returns:
            Generated response incorporating both contexts
            
        Raises:
            QueryProcessingError: If response generation fails
        """
        try:
            # Format context documents
            normal_context_str = self._format_documents(normal_context)
            step_back_context_str = self._format_documents(step_back_context)
            
            # Generate response
            response = self.response_chain.invoke({
                "question": question,
                "normal_context": normal_context_str,
                "step_back_context": step_back_context_str
            })
            
            return response
            
        except Exception as e:
            raise QueryProcessingError(f"Failed to generate response with step-back context: {str(e)}")
    
    def _format_documents(self, documents: List[Document]) -> str:
        """
        Format a list of documents into a single context string.
        
        Args:
            documents: List of documents to format
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant context found."
        
        formatted_docs = []
        for i, doc in enumerate(documents, 1):
            formatted_docs.append(f"Document {i}: {doc.content}")
        
        return "\n\n".join(formatted_docs)
    
    def process_query_with_step_back(
        self,
        question: str,
        retriever_func: Callable[[str], List[Document]],
        top_k: int = 5
    ) -> str:
        """
        Process a query using the full step-back workflow.
        
        Args:
            question: The original specific question
            retriever_func: Function that retrieves documents for a question
            top_k: Number of documents to retrieve for each query
            
        Returns:
            Final response incorporating both normal and step-back context
            
        Raises:
            QueryProcessingError: If processing fails
        """
        try:
            # Step 1: Retrieve context for both questions
            normal_docs, step_back_docs, step_back_question = self.process_with_retriever(
                question, retriever_func, top_k
            )
            
            # Step 2: Generate response with both contexts
            response = self.generate_response_with_step_back(
                question, normal_docs, step_back_docs
            )
            
            return response
            
        except Exception as e:
            raise QueryProcessingError(f"Failed to process query with step-back workflow: {str(e)}")
    
    def process_query(self, question: str, **kwargs) -> ProcessedQuery:
        """
        Process a query and return a ProcessedQuery object with step-back question.
        
        Args:
            question: The original specific question
            **kwargs: Additional processing parameters
            
        Returns:
            ProcessedQuery object with step-back question
        """
        try:
            # Generate step-back question
            step_back_question = self.generate_step_back_question(question)
            
            # Create metadata
            metadata = {
                "step_back_question": step_back_question,
                "model_used": self.llm_model,
                "temperature": self.temperature,
                **kwargs
            }
            
            return ProcessedQuery(
                original_query=question,
                transformed_queries=[question, step_back_question],
                strategy_used="step_back",
                metadata=metadata
            )
            
        except Exception as e:
            raise QueryProcessingError(f"Failed to process query with step-back prompting: {str(e)}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return {
            "llm_model": self.llm_model,
            "temperature": self.temperature,
            "strategy": "step_back"
        }