"""
Query Decomposition for breaking down complex queries into simpler sub-questions
"""

from typing import List, Dict, Any, Optional, Callable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from ..core.models import Document, ProcessedQuery
from ..core.exceptions import QueryProcessingError


class QueryDecomposer:
    """
    Decomposes complex queries into simpler, self-contained sub-questions.
    
    Based on the Decomposition approach from workplan/01AdvancedQueryTransformations.md,
    this class uses an LLM to break down complex queries into a set of simpler sub-problems
    that can be answered in isolation, then synthesizes a final comprehensive answer.
    """
    
    def __init__(
        self,
        llm_model: str = "gemini-1.5-flash",
        temperature: float = 0.0,
        num_sub_questions: int = 3,
        **llm_kwargs
    ):
        """
        Initialize the QueryDecomposer.
        
        Args:
            llm_model: The Google Gemini model to use
            temperature: Temperature for generation (0.0 for deterministic)
            num_sub_questions: Number of sub-questions to generate
            **llm_kwargs: Additional arguments for the LLM
        """
        self.llm_model = llm_model
        self.temperature = temperature
        self.num_sub_questions = num_sub_questions
        self.llm_kwargs = llm_kwargs
        
        # Initialize the LLM
        self.llm = ChatGoogleGenerativeAI(
            model=llm_model,
            temperature=temperature,
            **llm_kwargs
        )
        
        # Create the prompt template for decomposition
        self.decomposition_prompt = self._create_decomposition_prompt()
        self.synthesis_prompt = self._create_synthesis_prompt()
        
        # Build the decomposition chain
        self.decomposition_chain = (
            self.decomposition_prompt
            | self.llm
            | StrOutputParser()
            | self._parse_sub_questions
        )
        
        # Build the synthesis chain
        self.synthesis_chain = (
            self.synthesis_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _create_decomposition_prompt(self) -> ChatPromptTemplate:
        """Create the prompt template for query decomposition."""
        template = f"""You are a helpful assistant that generates multiple sub-questions related to an input question.
The goal is to break down the input into a set of sub-problems / sub-questions that can be answered in isolation.
Generate multiple search queries related to: {{question}}
Output ({self.num_sub_questions} queries):"""
        
        return ChatPromptTemplate.from_template(template)
    
    def _create_synthesis_prompt(self) -> ChatPromptTemplate:
        """Create the prompt template for synthesizing answers."""
        template = """Here is a set of Q+A pairs:

{context}

Use these to synthesize an answer to the original question: {question}"""
        
        return ChatPromptTemplate.from_template(template)
    
    def _parse_sub_questions(self, output: str) -> List[str]:
        """
        Parse the LLM output into a list of sub-questions.
        
        Args:
            output: Raw output from the LLM
            
        Returns:
            List of parsed sub-questions
        """
        # Split by newlines and clean up
        questions = [q.strip() for q in output.split("\n") if q.strip()]
        
        # Remove numbering if present (e.g., "1. ", "2. ")
        cleaned_questions = []
        for question in questions:
            # Remove leading numbers and dots
            import re
            cleaned_question = re.sub(r'^\d+\.\s*', '', question)
            if cleaned_question:
                cleaned_questions.append(cleaned_question)
        
        return cleaned_questions[:self.num_sub_questions]  # Limit to requested number
    
    def decompose_query(self, question: str) -> List[str]:
        """
        Decompose a complex query into simpler sub-questions.
        
        Args:
            question: The original complex question
            
        Returns:
            List of sub-questions
            
        Raises:
            QueryProcessingError: If decomposition fails
        """
        try:
            # Generate sub-questions
            sub_questions = self.decomposition_chain.invoke({"question": question})
            
            # Validate sub-questions
            validated_questions = self._validate_sub_questions(sub_questions, question)
            
            return validated_questions
            
        except Exception as e:
            raise QueryProcessingError(f"Failed to decompose query: {str(e)}")
    
    def _validate_sub_questions(self, sub_questions: List[str], original_question: str) -> List[str]:
        """
        Validate and filter generated sub-questions.
        
        Args:
            sub_questions: List of generated sub-questions
            original_question: The original question for comparison
            
        Returns:
            List of validated sub-questions
        """
        validated = []
        
        for question in sub_questions:
            # Skip empty questions
            if not question or len(question.strip()) < 5:
                continue
                
            # Skip questions that are too similar to the original
            if self._is_too_similar(question, original_question):
                continue
                
            # Skip questions that are too long (likely hallucinated)
            if len(question) > 500:
                continue
                
            # Ensure it's actually a question
            if not self._is_question(question):
                # Try to convert to question format
                question = f"What is {question.lower()}?" if not question.endswith('?') else question
                
            validated.append(question)
        
        # Ensure we have at least some sub-questions
        if not validated:
            # Create fallback sub-questions
            validated = self._create_fallback_questions(original_question)
        
        return validated[:self.num_sub_questions]
    
    def _is_too_similar(self, question1: str, question2: str, threshold: float = 0.8) -> bool:
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
    
    def _is_question(self, text: str) -> bool:
        """
        Check if text is formatted as a question.
        
        Args:
            text: Text to check
            
        Returns:
            True if text appears to be a question
        """
        text = text.strip()
        if not text:
            return False
            
        # Check for question marks
        if text.endswith('?'):
            return True
            
        # Check for question words at the beginning
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'can', 'could', 'would', 'should', 'is', 'are', 'do', 'does', 'did']
        first_word = text.lower().split()[0] if text.split() else ""
        
        return first_word in question_words
    
    def _create_fallback_questions(self, original_question: str) -> List[str]:
        """
        Create fallback sub-questions when decomposition fails.
        
        Args:
            original_question: The original question
            
        Returns:
            List of fallback sub-questions
        """
        # Extract key terms from the original question
        import re
        
        # Remove common question words and extract key terms
        clean_question = re.sub(r'\b(what|how|why|when|where|who|which|is|are|do|does|did|can|could|would|should)\b', '', original_question.lower())
        key_terms = [term.strip() for term in clean_question.split() if len(term.strip()) > 2]
        
        fallback_questions = []
        
        if key_terms:
            # Create basic sub-questions
            fallback_questions.append(f"What are the main components of {' '.join(key_terms[:3])}?")
            fallback_questions.append(f"How does {' '.join(key_terms[:2])} work?")
            fallback_questions.append(f"What are the benefits of {' '.join(key_terms[:2])}?")
        else:
            # Very basic fallback
            fallback_questions = [
                "What are the main concepts involved?",
                "How does this process work?",
                "What are the key benefits?"
            ]
        
        return fallback_questions
    
    def answer_sub_questions(
        self,
        sub_questions: List[str],
        retriever_func: Callable[[str], List[Document]],
        answerer_func: Callable[[str, List[Document]], str]
    ) -> List[str]:
        """
        Answer each sub-question using provided retriever and answerer functions.
        
        Args:
            sub_questions: List of sub-questions to answer
            retriever_func: Function that retrieves documents for a question
            answerer_func: Function that generates an answer from question and documents
            
        Returns:
            List of answers corresponding to sub-questions
            
        Raises:
            QueryProcessingError: If answering fails
        """
        try:
            answers = []
            
            for sub_question in sub_questions:
                # Retrieve relevant documents
                documents = retriever_func(sub_question)
                
                # Generate answer
                answer = answerer_func(sub_question, documents)
                answers.append(answer)
            
            return answers
            
        except Exception as e:
            raise QueryProcessingError(f"Failed to answer sub-questions: {str(e)}")
    
    def synthesize_answer(
        self,
        original_question: str,
        sub_questions: List[str],
        sub_answers: List[str]
    ) -> str:
        """
        Synthesize a final answer from sub-question/answer pairs.
        
        Args:
            original_question: The original complex question
            sub_questions: List of sub-questions
            sub_answers: List of corresponding answers
            
        Returns:
            Synthesized final answer
            
        Raises:
            QueryProcessingError: If synthesis fails
        """
        try:
            # Format Q&A pairs
            qa_context = self._format_qa_pairs(sub_questions, sub_answers)
            
            # Synthesize final answer
            final_answer = self.synthesis_chain.invoke({
                "context": qa_context,
                "question": original_question
            })
            
            return final_answer
            
        except Exception as e:
            raise QueryProcessingError(f"Failed to synthesize answer: {str(e)}")
    
    def _format_qa_pairs(self, questions: List[str], answers: List[str]) -> str:
        """
        Format question and answer pairs into a single context string.
        
        Args:
            questions: List of questions
            answers: List of corresponding answers
            
        Returns:
            Formatted Q&A context string
        """
        formatted_string = ""
        for i, (question, answer) in enumerate(zip(questions, answers), start=1):
            formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
        return formatted_string.strip()
    
    def process_query_with_decomposition(
        self,
        question: str,
        retriever_func: Callable[[str], List[Document]],
        answerer_func: Callable[[str, List[Document]], str]
    ) -> str:
        """
        Process a complex query using full decomposition workflow.
        
        Args:
            question: The original complex question
            retriever_func: Function that retrieves documents for a question
            answerer_func: Function that generates an answer from question and documents
            
        Returns:
            Final synthesized answer
            
        Raises:
            QueryProcessingError: If processing fails
        """
        try:
            # Step 1: Decompose the query
            sub_questions = self.decompose_query(question)
            
            # Step 2: Answer each sub-question
            sub_answers = self.answer_sub_questions(sub_questions, retriever_func, answerer_func)
            
            # Step 3: Synthesize final answer
            final_answer = self.synthesize_answer(question, sub_questions, sub_answers)
            
            return final_answer
            
        except Exception as e:
            raise QueryProcessingError(f"Failed to process query with decomposition: {str(e)}")
    
    def process_query(self, question: str, **kwargs) -> ProcessedQuery:
        """
        Process a query and return a ProcessedQuery object with sub-questions.
        
        Args:
            question: The original complex question
            **kwargs: Additional processing parameters
            
        Returns:
            ProcessedQuery object with sub-questions
        """
        try:
            # Decompose the query
            sub_questions = self.decompose_query(question)
            
            # Create metadata
            metadata = {
                "num_sub_questions": len(sub_questions),
                "model_used": self.llm_model,
                "temperature": self.temperature,
                **kwargs
            }
            
            return ProcessedQuery(
                original_query=question,
                transformed_queries=sub_questions,
                strategy_used="decomposition",
                metadata=metadata
            )
            
        except Exception as e:
            raise QueryProcessingError(f"Failed to process query with decomposition: {str(e)}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return {
            "llm_model": self.llm_model,
            "temperature": self.temperature,
            "num_sub_questions": self.num_sub_questions,
            "strategy": "decomposition"
        }