"""
Enhanced generation engine implementation with grounding, citation, and validation features.
Adapts from workplan/00BasicRAGSystem.md and implements requirement 5.4 for grounded generation.
Uses Google Gemini instead of OpenAI as per steering rules.
"""

from typing import List, Optional, Dict, Any, Tuple
import logging
import re
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain import hub

from ..core.interfaces import BaseLLMProvider
from ..core.models import Document, RAGResponse
from ..core.config import PipelineConfig
from .llm_providers import LLMProviderFactory


logger = logging.getLogger(__name__)


class GenerationEngine:
    """
    Enhanced generation engine with grounding, citation, and validation features.
    Implements requirement 5.4 for grounded generation with source attribution.
    Adapts the generation implementation from workplan/00BasicRAGSystem.md but uses Google Gemini.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the GenerationEngine.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.llm_provider = LLMProviderFactory.create_provider(config.llm_provider, config)
        self.output_parser = StrOutputParser()
        
        # Load prompt template from LangChain Hub as in workplan
        try:
            self.prompt_template = hub.pull("rlm/rag-prompt")
            logger.info("Loaded RAG prompt template from LangChain Hub")
        except Exception as e:
            logger.warning(f"Could not load prompt from hub: {str(e)}, using default")
            self.prompt_template = self._get_default_prompt()
        
        # Initialize grounded generation prompt template
        self.grounded_prompt_template = self._get_grounded_prompt()
        
        # Initialize citation prompt template
        self.citation_prompt_template = self._get_citation_prompt()
        
        # Initialize validation prompt template
        self.validation_prompt_template = self._get_validation_prompt()
        
        logger.info("GenerationEngine initialized with grounding features")
    
    def _get_default_prompt(self):
        """Get default RAG prompt template if hub loading fails"""
        from langchain_core.prompts import PromptTemplate
        
        template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}
Context: {context}
Answer:"""
        
        return PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
    
    def _get_grounded_prompt(self):
        """Get grounded generation prompt template that ensures response grounding"""
        template = """You are an assistant for question-answering tasks. Use ONLY the following pieces of retrieved context to answer the question. Your response must be grounded in the provided context.

IMPORTANT INSTRUCTIONS:
1. Base your answer ONLY on the information provided in the context below
2. If the context doesn't contain enough information to answer the question, say "I don't have enough information in the provided context to answer this question"
3. Do not add information that is not present in the context
4. Keep your answer concise and factual
5. When possible, reference specific parts of the context in your answer

Question: {question}

Context:
{context}

Grounded Answer:"""
        
        return PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
    
    def _get_citation_prompt(self):
        """Get citation generation prompt template for source attribution"""
        template = """You are an assistant that provides answers with proper citations. Use the following numbered context sources to answer the question, and include citations in your response.

CITATION INSTRUCTIONS:
1. Include citations as [1], [2], etc. referring to the source numbers below
2. Place citations immediately after the relevant information
3. Use multiple citations when information comes from multiple sources
4. Ensure every factual claim has a citation
5. If you cannot answer based on the provided sources, say so clearly

Question: {question}

Context Sources:
{numbered_context}

Answer with Citations:"""
        
        return PromptTemplate(
            input_variables=["numbered_context", "question"],
            template=template
        )
    
    def _get_validation_prompt(self):
        """Get validation prompt template for response grounding validation"""
        template = """You are a validator checking if an answer is properly grounded in the provided context. Analyze the answer and context carefully.

Question: {question}

Context:
{context}

Answer to Validate:
{answer}

VALIDATION CRITERIA:
1. Is every factual claim in the answer supported by the context?
2. Does the answer contain information not present in the context?
3. Are there any contradictions between the answer and context?
4. Is the answer relevant to the question?

Provide your validation in this format:
GROUNDED: [YES/NO/PARTIALLY]
CONFIDENCE: [0.0-1.0]
ISSUES: [List any specific issues found, or "None" if no issues]
EXPLANATION: [Brief explanation of your assessment]"""
        
        return PromptTemplate(
            input_variables=["question", "context", "answer"],
            template=template
        )
    
    def format_docs(self, docs: List[Document]) -> str:
        """
        Format documents for context.
        Adapts the format_docs helper function from workplan/00BasicRAGSystem.md
        
        Args:
            docs: List of documents to format
            
        Returns:
            str: Formatted context string
        """
        return "\n\n".join(doc.content for doc in docs)
    
    def generate(self, query: str, context: List[Document]) -> str:
        """
        Generate response using query and retrieved context.
        
        Args:
            query: User query
            context: Retrieved documents for context
            
        Returns:
            str: Generated response
        """
        try:
            if not context:
                logger.warning("No context provided for generation")
                return "I don't have enough information to answer this question."
            
            # Format context
            formatted_context = self.format_docs(context)
            
            # Create the prompt
            prompt_input = {
                "context": formatted_context,
                "question": query
            }
            
            # Generate response using the chain pattern from workplan
            formatted_prompt = self.prompt_template.format(**prompt_input)
            response = self.llm_provider.generate(formatted_prompt)
            
            # Parse output
            parsed_response = self.output_parser.parse(response)
            
            logger.info(f"Generated response for query: '{query[:50]}...'")
            return parsed_response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while generating a response."
    
    def generate_with_validation(self, query: str, context: List[Document], 
                               self_correction_engine=None) -> tuple[str, Dict[str, Any]]:
        """
        Generate response with optional self-correction validation.
        Implements Self-RAG validation from workplan/04AdvancedRetrieval-Generation.md
        
        Args:
            query: User query
            context: Retrieved documents for context
            self_correction_engine: Optional self-correction engine for validation
            
        Returns:
            tuple[str, Dict[str, Any]]: Generated response and validation metadata
        """
        try:
            # Generate initial response
            response = self.generate(query, context)
            
            # Apply self-correction validation if available
            if (self_correction_engine is not None and 
                hasattr(self_correction_engine, 'validate_generation')):
                
                validated_response, validation_metadata = self_correction_engine.validate_generation(
                    query, context, response
                )
                
                logger.info("Response validation applied")
                return validated_response, validation_metadata
            else:
                # No validation available
                return response, {"validation_applied": False, "reason": "not_available"}
                
        except Exception as e:
            logger.error(f"Error in generate_with_validation: {str(e)}")
            return "I apologize, but I encountered an error while generating a response.", {
                "validation_applied": False, 
                "error": str(e)
            }
    
    def create_rag_chain(self, retriever):
        """
        Create a RAG chain similar to workplan/00BasicRAGSystem.md
        Uses LangChain Expression Language (LCEL) pattern
        
        Args:
            retriever: Document retriever
            
        Returns:
            Runnable chain for RAG processing
        """
        try:
            from langchain_core.runnables import RunnableLambda
            
            # Create a lambda function for format_docs to make it compatible with LCEL
            format_docs_runnable = RunnableLambda(self.format_docs)
            
            # Create the chain using LCEL pattern from workplan
            rag_chain = (
                {"context": retriever | format_docs_runnable, "question": RunnablePassthrough()}
                | self.prompt_template
                | self.llm_provider.llm
                | self.output_parser
            )
            
            logger.info("RAG chain created successfully")
            return rag_chain
            
        except Exception as e:
            logger.error(f"Error creating RAG chain: {str(e)}")
            return None
    
    def invoke_chain(self, chain, query: str) -> str:
        """
        Invoke a RAG chain with a query.
        
        Args:
            chain: RAG chain to invoke
            query: User query
            
        Returns:
            str: Generated response
        """
        try:
            if chain is None:
                return "Chain not available for processing."
            
            response = chain.invoke(query)
            logger.info(f"Chain invoked successfully for query: '{query[:50]}...'")
            return response
            
        except Exception as e:
            logger.error(f"Error invoking chain: {str(e)}")
            return "I apologize, but I encountered an error while processing your request."
    
    def set_prompt_template(self, template: str) -> None:
        """
        Set a custom prompt template.
        
        Args:
            template: Custom prompt template string
        """
        try:
            from langchain_core.prompts import PromptTemplate
            
            self.prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template=template
            )
            logger.info("Custom prompt template set")
            
        except Exception as e:
            logger.error(f"Error setting prompt template: {str(e)}")
    
    def set_llm_provider(self, provider: BaseLLMProvider) -> None:
        """
        Set a custom LLM provider.
        
        Args:
            provider: Custom LLM provider
        """
        self.llm_provider = provider
        logger.info("Custom LLM provider set")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        return self.llm_provider.get_model_info()
    
    def generate_grounded(self, query: str, context: List[Document]) -> str:
        """
        Generate a grounded response that strictly adheres to the provided context.
        Implements requirement 5.4 for ensuring responses are grounded in retrieved context.
        
        Args:
            query: User query
            context: Retrieved documents for context
            
        Returns:
            str: Grounded response
        """
        try:
            if not context:
                logger.warning("No context provided for grounded generation")
                return "I don't have enough information in the provided context to answer this question."
            
            # Format context
            formatted_context = self.format_docs(context)
            
            # Create the grounded prompt
            prompt_input = {
                "context": formatted_context,
                "question": query
            }
            
            # Generate grounded response
            formatted_prompt = self.grounded_prompt_template.format(**prompt_input)
            response = self.llm_provider.generate(formatted_prompt)
            
            # Parse output
            parsed_response = self.output_parser.parse(response)
            
            logger.info(f"Generated grounded response for query: '{query[:50]}...'")
            return parsed_response
            
        except Exception as e:
            logger.error(f"Error generating grounded response: {str(e)}")
            return "I apologize, but I encountered an error while generating a grounded response."
    
    def generate_with_citations(self, query: str, context: List[Document]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generate response with proper citations and source attribution.
        
        Args:
            query: User query
            context: Retrieved documents for context
            
        Returns:
            Tuple[str, List[Dict[str, Any]]]: Response with citations and source metadata
        """
        try:
            if not context:
                logger.warning("No context provided for citation generation")
                return "I don't have enough information to answer this question.", []
            
            # Create numbered context with source tracking
            numbered_context, source_metadata = self._create_numbered_context(context)
            
            # Create the citation prompt
            prompt_input = {
                "numbered_context": numbered_context,
                "question": query
            }
            
            # Generate response with citations
            formatted_prompt = self.citation_prompt_template.format(**prompt_input)
            response = self.llm_provider.generate(formatted_prompt)
            
            # Parse output
            parsed_response = self.output_parser.parse(response)
            
            logger.info(f"Generated response with citations for query: '{query[:50]}...'")
            return parsed_response, source_metadata
            
        except Exception as e:
            logger.error(f"Error generating response with citations: {str(e)}")
            return "I apologize, but I encountered an error while generating a response with citations.", []
    
    def _create_numbered_context(self, context: List[Document]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Create numbered context for citation and track source metadata.
        
        Args:
            context: List of documents
            
        Returns:
            Tuple[str, List[Dict[str, Any]]]: Numbered context string and source metadata
        """
        numbered_parts = []
        source_metadata = []
        
        for i, doc in enumerate(context, 1):
            numbered_parts.append(f"[{i}] {doc.content}")
            
            # Extract source information from metadata
            source_info = {
                "citation_number": i,
                "doc_id": doc.doc_id,
                "metadata": doc.metadata.copy() if doc.metadata else {}
            }
            
            # Add common source fields if available
            if "source" in doc.metadata:
                source_info["source"] = doc.metadata["source"]
            if "title" in doc.metadata:
                source_info["title"] = doc.metadata["title"]
            if "url" in doc.metadata:
                source_info["url"] = doc.metadata["url"]
            
            source_metadata.append(source_info)
        
        numbered_context = "\n\n".join(numbered_parts)
        return numbered_context, source_metadata
    
    def validate_response_grounding(self, query: str, context: List[Document], response: str) -> Dict[str, Any]:
        """
        Validate that a response is properly grounded in the provided context.
        
        Args:
            query: Original user query
            context: Context documents used for generation
            response: Generated response to validate
            
        Returns:
            Dict[str, Any]: Validation results with grounding assessment
        """
        try:
            if not response.strip():
                return {
                    "grounded": "NO",
                    "confidence": 0.0,
                    "issues": ["Empty response"],
                    "explanation": "Response is empty"
                }
            
            if not context:
                return {
                    "grounded": "NO",
                    "confidence": 0.0,
                    "issues": ["No context provided"],
                    "explanation": "Cannot validate grounding without context"
                }
            
            # Format context
            formatted_context = self.format_docs(context)
            
            # Create validation prompt
            prompt_input = {
                "question": query,
                "context": formatted_context,
                "answer": response
            }
            
            # Generate validation assessment
            formatted_prompt = self.validation_prompt_template.format(**prompt_input)
            validation_response = self.llm_provider.generate(formatted_prompt)
            
            # Parse validation results
            validation_results = self._parse_validation_response(validation_response)
            
            logger.info(f"Response validation completed: {validation_results['grounded']}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating response grounding: {str(e)}")
            return {
                "grounded": "UNKNOWN",
                "confidence": 0.0,
                "issues": [f"Validation error: {str(e)}"],
                "explanation": "Error occurred during validation"
            }
    
    def _parse_validation_response(self, validation_response: str) -> Dict[str, Any]:
        """
        Parse the validation response into structured results.
        
        Args:
            validation_response: Raw validation response from LLM
            
        Returns:
            Dict[str, Any]: Parsed validation results
        """
        lines = validation_response.strip().split('\n')
        
        grounded = "UNKNOWN"
        confidence = 0.0
        issues = []
        explanation = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith("GROUNDED:"):
                grounded_str = line.split(":", 1)[1].strip().upper()
                if grounded_str in ["YES", "NO", "PARTIALLY"]:
                    grounded = grounded_str
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                    confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
                except ValueError:
                    confidence = 0.0
            elif line.startswith("ISSUES:"):
                issues_str = line.split(":", 1)[1].strip()
                if issues_str.lower() != "none":
                    issues = [issue.strip() for issue in issues_str.split(",")]
            elif line.startswith("EXPLANATION:"):
                explanation = line.split(":", 1)[1].strip()
        
        return {
            "grounded": grounded,
            "confidence": confidence,
            "issues": issues,
            "explanation": explanation
        }
    
    def generate_with_full_grounding(self, query: str, context: List[Document], 
                                   include_citations: bool = True,
                                   validate_grounding: bool = True) -> RAGResponse:
        """
        Generate a fully grounded response with citations and validation.
        This is the main method implementing requirement 5.4.
        
        Args:
            query: User query
            context: Retrieved documents for context
            include_citations: Whether to include citations in the response
            validate_grounding: Whether to validate response grounding
            
        Returns:
            RAGResponse: Complete response with grounding metadata
        """
        start_time = time.time()
        
        try:
            if not context:
                logger.warning("No context provided for full grounding generation")
                return RAGResponse(
                    answer="I don't have enough information to answer this question.",
                    source_documents=[],
                    confidence_score=0.0,
                    processing_time=time.time() - start_time,
                    metadata={
                        "grounding_applied": True,
                        "citations_included": False,
                        "validation_applied": False,
                        "error": "No context provided"
                    }
                )
            
            # Generate response with citations if requested
            if include_citations:
                response, source_metadata = self.generate_with_citations(query, context)
            else:
                response = self.generate_grounded(query, context)
                source_metadata = []
            
            # Validate grounding if requested
            validation_results = {}
            if validate_grounding:
                validation_results = self.validate_response_grounding(query, context, response)
            
            # Calculate confidence score based on validation
            confidence_score = 0.8  # Default confidence
            if validation_results:
                if validation_results["grounded"] == "YES":
                    confidence_score = max(0.8, validation_results["confidence"])
                elif validation_results["grounded"] == "PARTIALLY":
                    confidence_score = max(0.5, validation_results["confidence"] * 0.7)
                else:  # NO or UNKNOWN
                    confidence_score = min(0.3, validation_results["confidence"])
            
            processing_time = time.time() - start_time
            
            # Create comprehensive metadata
            metadata = {
                "grounding_applied": True,
                "citations_included": include_citations,
                "validation_applied": validate_grounding,
                "source_count": len(context),
                "processing_time": processing_time
            }
            
            if source_metadata:
                metadata["source_metadata"] = source_metadata
            
            if validation_results:
                metadata["validation_results"] = validation_results
            
            logger.info(f"Generated fully grounded response with confidence: {confidence_score:.2f}")
            
            return RAGResponse(
                answer=response,
                source_documents=context,
                confidence_score=confidence_score,
                processing_time=processing_time,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error in full grounding generation: {str(e)}")
            processing_time = time.time() - start_time
            return RAGResponse(
                answer="I apologize, but I encountered an error while generating a grounded response.",
                source_documents=context,
                confidence_score=0.0,
                processing_time=processing_time,
                metadata={
                    "grounding_applied": True,
                    "citations_included": include_citations,
                    "validation_applied": validate_grounding,
                    "processing_time": processing_time,
                    "error": str(e)
                }
            )
    
    def extract_citations_from_response(self, response: str) -> List[int]:
        """
        Extract citation numbers from a response.
        
        Args:
            response: Response text with citations
            
        Returns:
            List[int]: List of citation numbers found in the response
        """
        citation_pattern = r'\[(\d+)\]'
        citations = re.findall(citation_pattern, response)
        return [int(citation) for citation in citations]
    
    def update_config(self, config: PipelineConfig) -> None:
        """
        Update the generation engine configuration.
        
        Args:
            config: New pipeline configuration
        """
        self.config = config
        self.llm_provider = LLMProviderFactory.create_provider(config.llm_provider, config)
        logger.info("GenerationEngine configuration updated")