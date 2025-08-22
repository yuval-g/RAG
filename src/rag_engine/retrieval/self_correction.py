"""
Self-correction mechanisms for RAG systems implementing CRAG and Self-RAG approaches.
Based on concepts from workplan/04AdvancedRetrieval-Generation.md
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
from enum import Enum
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from ..core.models import Document, RAGResponse
from ..core.config import PipelineConfig
from ..core.exceptions import RetrievalError, GenerationError
from ..common.utils import get_llm


logger = logging.getLogger(__name__)


class RelevanceGrade(str, Enum):
    """Relevance grades for document assessment"""
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"
    AMBIGUOUS = "ambiguous"


class FactualityGrade(str, Enum):
    """Factuality grades for response assessment"""
    GROUNDED = "grounded"
    NOT_GROUNDED = "not_grounded"
    PARTIALLY_GROUNDED = "partially_grounded"


class RelevanceAssessment(BaseModel):
    """Assessment of document relevance to query"""
    grade: RelevanceGrade
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


class FactualityAssessment(BaseModel):
    """Assessment of response factuality"""
    grade: FactualityGrade
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    citations_found: bool = False


class CRAGRelevanceChecker:
    """
    CRAG-style relevance checker for retrieved documents.
    Implements corrective retrieval when documents are not relevant.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize CRAG relevance checker.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.llm = get_llm(config.llm_model, 0.0)
        
        # Relevance grading prompt
        self.relevance_prompt = PromptTemplate(
            input_variables=["question", "document"],
            template="""You are a grader assessing the relevance of a retrieved document to a user question.

Here is the retrieved document:
{document}

Here is the user question:
{question}

If the document contains keywords related to the user question, grade it as relevant.
If the document does not contain keywords related to the user question, grade it as irrelevant.
If the document is somewhat related but doesn't directly answer the question, grade it as ambiguous.

Give a binary score 'relevant', 'irrelevant', or 'ambiguous' to indicate whether the document is relevant to the question.
Also provide a confidence score between 0 and 1, and brief reasoning.

Format your response as:
Grade: [relevant/irrelevant/ambiguous]
Confidence: [0.0-1.0]
Reasoning: [brief explanation]"""
        )
        
        self.output_parser = StrOutputParser()
        
        # Create grading chain
        self.grading_chain = (
            self.relevance_prompt
            | self.llm
            | self.output_parser
        )
        
        logger.info("CRAGRelevanceChecker initialized")
    
    def assess_relevance(self, query: str, document: Document) -> RelevanceAssessment:
        """
        Assess the relevance of a document to a query.
        
        Args:
            query: User query
            document: Document to assess
            
        Returns:
            RelevanceAssessment: Assessment result
        """
        try:
            # Invoke the grading chain
            response = self.grading_chain.invoke({
                "question": query,
                "document": document.content
            })
            
            # Parse the response
            assessment = self._parse_relevance_response(response)
            
            logger.debug(f"Document relevance: {assessment.grade} (confidence: {assessment.confidence})")
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing document relevance: {str(e)}")
            # Return default assessment on error
            return RelevanceAssessment(
                grade=RelevanceGrade.AMBIGUOUS,
                confidence=0.5,
                reasoning="Error occurred during assessment"
            )
    
    def _parse_relevance_response(self, response: str) -> RelevanceAssessment:
        """Parse the LLM response into a RelevanceAssessment"""
        lines = response.strip().split('\n')
        
        grade = RelevanceGrade.AMBIGUOUS
        confidence = 0.5
        reasoning = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith("Grade:"):
                grade_str = line.split(":", 1)[1].strip().lower()
                if grade_str in ["relevant", "irrelevant", "ambiguous"]:
                    grade = RelevanceGrade(grade_str)
            elif line.startswith("Confidence:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                    confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
                except ValueError:
                    confidence = 0.5
            elif line.startswith("Reasoning:"):
                reasoning = line.split(":", 1)[1].strip()
        
        return RelevanceAssessment(
            grade=grade,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def filter_relevant_documents(self, query: str, documents: List[Document], 
                                min_confidence: float = 0.7) -> Tuple[List[Document], List[RelevanceAssessment]]:
        """
        Filter documents based on relevance assessment.
        
        Args:
            query: User query
            documents: List of documents to filter
            min_confidence: Minimum confidence threshold for relevance
            
        Returns:
            Tuple[List[Document], List[RelevanceAssessment]]: Filtered documents and assessments
        """
        relevant_docs = []
        assessments = []
        
        for doc in documents:
            assessment = self.assess_relevance(query, doc)
            assessments.append(assessment)
            
            # Keep document if it's relevant with sufficient confidence
            if (assessment.grade == RelevanceGrade.RELEVANT and 
                assessment.confidence >= min_confidence):
                relevant_docs.append(doc)
            # Also keep ambiguous documents with high confidence
            elif (assessment.grade == RelevanceGrade.AMBIGUOUS and 
                  assessment.confidence >= min_confidence):
                relevant_docs.append(doc)
        
        logger.info(f"Filtered {len(relevant_docs)} relevant documents from {len(documents)} total")
        return relevant_docs, assessments


class SelfRAGValidator:
    """
    Self-RAG validator for generated responses.
    Validates responses for factual grounding and consistency.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize Self-RAG validator.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.llm = get_llm(config.llm_model, 0.0)
        
        # Factuality grading prompt
        self.factuality_prompt = PromptTemplate(
            input_variables=["question", "context", "answer"],
            template="""You are a grader assessing whether an answer is grounded in / supported by a set of retrieved facts.

Here are the retrieved facts:
{context}

Here is the answer:
{answer}

Here is the user question:
{question}

Determine if the answer is grounded in the provided facts. Look for:
1. Whether the answer's claims are supported by the retrieved facts
2. Whether the answer contains information not present in the facts (hallucination)
3. Whether the answer appropriately cites or references the source material

Grade the answer as:
- 'grounded': The answer is fully supported by the retrieved facts
- 'not_grounded': The answer contains claims not supported by the facts
- 'partially_grounded': The answer is mostly supported but contains some unsupported claims

Also provide a confidence score between 0 and 1, reasoning, and whether citations are present.

Format your response as:
Grade: [grounded/not_grounded/partially_grounded]
Confidence: [0.0-1.0]
Citations: [true/false]
Reasoning: [brief explanation]"""
        )
        
        self.output_parser = StrOutputParser()
        
        # Create validation chain
        self.validation_chain = (
            self.factuality_prompt
            | self.llm
            | self.output_parser
        )
        
        logger.info("SelfRAGValidator initialized")
    
    def validate_response(self, query: str, context: List[Document], 
                         response: str) -> FactualityAssessment:
        """
        Validate a generated response against the provided context.
        
        Args:
            query: Original user query
            context: Retrieved documents used for generation
            response: Generated response to validate
            
        Returns:
            FactualityAssessment: Validation result
        """
        try:
            # Format context
            formatted_context = "\n\n".join([doc.content for doc in context])
            
            # Invoke the validation chain
            validation_response = self.validation_chain.invoke({
                "question": query,
                "context": formatted_context,
                "answer": response
            })
            
            # Parse the response
            assessment = self._parse_factuality_response(validation_response)
            
            logger.debug(f"Response validation: {assessment.grade} (confidence: {assessment.confidence})")
            return assessment
            
        except Exception as e:
            logger.error(f"Error validating response: {str(e)}")
            # Return default assessment on error
            return FactualityAssessment(
                grade=FactualityGrade.PARTIALLY_GROUNDED,
                confidence=0.5,
                reasoning="Error occurred during validation"
            )
    
    def _parse_factuality_response(self, response: str) -> FactualityAssessment:
        """Parse the LLM response into a FactualityAssessment"""
        lines = response.strip().split('\n')
        
        grade = FactualityGrade.PARTIALLY_GROUNDED
        confidence = 0.5
        citations_found = False
        reasoning = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith("Grade:"):
                grade_str = line.split(":", 1)[1].strip().lower()
                if grade_str in ["grounded", "not_grounded", "partially_grounded"]:
                    grade = FactualityGrade(grade_str)
            elif line.startswith("Confidence:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                    confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
                except ValueError:
                    confidence = 0.5
            elif line.startswith("Citations:"):
                citations_str = line.split(":", 1)[1].strip().lower()
                citations_found = citations_str in ["true", "yes", "1"]
            elif line.startswith("Reasoning:"):
                reasoning = line.split(":", 1)[1].strip()
        
        return FactualityAssessment(
            grade=grade,
            confidence=confidence,
            reasoning=reasoning,
            citations_found=citations_found
        )


class SelfCorrectionEngine:
    """
    Main self-correction engine that orchestrates CRAG and Self-RAG mechanisms.
    Implements fallback strategies for low-confidence results.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize self-correction engine.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.relevance_checker = CRAGRelevanceChecker(config)
        self.response_validator = SelfRAGValidator(config)
        
        # Thresholds for correction decisions
        self.relevance_threshold = getattr(config, 'relevance_threshold', 0.7)
        self.factuality_threshold = getattr(config, 'factuality_threshold', 0.7)
        self.min_relevant_docs = getattr(config, 'min_relevant_docs', 2)
        
        logger.info("SelfCorrectionEngine initialized")
    
    def correct_retrieval(self, query: str, documents: List[Document]) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Apply CRAG-style correction to retrieved documents.
        
        Args:
            query: User query
            documents: Retrieved documents to correct
            
        Returns:
            Tuple[List[Document], Dict[str, Any]]: Corrected documents and metadata
        """
        if not documents:
            logger.warning("No documents provided for correction")
            return [], {"correction_applied": False, "reason": "no_documents"}
        
        try:
            # Filter documents based on relevance
            relevant_docs, assessments = self.relevance_checker.filter_relevant_documents(
                query, documents, self.relevance_threshold
            )
            
            correction_metadata = {
                "original_count": len(documents),
                "filtered_count": len(relevant_docs),
                "correction_applied": len(relevant_docs) < len(documents),
                "assessments": [
                    {
                        "grade": assessment.grade.value,
                        "confidence": assessment.confidence,
                        "reasoning": assessment.reasoning
                    }
                    for assessment in assessments
                ],
                "fallback_triggered": False
            }
            
            # Check if we have enough relevant documents
            if len(relevant_docs) < self.min_relevant_docs:
                logger.warning(f"Only {len(relevant_docs)} relevant documents found, below threshold of {self.min_relevant_docs}")
                correction_metadata["fallback_triggered"] = True
                correction_metadata["fallback_reason"] = "insufficient_relevant_docs"
                
                # Fallback strategy: return original documents with warning
                # In a full implementation, this could trigger web search or alternative retrieval
                return documents, correction_metadata
            
            logger.info(f"CRAG correction: {len(documents)} -> {len(relevant_docs)} documents")
            return relevant_docs, correction_metadata
            
        except Exception as e:
            logger.error(f"Error in retrieval correction: {str(e)}")
            return documents, {"correction_applied": False, "error": str(e)}
    
    def validate_generation(self, query: str, context: List[Document], 
                          response: str) -> Tuple[str, Dict[str, Any]]:
        """
        Apply Self-RAG validation to generated response.
        
        Args:
            query: User query
            context: Context documents used for generation
            response: Generated response to validate
            
        Returns:
            Tuple[str, Dict[str, Any]]: Validated/corrected response and metadata
        """
        if not response.strip():
            logger.warning("Empty response provided for validation")
            return response, {"validation_applied": False, "reason": "empty_response"}
        
        try:
            # Validate the response
            assessment = self.response_validator.validate_response(query, context, response)
            
            validation_metadata = {
                "grade": assessment.grade.value,
                "confidence": assessment.confidence,
                "reasoning": assessment.reasoning,
                "citations_found": assessment.citations_found,
                "validation_applied": True,
                "correction_needed": False
            }
            
            # Check if correction is needed
            if (assessment.grade == FactualityGrade.NOT_GROUNDED or 
                (assessment.grade == FactualityGrade.PARTIALLY_GROUNDED and 
                 assessment.confidence < self.factuality_threshold)):
                
                validation_metadata["correction_needed"] = True
                
                # Apply correction strategy
                corrected_response = self._apply_response_correction(
                    query, context, response, assessment
                )
                
                logger.info("Self-RAG correction applied to response")
                return corrected_response, validation_metadata
            
            logger.info(f"Response validation passed: {assessment.grade.value}")
            return response, validation_metadata
            
        except Exception as e:
            logger.error(f"Error in response validation: {str(e)}")
            return response, {"validation_applied": False, "error": str(e)}
    
    def _apply_response_correction(self, query: str, context: List[Document], 
                                 response: str, assessment: FactualityAssessment) -> str:
        """
        Apply correction to a response that failed validation.
        
        Args:
            query: User query
            context: Context documents
            response: Original response
            assessment: Validation assessment
            
        Returns:
            str: Corrected response
        """
        try:
            # Simple correction strategy: add disclaimer and context reference
            if assessment.grade == FactualityGrade.NOT_GROUNDED:
                corrected_response = (
                    "Based on the provided context, I cannot fully verify all claims in the original response. "
                    f"Here's what I can confirm from the available information:\n\n{response}\n\n"
                    "Please note that some information may not be fully supported by the provided sources."
                )
            else:  # PARTIALLY_GROUNDED
                corrected_response = (
                    f"{response}\n\n"
                    "Note: This response is based on the provided context. "
                    "Some details may require additional verification."
                )
            
            return corrected_response
            
        except Exception as e:
            logger.error(f"Error applying response correction: {str(e)}")
            return response  # Return original on error
    
    def process_rag_pipeline(self, query: str, documents: List[Document], 
                           response: str) -> Tuple[List[Document], str, Dict[str, Any]]:
        """
        Apply full self-correction pipeline to RAG results.
        
        Args:
            query: User query
            documents: Retrieved documents
            response: Generated response
            
        Returns:
            Tuple[List[Document], str, Dict[str, Any]]: Corrected documents, response, and metadata
        """
        try:
            # Apply retrieval correction
            corrected_docs, retrieval_metadata = self.correct_retrieval(query, documents)
            
            # Apply response validation
            validated_response, validation_metadata = self.validate_generation(
                query, corrected_docs, response
            )
            
            # Combine metadata
            combined_metadata = {
                "self_correction_applied": True,
                "retrieval_correction": retrieval_metadata,
                "response_validation": validation_metadata,
                "total_corrections": (
                    int(retrieval_metadata.get("correction_applied", False)) +
                    int(validation_metadata.get("correction_needed", False))
                )
            }
            
            logger.info(f"Self-correction pipeline completed with {combined_metadata['total_corrections']} corrections")
            return corrected_docs, validated_response, combined_metadata
            
        except Exception as e:
            logger.error(f"Error in self-correction pipeline: {str(e)}")
            return documents, response, {"self_correction_applied": False, "error": str(e)}
    
    def get_correction_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the self-correction engine.
        
        Returns:
            Dict[str, Any]: Correction statistics
        """
        return {
            "relevance_threshold": self.relevance_threshold,
            "factuality_threshold": self.factuality_threshold,
            "min_relevant_docs": self.min_relevant_docs,
            "components": {
                "relevance_checker": "CRAGRelevanceChecker",
                "response_validator": "SelfRAGValidator"
            }
        }
    
    def update_thresholds(self, relevance_threshold: Optional[float] = None,
                         factuality_threshold: Optional[float] = None,
                         min_relevant_docs: Optional[int] = None) -> None:
        """
        Update correction thresholds.
        
        Args:
            relevance_threshold: New relevance threshold
            factuality_threshold: New factuality threshold
            min_relevant_docs: New minimum relevant documents threshold
        """
        if relevance_threshold is not None:
            self.relevance_threshold = max(0.0, min(1.0, relevance_threshold))
        
        if factuality_threshold is not None:
            self.factuality_threshold = max(0.0, min(1.0, factuality_threshold))
        
        if min_relevant_docs is not None:
            self.min_relevant_docs = max(1, min_relevant_docs)
        
        logger.info("Self-correction thresholds updated")