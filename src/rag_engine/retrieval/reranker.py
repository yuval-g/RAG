"""
Document re-ranking implementation adapting from workplan/04AdvancedRetrieval-Generation.md
"""

from typing import List, Tuple, Optional, Dict, Any
import logging
from abc import ABC, abstractmethod

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

from ..core.interfaces import BaseReRanker
from ..core.models import Document
from ..core.config import PipelineConfig
from ..common.utils import get_llm


logger = logging.getLogger(__name__)


class BaseReRankerImpl(BaseReRanker):
    """Base implementation for document re-rankers"""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        logger.info(f"Initialized {self.__class__.__name__}")

    @abstractmethod
    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """Re-rank documents based on relevance to query"""
        pass

    @abstractmethod
    def rerank_with_scores(self, query: str, documents: List[Document], top_k: int = 5) -> List[Tuple[Document, float]]:
        """Re-rank documents with relevance scores"""
        pass


class LLMReRanker(BaseReRankerImpl):
    """
    LLM-based re-ranker using Google Gemini for document relevance scoring.
    Adapts the re-ranking concept from workplan/04AdvancedRetrieval-Generation.md
    
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        super().__init__(config)
        self.llm = get_llm(self.config.llm_model, 0.0)

        # Prompt template for relevance scoring
        self.scoring_prompt = PromptTemplate(
            input_variables=["query", "document"],
            template="""You are an expert at evaluating document relevance. Given a query and a document, 
rate how relevant the document is to answering the query on a scale of 0.0 to 1.0.

Query: {query}

Document: {document}

Consider:
- Does the document contain information that directly answers the query?
- How specific and detailed is the relevant information?
- Is the information accurate and up-to-date?

Provide only a numerical score between 0.0 and 1.0, where:
- 1.0 = Highly relevant, directly answers the query
- 0.7-0.9 = Very relevant, contains useful information
- 0.4-0.6 = Somewhat relevant, tangentially related
- 0.1-0.3 = Minimally relevant, barely related
- 0.0 = Not relevant at all

Score:"""
        )

        logger.info("LLMReRanker initialized with Google Gemini")

    def _score_document(self, query: str, document: Document) -> float:
        """Score a single document's relevance to the query"""
        try:
            prompt = self.scoring_prompt.format(
                query=query,
                document=document.content[:1000]  # Limit content length
            )

            response = self.llm.invoke(prompt)
            score_text = response.content.strip()

            # Extract numerical score
            try:
                score = float(score_text)
                return max(0.0, min(1.0, score))  # Clamp between 0 and 1
            except ValueError:
                logger.warning(f"Could not parse score: {score_text}")
                return 0.5  # Default score

        except Exception as e:
            logger.error(f"Error scoring document: {str(e)}")
            return 0.5  # Default score

    def rerank_with_scores(self, query: str, documents: List[Document], top_k: int = 5) -> List[Tuple[Document, float]]:
        """Re-rank documents with relevance scores"""
        if not documents:
            return []

        logger.info(f"Re-ranking {len(documents)} documents for query: '{query[:50]}...'")

        # Score all documents
        scored_documents = []
        for doc in documents:
            score = self._score_document(query, doc)
            scored_documents.append((doc, score))

        # Sort by score (descending)
        scored_documents.sort(key=lambda x: x[1], reverse=True)

        # Return top_k results
        result = scored_documents[:top_k]

        logger.info(f"Re-ranking complete. Top score: {result[0][1]:.4f}" if result else "No documents to rank")
        return result

    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """Re-rank documents based on relevance to query"""
        scored_docs = self.rerank_with_scores(query, documents, top_k)
        return [doc for doc, score in scored_docs]


class ContextualCompressionReRanker(BaseReRankerImpl):
    """
    Re-ranker using LangChain's ContextualCompressionRetriever approach.
    Adapts the compression retriever concept from workplan/04AdvancedRetrieval-Generation.md
    """

    def __init__(self, base_retriever, config: Optional[PipelineConfig] = None):
        super().__init__(config)
        self.base_retriever = base_retriever

        # Create LLM chain extractor using Google Gemini
        self.llm = get_llm(self.config.llm_model, 0.0)

        # Create compressor that extracts relevant parts
        self.compressor = LLMChainExtractor.from_llm(self.llm)

        # Create compression retriever
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=self.base_retriever
        )

        logger.info("ContextualCompressionReRanker initialized")

    def rerank_with_scores(self, query: str, documents: List[Document], top_k: int = 5) -> List[Tuple[Document, float]]:
        """Re-rank documents with relevance scores using contextual compression"""
        if not documents:
            return []

        try:
            # Convert our documents to LangChain format for compression
            from langchain.schema import Document as LangChainDocument
            langchain_docs = [
                LangChainDocument(
                    page_content=doc.content,
                    metadata=doc.metadata or {}
                )
                for doc in documents
            ]

            # Use the compressor to filter and rank documents
            compressed_docs = self.compressor.compress_documents(langchain_docs, query)

            # Convert back to our format with scores
            result = []
            for i, compressed_doc in enumerate(compressed_docs[:top_k]):
                # Create score based on position (higher position = lower score)
                score = 1.0 - (i * 0.1)  # Simple scoring based on order
                score = max(0.1, score)  # Minimum score of 0.1

                doc = Document(
                    content=compressed_doc.page_content,
                    metadata=compressed_doc.metadata,
                    doc_id=compressed_doc.metadata.get('doc_id', f'compressed_doc_{i}')
                )
                result.append((doc, score))

            logger.info(f"Contextual compression complete. Returned {len(result)} documents")
            return result

        except Exception as e:
            logger.error(f"Error in contextual compression: {str(e)}")
            # Fallback: return original documents with default scores
            return [(doc, 0.5) for doc in documents[:top_k]]

    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """Re-rank documents using contextual compression"""
        scored_docs = self.rerank_with_scores(query, documents, top_k)
        return [doc for doc, score in scored_docs]


class ReRanker:
    """
    Main re-ranker class that provides a unified interface for different re-ranking strategies.
    Adapts the re-ranking approach from workplan/04AdvancedRetrieval-Generation.md
    """

    def __init__(self, strategy: str = "llm", base_retriever=None, config: Optional[PipelineConfig] = None):
        """
        Initialize the re-ranker with specified strategy.

        Args:
            strategy: Re-ranking strategy ("llm" or "contextual")
            base_retriever: Base retriever for contextual compression (required for contextual strategy)
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self.strategy = strategy

        if strategy == "llm":
            self.reranker = LLMReRanker(config)
        elif strategy == "contextual":
            if base_retriever is None:
                raise ValueError("base_retriever is required for contextual re-ranking strategy")
            self.reranker = ContextualCompressionReRanker(base_retriever, config)
        else:
            raise ValueError(f"Unknown re-ranking strategy: {strategy}")

        logger.info(f"ReRanker initialized with strategy: {strategy}")

    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """
        Re-rank documents based on relevance to query.

        Args:
            query: Query string
            documents: List of documents to re-rank
            top_k: Number of top documents to return

        Returns:
            List[Document]: Re-ranked documents
        """
        return self.reranker.rerank(query, documents, top_k)

    def rerank_with_scores(self, query: str, documents: List[Document], top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        Re-rank documents with relevance scores.

        Args:
            query: Query string
            documents: List of documents to re-rank
            top_k: Number of top documents to return

        Returns:
            List[Tuple[Document, float]]: List of (document, score) tuples
        """
        return self.reranker.rerank_with_scores(query, documents, top_k)

    def get_strategy(self) -> str:
        """Get the current re-ranking strategy"""
        return self.strategy

    def set_config(self, config: PipelineConfig) -> None:
        """Update the configuration"""
        self.config = config
        if hasattr(self.reranker, 'config'):
            self.reranker.config = config
