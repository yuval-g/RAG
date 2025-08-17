"""
Keyword-based retrieval using BM25 algorithm
"""

from typing import List, Optional, Tuple, Dict, Any
import logging
import math
from collections import Counter, defaultdict

from ..core.interfaces import BaseRetriever
from ..core.models import Document
from ..core.config import PipelineConfig


logger = logging.getLogger(__name__)


class KeywordRetriever(BaseRetriever):
    """
    Keyword-based retriever using BM25 algorithm for sparse retrieval.
    Complements vector search with exact keyword matching capabilities.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the keyword retriever.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self.documents: List[Document] = []
        self.document_index: Dict[str, Document] = {}
        self.term_frequencies: Dict[str, Dict[str, int]] = {}  # doc_id -> {term: frequency}
        self.document_frequencies: Dict[str, int] = {}  # term -> number of docs containing term
        self.document_lengths: Dict[str, int] = {}  # doc_id -> document length
        self.average_document_length: float = 0.0
        self.vocabulary: set = set()
        
        # BM25 parameters
        self.k1 = 1.5  # Term frequency saturation parameter
        self.b = 0.75  # Length normalization parameter
        
        logger.info("KeywordRetriever initialized")
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        Add documents to the keyword index.
        
        Args:
            documents: List of documents to index
            
        Returns:
            bool: True if successful
        """
        try:
            for doc in documents:
                doc_id = doc.doc_id or f"doc_{len(self.documents)}"
                doc.doc_id = doc_id
                
                self.documents.append(doc)
                self.document_index[doc_id] = doc
                
                # Tokenize and count terms
                terms = self._tokenize(doc.content)
                term_counts = Counter(terms)
                
                self.term_frequencies[doc_id] = dict(term_counts)
                self.document_lengths[doc_id] = len(terms)
                self.vocabulary.update(terms)
                
                # Update document frequencies
                for term in set(terms):
                    self.document_frequencies[term] = self.document_frequencies.get(term, 0) + 1
            
            # Update average document length
            if self.document_lengths:
                self.average_document_length = sum(self.document_lengths.values()) / len(self.document_lengths)
            
            logger.info(f"Added {len(documents)} documents to keyword index. "
                       f"Total documents: {len(self.documents)}, Vocabulary size: {len(self.vocabulary)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to keyword index: {str(e)}")
            return False
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization of text.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List[str]: List of tokens
        """
        # Simple tokenization: lowercase, split on whitespace and punctuation
        import re
        
        # Convert to lowercase and split on non-alphanumeric characters
        tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        
        # Filter out very short tokens
        tokens = [token for token in tokens if len(token) > 2]
        
        return tokens
    
    def _calculate_bm25_score(self, query_terms: List[str], doc_id: str) -> float:
        """
        Calculate BM25 score for a document given query terms.
        
        Args:
            query_terms: List of query terms
            doc_id: Document ID
            
        Returns:
            float: BM25 score
        """
        if doc_id not in self.term_frequencies:
            return 0.0
        
        score = 0.0
        doc_length = self.document_lengths.get(doc_id, 0)
        doc_term_freqs = self.term_frequencies[doc_id]
        
        for term in query_terms:
            if term not in self.vocabulary:
                continue
            
            # Term frequency in document
            tf = doc_term_freqs.get(term, 0)
            if tf == 0:
                continue
            
            # Document frequency (number of documents containing the term)
            df = self.document_frequencies.get(term, 0)
            if df == 0:
                continue
            
            # Inverse document frequency
            idf = math.log((len(self.documents) - df + 0.5) / (df + 0.5))
            
            # BM25 term score
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.average_document_length))
            
            term_score = idf * (numerator / denominator)
            score += term_score
        
        return score
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve documents using BM25 keyword matching.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            List[Document]: Retrieved documents
        """
        if not self.documents:
            logger.warning("No documents in keyword index")
            return []
        
        try:
            # Tokenize query
            query_terms = self._tokenize(query)
            if not query_terms:
                logger.warning("No valid query terms found")
                return []
            
            # Calculate BM25 scores for all documents
            doc_scores = []
            for doc_id in self.document_index.keys():
                score = self._calculate_bm25_score(query_terms, doc_id)
                if score > 0:
                    doc_scores.append((doc_id, score))
            
            # Sort by score and return top k
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            top_docs = doc_scores[:k]
            
            # Return documents
            retrieved_docs = [self.document_index[doc_id] for doc_id, _ in top_docs]
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents using keyword search")
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Error in keyword retrieval: {str(e)}")
            return []
    
    def retrieve_with_scores(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Retrieve documents with BM25 scores.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            List[Tuple[Document, float]]: List of (document, score) tuples
        """
        if not self.documents:
            logger.warning("No documents in keyword index")
            return []
        
        try:
            # Tokenize query
            query_terms = self._tokenize(query)
            if not query_terms:
                logger.warning("No valid query terms found")
                return []
            
            # Calculate BM25 scores for all documents
            doc_scores = []
            for doc_id in self.document_index.keys():
                score = self._calculate_bm25_score(query_terms, doc_id)
                if score > 0:
                    doc_scores.append((doc_id, score))
            
            # Sort by score and return top k
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            top_docs = doc_scores[:k]
            
            # Return documents with scores
            retrieved_docs_with_scores = [
                (self.document_index[doc_id], score) 
                for doc_id, score in top_docs
            ]
            
            logger.info(f"Retrieved {len(retrieved_docs_with_scores)} documents with scores using keyword search")
            return retrieved_docs_with_scores
            
        except Exception as e:
            logger.error(f"Error in keyword retrieval with scores: {str(e)}")
            return []
    
    def get_document_count(self) -> int:
        """Get the number of indexed documents"""
        return len(self.documents)
    
    def is_ready(self) -> bool:
        """Check if the retriever is ready for use"""
        return len(self.documents) > 0 and len(self.vocabulary) > 0
    
    def clear_index(self) -> bool:
        """Clear the keyword index"""
        try:
            self.documents.clear()
            self.document_index.clear()
            self.term_frequencies.clear()
            self.document_frequencies.clear()
            self.document_lengths.clear()
            self.vocabulary.clear()
            self.average_document_length = 0.0
            
            logger.info("Keyword index cleared")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing keyword index: {str(e)}")
            return False
    
    def get_vocabulary_size(self) -> int:
        """Get the size of the vocabulary"""
        return len(self.vocabulary)
    
    def get_term_statistics(self, term: str) -> Dict[str, Any]:
        """
        Get statistics for a specific term.
        
        Args:
            term: Term to get statistics for
            
        Returns:
            Dict[str, Any]: Term statistics
        """
        term = term.lower()
        
        if term not in self.vocabulary:
            return {"term": term, "in_vocabulary": False}
        
        df = self.document_frequencies.get(term, 0)
        idf = math.log((len(self.documents) - df + 0.5) / (df + 0.5)) if df > 0 else 0.0
        
        return {
            "term": term,
            "in_vocabulary": True,
            "document_frequency": df,
            "inverse_document_frequency": idf,
            "collection_frequency": sum(
                tf_dict.get(term, 0) for tf_dict in self.term_frequencies.values()
            )
        }
    
    def get_index_statistics(self) -> Dict[str, Any]:
        """Get overall index statistics"""
        return {
            "total_documents": len(self.documents),
            "vocabulary_size": len(self.vocabulary),
            "average_document_length": self.average_document_length,
            "total_terms": sum(self.document_lengths.values()),
            "bm25_parameters": {
                "k1": self.k1,
                "b": self.b
            }
        }
    
    def update_config(self, config: PipelineConfig) -> None:
        """
        Update the configuration.
        
        Args:
            config: New pipeline configuration
        """
        self.config = config
        logger.info("KeywordRetriever configuration updated")
    
    def set_bm25_parameters(self, k1: float = 1.5, b: float = 0.75) -> None:
        """
        Set BM25 parameters.
        
        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b
        logger.info(f"BM25 parameters updated: k1={k1}, b={b}")