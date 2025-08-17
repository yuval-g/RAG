"""
Utility functions for document processing and formatting
"""

import re
import hashlib
import uuid
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone
from .models import Document, RAGResponse, ProcessedQuery, EvaluationResult


def generate_doc_id(content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate a unique document ID based on content and metadata
    
    Args:
        content: Document content
        metadata: Optional metadata dictionary
        
    Returns:
        Unique document ID string
    """
    # Create a hash based on content and key metadata
    hash_input = content
    if metadata:
        # Sort metadata keys for consistent hashing
        sorted_metadata = sorted(metadata.items())
        hash_input += str(sorted_metadata)
    
    # Generate SHA-256 hash
    doc_hash = hashlib.sha256(hash_input.encode('utf-8')).hexdigest()
    return f"doc_{doc_hash[:16]}"


def clean_text(text: str) -> str:
    """
    Clean and normalize text content
    
    Args:
        text: Raw text content
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    return text


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks
    
    Args:
        text: Text to chunk
        chunk_size: Maximum size of each chunk
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text or chunk_size <= 0:
        return []
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # If this is not the last chunk, try to break at word boundary
        if end < len(text):
            # Look for the last space within the chunk
            last_space = text.rfind(' ', start, end)
            if last_space > start:
                end = last_space
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = max(start + 1, end - chunk_overlap)
        
        # Prevent infinite loop
        if start >= len(text):
            break
    
    return chunks


def create_document_from_text(
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    auto_clean: bool = True,
    auto_id: bool = True
) -> Document:
    """
    Create a Document object from text content
    
    Args:
        content: Text content
        metadata: Optional metadata dictionary
        auto_clean: Whether to automatically clean the text
        auto_id: Whether to automatically generate document ID
        
    Returns:
        Document object
    """
    if auto_clean:
        content = clean_text(content)
    
    if metadata is None:
        metadata = {}
    
    # Add creation timestamp
    metadata.setdefault('created_at', datetime.now(timezone.utc).isoformat())
    
    doc_id = None
    if auto_id:
        doc_id = generate_doc_id(content, metadata)
    
    return Document(
        content=content,
        metadata=metadata,
        doc_id=doc_id
    )


def create_documents_from_texts(
    texts: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    auto_clean: bool = True,
    auto_id: bool = True
) -> List[Document]:
    """
    Create multiple Document objects from text list
    
    Args:
        texts: List of text contents
        metadatas: Optional list of metadata dictionaries
        auto_clean: Whether to automatically clean the text
        auto_id: Whether to automatically generate document IDs
        
    Returns:
        List of Document objects
    """
    if metadatas is None:
        metadatas = [{}] * len(texts)
    elif len(metadatas) != len(texts):
        raise ValueError("Length of metadatas must match length of texts")
    
    documents = []
    for text, metadata in zip(texts, metadatas):
        doc = create_document_from_text(
            content=text,
            metadata=metadata.copy(),
            auto_clean=auto_clean,
            auto_id=auto_id
        )
        documents.append(doc)
    
    return documents


def format_documents_for_prompt(documents: List[Document], max_length: Optional[int] = None) -> str:
    """
    Format documents for use in LLM prompts
    
    Args:
        documents: List of documents to format
        max_length: Optional maximum length of formatted text
        
    Returns:
        Formatted document string
    """
    if not documents:
        return ""
    
    formatted_docs = []
    current_length = 0
    
    for i, doc in enumerate(documents):
        # Format document with index and content
        doc_text = f"Document {i + 1}:\n{doc.content}\n"
        
        # Add metadata if available
        if doc.metadata:
            relevant_metadata = {k: v for k, v in doc.metadata.items() 
                               if k not in ['embedding', 'created_at']}
            if relevant_metadata:
                doc_text += f"Metadata: {relevant_metadata}\n"
        
        doc_text += "\n"
        
        # Check length limit
        if max_length and current_length + len(doc_text) > max_length:
            break
        
        formatted_docs.append(doc_text)
        current_length += len(doc_text)
    
    return "".join(formatted_docs).strip()


def extract_document_metadata(document: Document, keys: List[str]) -> Dict[str, Any]:
    """
    Extract specific metadata keys from a document
    
    Args:
        document: Document to extract metadata from
        keys: List of metadata keys to extract
        
    Returns:
        Dictionary with extracted metadata
    """
    extracted = {}
    for key in keys:
        if key in document.metadata:
            extracted[key] = document.metadata[key]
    return extracted


def merge_documents(documents: List[Document], separator: str = "\n\n") -> Document:
    """
    Merge multiple documents into a single document
    
    Args:
        documents: List of documents to merge
        separator: String to use between document contents
        
    Returns:
        Merged document
    """
    if not documents:
        raise ValueError("Cannot merge empty document list")
    
    if len(documents) == 1:
        return documents[0]
    
    # Merge content
    merged_content = separator.join(doc.content for doc in documents)
    
    # Merge metadata
    merged_metadata = {}
    for doc in documents:
        for key, value in doc.metadata.items():
            if key in merged_metadata:
                # Handle conflicts by creating lists
                if not isinstance(merged_metadata[key], list):
                    merged_metadata[key] = [merged_metadata[key]]
                if value not in merged_metadata[key]:
                    merged_metadata[key].append(value)
            else:
                merged_metadata[key] = value
    
    # Add merge information
    merged_metadata['merged_from'] = [doc.doc_id for doc in documents if doc.doc_id]
    merged_metadata['merge_count'] = len(documents)
    
    return create_document_from_text(
        content=merged_content,
        metadata=merged_metadata,
        auto_clean=False,  # Assume content is already clean
        auto_id=True
    )


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple text similarity using Jaccard similarity
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
    
    # Convert to lowercase and split into words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Calculate Jaccard similarity
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)


def validate_document(document: Document) -> List[str]:
    """
    Validate a document and return list of validation errors
    
    Args:
        document: Document to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Check content
    if not document.content or not document.content.strip():
        errors.append("Document content cannot be empty")
    
    # Check content length
    if len(document.content) > 1_000_000:  # 1MB limit
        errors.append("Document content exceeds maximum length (1MB)")
    
    # Check metadata
    if not isinstance(document.metadata, dict):
        errors.append("Document metadata must be a dictionary")
    
    # Check doc_id format if present
    if document.doc_id and not re.match(r'^[a-zA-Z0-9_-]+$', document.doc_id):
        errors.append("Document ID contains invalid characters")
    
    # Check embedding format if present
    if document.embedding is not None:
        if not isinstance(document.embedding, list):
            errors.append("Document embedding must be a list")
        elif not all(isinstance(x, (int, float)) for x in document.embedding):
            errors.append("Document embedding must contain only numbers")
    
    return errors


def format_rag_response(response: RAGResponse, include_sources: bool = True) -> str:
    """
    Format a RAG response for display
    
    Args:
        response: RAG response to format
        include_sources: Whether to include source document information
        
    Returns:
        Formatted response string
    """
    formatted = f"Answer: {response.answer}\n"
    
    if response.confidence_score > 0:
        formatted += f"Confidence: {response.confidence_score:.2f}\n"
    
    if response.processing_time > 0:
        formatted += f"Processing Time: {response.processing_time:.3f}s\n"
    
    if include_sources and response.source_documents:
        formatted += f"\nSources ({len(response.source_documents)} documents):\n"
        for i, doc in enumerate(response.source_documents, 1):
            formatted += f"{i}. {doc.content[:100]}...\n"
            if doc.metadata:
                relevant_metadata = {k: v for k, v in doc.metadata.items() 
                                   if k not in ['embedding', 'created_at']}
                if relevant_metadata:
                    formatted += f"   Metadata: {relevant_metadata}\n"
    
    return formatted.strip()


def create_test_documents(count: int = 5) -> List[Document]:
    """
    Create test documents for development and testing
    
    Args:
        count: Number of test documents to create
        
    Returns:
        List of test documents
    """
    test_contents = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
        "Natural language processing enables computers to understand and process human language.",
        "Vector databases are specialized databases designed to store and query high-dimensional vectors.",
        "Retrieval-augmented generation combines information retrieval with text generation for better AI responses."
    ]
    
    documents = []
    for i in range(min(count, len(test_contents))):
        metadata = {
            'topic': ['programming', 'AI', 'NLP', 'databases', 'RAG'][i],
            'difficulty': 'beginner',
            'test_doc': True,
            'index': i
        }
        doc = create_document_from_text(
            content=test_contents[i],
            metadata=metadata
        )
        documents.append(doc)
    
    # If more documents requested, create variations
    while len(documents) < count:
        base_doc = documents[len(documents) % len(test_contents)]
        variation = create_document_from_text(
            content=f"Variation: {base_doc.content}",
            metadata={**base_doc.metadata, 'variation': True, 'index': len(documents)}
        )
        documents.append(variation)
    
    return documents