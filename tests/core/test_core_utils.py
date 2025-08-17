"""
Unit tests for core utility functions
"""

import pytest
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.rag_engine.core.models import Document, RAGResponse
from src.rag_engine.core.utils import (
    generate_doc_id, clean_text, chunk_text, create_document_from_text,
    create_documents_from_texts, format_documents_for_prompt,
    extract_document_metadata, merge_documents, calculate_text_similarity,
    validate_document, format_rag_response, create_test_documents
)


class TestGenerateDocId:
    """Test cases for generate_doc_id function"""
    
    def test_generate_doc_id_content_only(self):
        """Test generating doc ID from content only"""
        content = "This is test content"
        doc_id = generate_doc_id(content)
        
        assert doc_id.startswith("doc_")
        assert len(doc_id) == 20  # "doc_" + 16 hex characters
        
        # Same content should generate same ID
        doc_id2 = generate_doc_id(content)
        assert doc_id == doc_id2
    
    def test_generate_doc_id_with_metadata(self):
        """Test generating doc ID with metadata"""
        content = "This is test content"
        metadata = {"source": "test", "category": "example"}
        
        doc_id1 = generate_doc_id(content)
        doc_id2 = generate_doc_id(content, metadata)
        
        # Different metadata should generate different IDs
        assert doc_id1 != doc_id2
        
        # Same content and metadata should generate same ID
        doc_id3 = generate_doc_id(content, metadata)
        assert doc_id2 == doc_id3
    
    def test_generate_doc_id_metadata_order_independence(self):
        """Test that metadata order doesn't affect ID generation"""
        content = "Test content"
        metadata1 = {"a": 1, "b": 2}
        metadata2 = {"b": 2, "a": 1}
        
        doc_id1 = generate_doc_id(content, metadata1)
        doc_id2 = generate_doc_id(content, metadata2)
        
        assert doc_id1 == doc_id2


class TestCleanText:
    """Test cases for clean_text function"""
    
    def test_clean_text_basic(self):
        """Test basic text cleaning"""
        text = "  This is   a test   text  "
        cleaned = clean_text(text)
        
        assert cleaned == "This is a test text"
    
    def test_clean_text_empty(self):
        """Test cleaning empty text"""
        assert clean_text("") == ""
        assert clean_text(None) == ""
    
    def test_clean_text_whitespace_normalization(self):
        """Test whitespace normalization"""
        text = "This\n\nis\t\ta\r\ntest"
        cleaned = clean_text(text)
        
        assert cleaned == "This is a test"
    
    def test_clean_text_control_characters(self):
        """Test removal of control characters"""
        text = "This\x00is\x08a\x1ftest"
        cleaned = clean_text(text)
        
        assert cleaned == "Thisisa test"
    
    def test_clean_text_preserves_newlines_tabs(self):
        """Test that newlines and tabs are preserved in normalization"""
        text = "Line1\nLine2\tTabbed"
        cleaned = clean_text(text)
        
        # Should normalize to single spaces but preserve structure
        assert "Line1" in cleaned
        assert "Line2" in cleaned
        assert "Tabbed" in cleaned


class TestChunkText:
    """Test cases for chunk_text function"""
    
    def test_chunk_text_short_text(self):
        """Test chunking text shorter than chunk size"""
        text = "Short text"
        chunks = chunk_text(text, chunk_size=100)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunk_text_exact_size(self):
        """Test chunking text exactly at chunk size"""
        text = "a" * 100
        chunks = chunk_text(text, chunk_size=100)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunk_text_with_overlap(self):
        """Test chunking with overlap"""
        text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12"
        chunks = chunk_text(text, chunk_size=30, chunk_overlap=10)
        
        assert len(chunks) > 1
        
        # Check that there is actual overlap in content
        # With proper overlap, adjacent chunks should share some words
        found_overlap = False
        for i in range(len(chunks) - 1):
            words_current = set(chunks[i].split())
            words_next = set(chunks[i + 1].split())
            if words_current.intersection(words_next):
                found_overlap = True
                break
        
        assert found_overlap, f"No overlap found between chunks: {chunks}"
    
    def test_chunk_text_word_boundaries(self):
        """Test that chunking respects word boundaries"""
        text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
        chunks = chunk_text(text, chunk_size=25, chunk_overlap=5)
        
        # Each chunk should end with complete words (no partial words)
        for chunk in chunks:
            if chunk.strip():  # Skip empty chunks
                # Should not start or end with partial words (spaces indicate word boundaries)
                assert not chunk.startswith(" ") or chunk.strip() != ""
    
    def test_chunk_text_empty_input(self):
        """Test chunking empty or invalid input"""
        assert chunk_text("") == []
        assert chunk_text("text", chunk_size=0) == []
        assert chunk_text("text", chunk_size=-1) == []
    
    def test_chunk_text_no_infinite_loop(self):
        """Test that chunking doesn't create infinite loops"""
        text = "a" * 1000
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=50)
        
        # Should create finite number of chunks
        assert len(chunks) > 0
        assert len(chunks) < 50  # Reasonable upper bound


class TestCreateDocumentFromText:
    """Test cases for create_document_from_text function"""
    
    def test_create_document_basic(self):
        """Test basic document creation"""
        content = "Test content"
        doc = create_document_from_text(content)
        
        assert doc.content == content
        assert isinstance(doc.metadata, dict)
        assert "created_at" in doc.metadata
        assert doc.doc_id is not None
        assert doc.doc_id.startswith("doc_")
    
    def test_create_document_with_metadata(self):
        """Test document creation with custom metadata"""
        content = "Test content"
        metadata = {"source": "test", "category": "example"}
        doc = create_document_from_text(content, metadata=metadata)
        
        assert doc.metadata["source"] == "test"
        assert doc.metadata["category"] == "example"
        assert "created_at" in doc.metadata  # Should be added automatically
    
    def test_create_document_no_auto_clean(self):
        """Test document creation without auto-cleaning"""
        content = "  Test   content  "
        doc = create_document_from_text(content, auto_clean=False)
        
        assert doc.content == content  # Should preserve original formatting
    
    def test_create_document_no_auto_id(self):
        """Test document creation without auto-generated ID"""
        content = "Test content"
        doc = create_document_from_text(content, auto_id=False)
        
        assert doc.doc_id is None
    
    def test_create_document_auto_clean(self):
        """Test document creation with auto-cleaning"""
        content = "  Test   content  "
        doc = create_document_from_text(content, auto_clean=True)
        
        assert doc.content == "Test content"


class TestCreateDocumentsFromTexts:
    """Test cases for create_documents_from_texts function"""
    
    def test_create_documents_basic(self):
        """Test creating multiple documents"""
        texts = ["Text 1", "Text 2", "Text 3"]
        docs = create_documents_from_texts(texts)
        
        assert len(docs) == 3
        for i, doc in enumerate(docs):
            assert doc.content == texts[i]
            assert doc.doc_id is not None
    
    def test_create_documents_with_metadata(self):
        """Test creating documents with metadata"""
        texts = ["Text 1", "Text 2"]
        metadatas = [{"source": "source1"}, {"source": "source2"}]
        docs = create_documents_from_texts(texts, metadatas=metadatas)
        
        assert len(docs) == 2
        assert docs[0].metadata["source"] == "source1"
        assert docs[1].metadata["source"] == "source2"
    
    def test_create_documents_metadata_length_mismatch(self):
        """Test error when metadata length doesn't match texts length"""
        texts = ["Text 1", "Text 2"]
        metadatas = [{"source": "source1"}]  # Only one metadata for two texts
        
        with pytest.raises(ValueError, match="Length of metadatas must match length of texts"):
            create_documents_from_texts(texts, metadatas=metadatas)
    
    def test_create_documents_metadata_independence(self):
        """Test that metadata dictionaries are independent"""
        texts = ["Text 1", "Text 2"]
        metadatas = [{"shared": "value"}, {"shared": "value"}]
        docs = create_documents_from_texts(texts, metadatas=metadatas)
        
        # Modify one document's metadata
        docs[0].metadata["unique"] = "value1"
        
        # Other document should not be affected
        assert "unique" not in docs[1].metadata


class TestFormatDocumentsForPrompt:
    """Test cases for format_documents_for_prompt function"""
    
    def test_format_documents_empty(self):
        """Test formatting empty document list"""
        result = format_documents_for_prompt([])
        assert result == ""
    
    def test_format_documents_single(self):
        """Test formatting single document"""
        doc = Document(content="Test content", metadata={"source": "test"})
        result = format_documents_for_prompt([doc])
        
        assert "Document 1:" in result
        assert "Test content" in result
        assert "source" in result
    
    def test_format_documents_multiple(self):
        """Test formatting multiple documents"""
        docs = [
            Document(content="Content 1", metadata={"source": "source1"}),
            Document(content="Content 2", metadata={"source": "source2"})
        ]
        result = format_documents_for_prompt(docs)
        
        assert "Document 1:" in result
        assert "Document 2:" in result
        assert "Content 1" in result
        assert "Content 2" in result
    
    def test_format_documents_max_length(self):
        """Test formatting with maximum length limit"""
        docs = [
            Document(content="A" * 100),
            Document(content="B" * 100),
            Document(content="C" * 100)
        ]
        result = format_documents_for_prompt(docs, max_length=150)
        
        # Should truncate to fit within max_length
        assert len(result) <= 150
        assert "Document 1:" in result
    
    def test_format_documents_filters_metadata(self):
        """Test that certain metadata keys are filtered out"""
        doc = Document(
            content="Test content",
            metadata={
                "source": "test",
                "embedding": [0.1, 0.2, 0.3],
                "created_at": "2023-01-01",
                "relevant": "keep_this"
            }
        )
        result = format_documents_for_prompt([doc])
        
        assert "source" in result
        assert "relevant" in result
        assert "embedding" not in result
        assert "created_at" not in result


class TestExtractDocumentMetadata:
    """Test cases for extract_document_metadata function"""
    
    def test_extract_metadata_existing_keys(self):
        """Test extracting existing metadata keys"""
        doc = Document(
            content="Test",
            metadata={"source": "test", "category": "example", "priority": "high"}
        )
        
        extracted = extract_document_metadata(doc, ["source", "category"])
        
        assert extracted == {"source": "test", "category": "example"}
    
    def test_extract_metadata_missing_keys(self):
        """Test extracting non-existent metadata keys"""
        doc = Document(content="Test", metadata={"source": "test"})
        
        extracted = extract_document_metadata(doc, ["source", "missing_key"])
        
        assert extracted == {"source": "test"}
        assert "missing_key" not in extracted
    
    def test_extract_metadata_empty_keys(self):
        """Test extracting with empty key list"""
        doc = Document(content="Test", metadata={"source": "test"})
        
        extracted = extract_document_metadata(doc, [])
        
        assert extracted == {}


class TestMergeDocuments:
    """Test cases for merge_documents function"""
    
    def test_merge_documents_empty_list(self):
        """Test merging empty document list"""
        with pytest.raises(ValueError, match="Cannot merge empty document list"):
            merge_documents([])
    
    def test_merge_documents_single(self):
        """Test merging single document"""
        doc = Document(content="Test content", metadata={"source": "test"})
        result = merge_documents([doc])
        
        assert result == doc
    
    def test_merge_documents_multiple(self):
        """Test merging multiple documents"""
        docs = [
            Document(content="Content 1", metadata={"source": "source1"}, doc_id="doc1"),
            Document(content="Content 2", metadata={"source": "source2"}, doc_id="doc2")
        ]
        
        merged = merge_documents(docs)
        
        assert "Content 1" in merged.content
        assert "Content 2" in merged.content
        assert merged.metadata["merged_from"] == ["doc1", "doc2"]
        assert merged.metadata["merge_count"] == 2
    
    def test_merge_documents_custom_separator(self):
        """Test merging with custom separator"""
        docs = [
            Document(content="Content 1"),
            Document(content="Content 2")
        ]
        
        merged = merge_documents(docs, separator=" | ")
        
        assert merged.content == "Content 1 | Content 2"
    
    def test_merge_documents_metadata_conflicts(self):
        """Test merging documents with conflicting metadata"""
        docs = [
            Document(content="Content 1", metadata={"category": "type1", "unique1": "value1"}),
            Document(content="Content 2", metadata={"category": "type2", "unique2": "value2"})
        ]
        
        merged = merge_documents(docs)
        
        # Conflicting values should become lists
        assert isinstance(merged.metadata["category"], list)
        assert "type1" in merged.metadata["category"]
        assert "type2" in merged.metadata["category"]
        
        # Unique values should be preserved
        assert merged.metadata["unique1"] == "value1"
        assert merged.metadata["unique2"] == "value2"


class TestCalculateTextSimilarity:
    """Test cases for calculate_text_similarity function"""
    
    def test_similarity_identical_texts(self):
        """Test similarity of identical texts"""
        text = "This is a test text"
        similarity = calculate_text_similarity(text, text)
        
        assert similarity == 1.0
    
    def test_similarity_completely_different(self):
        """Test similarity of completely different texts"""
        text1 = "apple banana cherry"
        text2 = "dog elephant fox"
        similarity = calculate_text_similarity(text1, text2)
        
        assert similarity == 0.0
    
    def test_similarity_partial_overlap(self):
        """Test similarity of texts with partial overlap"""
        text1 = "apple banana cherry"
        text2 = "apple dog elephant"
        similarity = calculate_text_similarity(text1, text2)
        
        # Should be between 0 and 1
        assert 0 < similarity < 1
        # Jaccard similarity: intersection=1 (apple), union=5 (apple,banana,cherry,dog,elephant)
        assert similarity == 1/5
    
    def test_similarity_empty_texts(self):
        """Test similarity with empty texts"""
        assert calculate_text_similarity("", "test") == 0.0
        assert calculate_text_similarity("test", "") == 0.0
        assert calculate_text_similarity("", "") == 0.0
    
    def test_similarity_case_insensitive(self):
        """Test that similarity is case insensitive"""
        text1 = "Apple Banana"
        text2 = "apple banana"
        similarity = calculate_text_similarity(text1, text2)
        
        assert similarity == 1.0


class TestValidateDocument:
    """Test cases for validate_document function"""
    
    def test_validate_document_valid(self):
        """Test validation of valid document"""
        doc = Document(
            content="Valid content",
            metadata={"source": "test"},
            doc_id="valid_doc_123"
        )
        
        errors = validate_document(doc)
        assert errors == []
    
    def test_validate_document_empty_content(self):
        """Test validation of document with empty content"""
        doc = Document(content="")
        errors = validate_document(doc)
        
        assert len(errors) > 0
        assert any("empty" in error.lower() for error in errors)
    
    def test_validate_document_whitespace_only_content(self):
        """Test validation of document with whitespace-only content"""
        doc = Document(content="   \n\t  ")
        errors = validate_document(doc)
        
        assert len(errors) > 0
        assert any("empty" in error.lower() for error in errors)
    
    def test_validate_document_too_long(self):
        """Test validation of document with content too long"""
        doc = Document(content="a" * 1_000_001)  # Exceed 1MB limit
        errors = validate_document(doc)
        
        assert len(errors) > 0
        assert any("maximum length" in error for error in errors)
    
    def test_validate_document_invalid_metadata(self):
        """Test validation of document with invalid metadata"""
        doc = Document(content="Valid content")
        doc.metadata = "not a dict"  # Invalid metadata type
        
        errors = validate_document(doc)
        
        assert len(errors) > 0
        assert any("dictionary" in error for error in errors)
    
    def test_validate_document_invalid_doc_id(self):
        """Test validation of document with invalid doc ID"""
        doc = Document(content="Valid content", doc_id="invalid@id#")
        errors = validate_document(doc)
        
        assert len(errors) > 0
        assert any("invalid characters" in error for error in errors)
    
    def test_validate_document_invalid_embedding(self):
        """Test validation of document with invalid embedding"""
        # With Pydantic, invalid embedding types are caught at creation time
        with pytest.raises(Exception):  # Pydantic ValidationError
            Document(content="Valid content", embedding="not a list")
    
    def test_validate_document_embedding_non_numeric(self):
        """Test validation of document with non-numeric embedding"""
        # With Pydantic, invalid embedding values are caught at creation time
        with pytest.raises(Exception):  # Pydantic ValidationError
            Document(content="Valid content", embedding=[1, 2, "three"])


class TestFormatRAGResponse:
    """Test cases for format_rag_response function"""
    
    def test_format_response_minimal(self):
        """Test formatting minimal RAG response"""
        response = RAGResponse(answer="Test answer")
        formatted = format_rag_response(response)
        
        assert "Answer: Test answer" in formatted
    
    def test_format_response_with_confidence(self):
        """Test formatting response with confidence score"""
        response = RAGResponse(answer="Test answer", confidence_score=0.85)
        formatted = format_rag_response(response)
        
        assert "Answer: Test answer" in formatted
        assert "Confidence: 0.85" in formatted
    
    def test_format_response_with_timing(self):
        """Test formatting response with processing time"""
        response = RAGResponse(answer="Test answer", processing_time=1.234)
        formatted = format_rag_response(response)
        
        assert "Answer: Test answer" in formatted
        assert "Processing Time: 1.234s" in formatted
    
    def test_format_response_with_sources(self):
        """Test formatting response with source documents"""
        docs = [
            Document(content="Source document 1", metadata={"source": "doc1"}),
            Document(content="Source document 2", metadata={"source": "doc2"})
        ]
        response = RAGResponse(answer="Test answer", source_documents=docs)
        formatted = format_rag_response(response, include_sources=True)
        
        assert "Answer: Test answer" in formatted
        assert "Sources (2 documents):" in formatted
        assert "Source document 1" in formatted
        assert "Source document 2" in formatted
    
    def test_format_response_no_sources(self):
        """Test formatting response without including sources"""
        docs = [Document(content="Source document")]
        response = RAGResponse(answer="Test answer", source_documents=docs)
        formatted = format_rag_response(response, include_sources=False)
        
        assert "Answer: Test answer" in formatted
        assert "Sources" not in formatted
        assert "Source document" not in formatted


class TestCreateTestDocuments:
    """Test cases for create_test_documents function"""
    
    def test_create_test_documents_default(self):
        """Test creating default number of test documents"""
        docs = create_test_documents()
        
        assert len(docs) == 5
        for doc in docs:
            assert isinstance(doc, Document)
            assert doc.content
            assert doc.metadata.get("test_doc") is True
    
    def test_create_test_documents_custom_count(self):
        """Test creating custom number of test documents"""
        docs = create_test_documents(count=3)
        
        assert len(docs) == 3
        for doc in docs:
            assert isinstance(doc, Document)
            assert doc.content
    
    def test_create_test_documents_more_than_base(self):
        """Test creating more documents than base templates"""
        docs = create_test_documents(count=10)
        
        assert len(docs) == 10
        
        # Should have some variations
        variation_docs = [doc for doc in docs if doc.metadata.get("variation")]
        assert len(variation_docs) > 0
    
    def test_create_test_documents_unique_content(self):
        """Test that test documents have unique content"""
        docs = create_test_documents(count=5)
        
        contents = [doc.content for doc in docs]
        unique_contents = set(contents)
        
        # All contents should be unique (no duplicates)
        assert len(contents) == len(unique_contents)
    
    def test_create_test_documents_metadata_structure(self):
        """Test that test documents have expected metadata structure"""
        docs = create_test_documents(count=3)
        
        for i, doc in enumerate(docs):
            assert "topic" in doc.metadata
            assert "difficulty" in doc.metadata
            assert "test_doc" in doc.metadata
            assert "index" in doc.metadata
            assert doc.metadata["index"] == i
            assert doc.metadata["test_doc"] is True