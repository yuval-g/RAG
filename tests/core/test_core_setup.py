"""
Test the core setup and interfaces
"""

import pytest
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.rag_engine import RAGEngine, PipelineConfig, ConfigurationManager, Document


def test_pipeline_config_creation():
    """Test that PipelineConfig can be created with defaults"""
    config = PipelineConfig()
    
    assert config.llm_provider == "google"
    assert config.llm_model == "gemini-2.0-flash-lite"
    assert config.temperature == 0.0
    assert config.embedding_provider == "openai"
    assert config.vector_store == "chroma"
    assert config.chunk_size == 1000
    assert config.chunk_overlap == 200
    assert config.retrieval_k == 5


def test_pipeline_config_customization():
    """Test that PipelineConfig can be customized"""
    config = PipelineConfig(
        llm_provider="openai",
        llm_model="gpt-4",
        temperature=0.5,
        chunk_size=500,
        retrieval_k=10
    )
    
    assert config.llm_provider == "openai"
    assert config.llm_model == "gpt-4"
    assert config.temperature == 0.5
    assert config.chunk_size == 500
    assert config.retrieval_k == 10


def test_configuration_manager():
    """Test that ConfigurationManager can load default config"""
    config_manager = ConfigurationManager()
    config = config_manager.load_config()
    
    assert isinstance(config, PipelineConfig)
    assert config.llm_provider == "google"


def test_rag_engine_creation():
    """Test that RAGEngine can be created"""
    config = PipelineConfig()
    engine = RAGEngine(config)
    
    assert engine.config == config
    assert engine.config.llm_provider == "google"


def test_rag_engine_system_info():
    """Test that RAGEngine returns system info"""
    engine = RAGEngine()
    info = engine.get_system_info()
    
    assert "version" in info
    assert "config" in info
    assert "components" in info
    assert info["version"] == "0.1.0"


def test_document_creation():
    """Test that Document model works correctly"""
    doc = Document(
        content="This is a test document",
        metadata={"source": "test"},
        doc_id="test-1"
    )
    
    assert doc.content == "This is a test document"
    assert doc.metadata["source"] == "test"
    assert doc.doc_id == "test-1"
    assert doc.embedding is None


def test_rag_engine_add_documents():
    """Test that RAGEngine can validate documents"""
    engine = RAGEngine()
    
    documents = [
        Document(content="Document 1", doc_id="1"),
        Document(content="Document 2", doc_id="2")
    ]
    
    result = engine.add_documents(documents)
    assert result is True
    
    # Test with empty documents
    empty_docs = [Document(content="", doc_id="empty")]
    result = engine.add_documents(empty_docs)
    assert result is False


def test_rag_engine_query():
    """Test that RAGEngine can process queries"""
    engine = RAGEngine()
    
    response = engine.query("What is the capital of France?")
    
    assert response.answer is not None
    assert response.processing_time > 0
    assert "query" in response.metadata
    assert response.metadata["query"] == "What is the capital of France?"


if __name__ == "__main__":
    pytest.main([__file__])