"""
Unit tests for RAPTORIndexer
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from src.rag_engine.indexing.raptor_indexer import RAPTORIndexer, RAPTORNode
from src.rag_engine.core.models import Document
from src.rag_engine.core.config import PipelineConfig


@pytest.fixture
def config():
    """Create a test configuration"""
    return PipelineConfig(
        llm_model="gemini-2.0-flash-lite",
        embedding_model="models/embedding-001",
        temperature=0.0,
        chunk_size=1000,
        chunk_overlap=200,
        vector_store_config={"persist_directory": None}
    )


@pytest.fixture
def sample_documents():
    """Create sample documents for testing"""
    return [
        Document(
            content="This is a document about artificial intelligence and machine learning. "
                   "It covers various topics including neural networks, deep learning, and natural language processing.",
            metadata={"title": "AI Overview", "author": "Test Author"},
            doc_id="doc1"
        ),
        Document(
            content="This document discusses the history of computer science and programming languages. "
                   "It includes information about early computers, programming paradigms, and software development.",
            metadata={"title": "CS History", "author": "Another Author"},
            doc_id="doc2"
        ),
        Document(
            content="Machine learning algorithms are used to analyze data and make predictions. "
                   "Common algorithms include linear regression, decision trees, and neural networks.",
            metadata={"title": "ML Algorithms", "author": "ML Expert"},
            doc_id="doc3"
        ),
        Document(
            content="Data science involves extracting insights from large datasets using statistical methods. "
                   "It combines domain expertise, programming skills, and statistical knowledge.",
            metadata={"title": "Data Science", "author": "Data Scientist"},
            doc_id="doc4"
        )
    ]


class TestRAPTORNode:
    """Test cases for RAPTORNode dataclass"""
    
    def test_raptor_node_creation(self):
        """Test RAPTORNode creation"""
        node = RAPTORNode(
            node_id="test_id",
            content="Test content",
            summary="Test summary",
            embedding=[0.1, 0.2, 0.3],
            level=0,
            children=["child1", "child2"],
            parent="parent_id",
            metadata={"key": "value"}
        )
        
        assert node.node_id == "test_id"
        assert node.content == "Test content"
        assert node.summary == "Test summary"
        assert node.embedding == [0.1, 0.2, 0.3]
        assert node.level == 0
        assert node.children == ["child1", "child2"]
        assert node.parent == "parent_id"
        assert node.metadata == {"key": "value"}


class TestRAPTORIndexer:
    """Test cases for RAPTORIndexer"""
    
    @patch('src.rag_engine.indexing.raptor_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.raptor_indexer.ChatGoogleGenerativeAI')
    def test_initialization(self, mock_llm, mock_embeddings, config):
        """Test RAPTORIndexer initialization"""
        indexer = RAPTORIndexer(config, max_levels=3, cluster_size_threshold=5)
        
        # Check that components are initialized
        assert indexer.config == config
        assert indexer.max_levels == 3
        assert indexer.cluster_size_threshold == 5
        assert indexer._document_count == 0
        assert len(indexer.nodes) == 0
        assert len(indexer.root_nodes) == 0
        assert len(indexer.level_nodes) == 0
        assert len(indexer.vectorstores) == 0
        
        # Check that Google services are initialized
        mock_embeddings.assert_called_once()
        mock_llm.assert_called_once()
    
    @patch('src.rag_engine.indexing.raptor_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.raptor_indexer.ChatGoogleGenerativeAI')
    def test_initialization_default_params(self, mock_llm, mock_embeddings, config):
        """Test RAPTORIndexer initialization with default parameters"""
        indexer = RAPTORIndexer(config)
        
        assert indexer.max_levels == 3
        assert indexer.cluster_size_threshold == 5
    
    @patch('src.rag_engine.indexing.raptor_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.raptor_indexer.ChatGoogleGenerativeAI')
    @patch('src.rag_engine.indexing.raptor_indexer.Chroma')
    def test_create_leaf_nodes(self, mock_chroma, mock_llm, mock_embeddings, config, sample_documents):
        """Test creation of leaf nodes from documents"""
        mock_embeddings_instance = Mock()
        mock_embeddings_instance.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_embeddings.return_value = mock_embeddings_instance
        
        indexer = RAPTORIndexer(config)
        
        leaf_nodes = indexer._create_leaf_nodes(sample_documents)
        
        assert len(leaf_nodes) == 4
        assert all(node.level == 0 for node in leaf_nodes)
        assert all(len(node.children) == 0 for node in leaf_nodes)
        assert all(node.parent is None for node in leaf_nodes)
        assert all(node.embedding == [0.1, 0.2, 0.3] for node in leaf_nodes)
        
        # Check that nodes are stored in the indexer
        assert len(indexer.nodes) == 4
    
    @patch('src.rag_engine.indexing.raptor_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.raptor_indexer.ChatGoogleGenerativeAI')
    @patch('src.rag_engine.indexing.raptor_indexer.KMeans')
    def test_cluster_nodes(self, mock_kmeans, mock_llm, mock_embeddings, config):
        """Test clustering of nodes"""
        indexer = RAPTORIndexer(config, cluster_size_threshold=2)
        
        # Create test nodes
        nodes = []
        for i in range(6):
            node = RAPTORNode(
                node_id=f"node_{i}",
                content=f"Content {i}",
                summary=f"Summary {i}",
                embedding=[i * 0.1, i * 0.2, i * 0.3],
                level=0,
                children=[],
                parent=None,
                metadata={}
            )
            nodes.append(node)
        
        # Mock KMeans
        mock_kmeans_instance = Mock()
        mock_kmeans_instance.fit_predict.return_value = [0, 0, 1, 1, 2, 2]
        mock_kmeans.return_value = mock_kmeans_instance
        
        clusters = indexer._cluster_nodes(nodes)
        
        assert len(clusters) == 3
        assert all(len(cluster) == 2 for cluster in clusters)
        mock_kmeans.assert_called_once()
    
    @patch('src.rag_engine.indexing.raptor_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.raptor_indexer.ChatGoogleGenerativeAI')
    def test_cluster_nodes_small_group(self, mock_llm, mock_embeddings, config):
        """Test clustering with small group of nodes"""
        indexer = RAPTORIndexer(config, cluster_size_threshold=5)
        
        # Create test nodes (less than threshold)
        nodes = []
        for i in range(3):
            node = RAPTORNode(
                node_id=f"node_{i}",
                content=f"Content {i}",
                summary=f"Summary {i}",
                embedding=[i * 0.1, i * 0.2, i * 0.3],
                level=0,
                children=[],
                parent=None,
                metadata={}
            )
            nodes.append(node)
        
        clusters = indexer._cluster_nodes(nodes)
        
        # Should return single cluster with all nodes
        assert len(clusters) == 1
        assert len(clusters[0]) == 3
    
    @patch('src.rag_engine.indexing.raptor_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.raptor_indexer.ChatGoogleGenerativeAI')
    def test_create_parent_node(self, mock_llm, mock_embeddings, config):
        """Test creation of parent node from child nodes"""
        mock_embeddings_instance = Mock()
        mock_embeddings_instance.embed_query.return_value = [0.5, 0.6, 0.7]
        mock_embeddings.return_value = mock_embeddings_instance
        
        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance
        
        indexer = RAPTORIndexer(config)
        
        # Mock the summary chain
        indexer.cluster_summary_chain = Mock()
        indexer.cluster_summary_chain.invoke.return_value = "Combined summary"
        
        # Create child nodes
        child_nodes = []
        for i in range(2):
            node = RAPTORNode(
                node_id=f"child_{i}",
                content=f"Child content {i}",
                summary=f"Child summary {i}",
                embedding=[i * 0.1, i * 0.2, i * 0.3],
                level=0,
                children=[],
                parent=None,
                metadata={}
            )
            child_nodes.append(node)
            indexer.nodes[node.node_id] = node
        
        parent_node = indexer._create_parent_node(child_nodes, level=1)
        
        assert parent_node.level == 1
        assert parent_node.summary == "Combined summary"
        assert parent_node.embedding == [0.5, 0.6, 0.7]
        assert len(parent_node.children) == 2
        assert parent_node.children == ["child_0", "child_1"]
        
        # Check that child nodes now point to parent
        for child in child_nodes:
            assert child.parent == parent_node.node_id
    
    @patch('src.rag_engine.indexing.raptor_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.raptor_indexer.ChatGoogleGenerativeAI')
    @patch('src.rag_engine.indexing.raptor_indexer.Chroma')
    def test_create_vectorstore_for_level(self, mock_chroma, mock_llm, mock_embeddings, config):
        """Test creation of vector store for a level"""
        mock_vectorstore = Mock()
        mock_chroma.from_documents.return_value = mock_vectorstore
        
        indexer = RAPTORIndexer(config)
        
        # Create test nodes
        nodes = []
        for i in range(2):
            node = RAPTORNode(
                node_id=f"node_{i}",
                content=f"Content {i}",
                summary=f"Summary {i}",
                embedding=[i * 0.1, i * 0.2, i * 0.3],
                level=1,
                children=[],
                parent=None,
                metadata={"key": f"value_{i}"}
            )
            nodes.append(node)
        
        indexer._create_vectorstore_for_level(1, nodes)
        
        assert 1 in indexer.vectorstores
        assert indexer.vectorstores[1] == mock_vectorstore
        mock_chroma.from_documents.assert_called_once()
    
    @patch('src.rag_engine.indexing.raptor_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.raptor_indexer.ChatGoogleGenerativeAI')
    def test_index_empty_documents(self, mock_llm, mock_embeddings, config):
        """Test indexing with empty document list"""
        indexer = RAPTORIndexer(config)
        
        result = indexer.index_documents([])
        
        assert result is True
        assert indexer._document_count == 0
    
    @patch('src.rag_engine.indexing.raptor_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.raptor_indexer.ChatGoogleGenerativeAI')
    def test_get_document_count(self, mock_llm, mock_embeddings, config):
        """Test getting document count"""
        indexer = RAPTORIndexer(config)
        
        assert indexer.get_document_count() == 0
        
        indexer._document_count = 5
        assert indexer.get_document_count() == 5
    
    @patch('src.rag_engine.indexing.raptor_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.raptor_indexer.ChatGoogleGenerativeAI')
    def test_clear_index(self, mock_llm, mock_embeddings, config):
        """Test clearing the index"""
        indexer = RAPTORIndexer(config)
        
        # Set some state
        indexer.nodes["test"] = Mock()
        indexer.root_nodes = ["root1"]
        indexer.level_nodes = {0: ["node1"]}
        indexer.vectorstores = {0: Mock()}
        indexer._document_count = 5
        
        result = indexer.clear_index()
        
        assert result is True
        assert len(indexer.nodes) == 0
        assert len(indexer.root_nodes) == 0
        assert len(indexer.level_nodes) == 0
        assert len(indexer.vectorstores) == 0
        assert indexer._document_count == 0
    
    @patch('src.rag_engine.indexing.raptor_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.raptor_indexer.ChatGoogleGenerativeAI')
    def test_search_at_level_no_vectorstore(self, mock_llm, mock_embeddings, config):
        """Test search at level when no vectorstore exists"""
        indexer = RAPTORIndexer(config)
        
        results = indexer.search_at_level("test query", level=0)
        
        assert results == []
    
    @patch('src.rag_engine.indexing.raptor_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.raptor_indexer.ChatGoogleGenerativeAI')
    def test_search_at_level_success(self, mock_llm, mock_embeddings, config):
        """Test successful search at level"""
        indexer = RAPTORIndexer(config)
        
        # Create test node
        test_node = RAPTORNode(
            node_id="test_node",
            content="Test content",
            summary="Test summary",
            embedding=[0.1, 0.2, 0.3],
            level=0,
            children=[],
            parent=None,
            metadata={}
        )
        indexer.nodes["test_node"] = test_node
        
        # Mock vectorstore
        mock_vectorstore = Mock()
        mock_doc = Mock()
        mock_doc.metadata = {"node_id": "test_node"}
        mock_vectorstore.similarity_search.return_value = [mock_doc]
        indexer.vectorstores[0] = mock_vectorstore
        
        results = indexer.search_at_level("test query", level=0, k=1)
        
        assert len(results) == 1
        assert results[0] == test_node
        mock_vectorstore.similarity_search.assert_called_once_with("test query", k=1)
    
    @patch('src.rag_engine.indexing.raptor_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.raptor_indexer.ChatGoogleGenerativeAI')
    def test_hierarchical_search_no_vectorstores(self, mock_llm, mock_embeddings, config):
        """Test hierarchical search when no vectorstores exist"""
        indexer = RAPTORIndexer(config)
        
        results = indexer.hierarchical_search("test query")
        
        assert results == []
    
    @patch('src.rag_engine.indexing.raptor_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.raptor_indexer.ChatGoogleGenerativeAI')
    def test_get_tree_info(self, mock_llm, mock_embeddings, config):
        """Test getting tree information"""
        indexer = RAPTORIndexer(config, max_levels=3, cluster_size_threshold=5)
        
        # Set some state
        indexer.nodes = {"node1": Mock(), "node2": Mock()}
        indexer.level_nodes = {0: ["node1"], 1: ["node2"]}
        indexer.root_nodes = ["node2"]
        indexer._document_count = 10
        
        info = indexer.get_tree_info()
        
        expected_info = {
            "total_nodes": 2,
            "total_levels": 2,
            "nodes_per_level": {0: 1, 1: 1},
            "root_nodes": 1,
            "document_count": 10,
            "max_levels": 3,
            "cluster_size_threshold": 5
        }
        
        assert info == expected_info
    
    @patch('src.rag_engine.indexing.raptor_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.raptor_indexer.ChatGoogleGenerativeAI')
    def test_get_node(self, mock_llm, mock_embeddings, config):
        """Test getting a node by ID"""
        indexer = RAPTORIndexer(config)
        
        test_node = Mock()
        indexer.nodes["test_id"] = test_node
        
        # Test existing node
        result = indexer.get_node("test_id")
        assert result == test_node
        
        # Test non-existing node
        result = indexer.get_node("non_existing")
        assert result is None
    
    @patch('src.rag_engine.indexing.raptor_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.raptor_indexer.ChatGoogleGenerativeAI')
    def test_get_node_path(self, mock_llm, mock_embeddings, config):
        """Test getting path from node to root"""
        indexer = RAPTORIndexer(config)
        
        # Create a chain of nodes: child -> parent -> grandparent
        grandparent = RAPTORNode("gp", "content", "summary", [0.1], 2, [], None, {})
        parent = RAPTORNode("p", "content", "summary", [0.1], 1, [], "gp", {})
        child = RAPTORNode("c", "content", "summary", [0.1], 0, [], "p", {})
        
        indexer.nodes = {"gp": grandparent, "p": parent, "c": child}
        
        path = indexer.get_node_path("c")
        
        assert len(path) == 3
        assert path[0] == child
        assert path[1] == parent
        assert path[2] == grandparent
    
    @patch('src.rag_engine.indexing.raptor_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.raptor_indexer.ChatGoogleGenerativeAI')
    def test_get_node_path_single_node(self, mock_llm, mock_embeddings, config):
        """Test getting path for a single node with no parent"""
        indexer = RAPTORIndexer(config)
        
        single_node = RAPTORNode("single", "content", "summary", [0.1], 0, [], None, {})
        indexer.nodes = {"single": single_node}
        
        path = indexer.get_node_path("single")
        
        assert len(path) == 1
        assert path[0] == single_node