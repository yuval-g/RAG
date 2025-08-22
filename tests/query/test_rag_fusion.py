"""
Unit tests for RAGFusionProcessor
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.rag_engine.query.rag_fusion import RAGFusionProcessor
from src.rag_engine.core.models import Document, ProcessedQuery
from src.rag_engine.core.exceptions import QueryProcessingError


class TestRAGFusionProcessor:
    """Test cases for RAGFusionProcessor"""
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing"""
        return [
            Document(content="Task decomposition is breaking down complex tasks", metadata={"id": "1"}),
            Document(content="LLM agents use planning for task management", metadata={"id": "2"}),
            Document(content="Agents can decompose tasks into subtasks", metadata={"id": "3"}),
            Document(content="Planning involves creating step-by-step approaches", metadata={"id": "4"}),
            Document(content="Task breakdown helps in systematic problem solving", metadata={"id": "5"})
        ]
    
    @pytest.fixture
    def processor(self):
        """Create a RAGFusionProcessor instance for testing"""
        with patch('src.rag_engine.query.rag_fusion.ChatGoogleGenerativeAI'):
            return RAGFusionProcessor(
                llm_model="gemini-2.0-flash-lite",
                temperature=0.0,
                num_queries=4,
                rrf_k=60
            )
    
    def test_init(self):
        """Test RAGFusionProcessor initialization"""
        with patch('src.rag_engine.query.rag_fusion.ChatGoogleGenerativeAI') as mock_llm:
            processor = RAGFusionProcessor(
                llm_model="gemini-2.0-flash-lite",
                temperature=0.2,
                num_queries=3,
                rrf_k=50
            )
            
            assert processor.llm_model == "gemini-2.0-flash-lite"
            assert processor.temperature == 0.2
            assert processor.num_queries == 3
            assert processor.rrf_k == 50
            mock_llm.assert_called_once()
    
    def test_parse_queries(self, processor):
        """Test query parsing from LLM output"""
        output = """1. How do LLM agents handle task decomposition?
2. What are the methods for breaking down complex tasks?
3. Explain task decomposition in AI systems
4. What is the process of task breakdown?"""
        
        parsed = processor._parse_queries(output)
        
        expected = [
            "How do LLM agents handle task decomposition?",
            "What are the methods for breaking down complex tasks?",
            "Explain task decomposition in AI systems",
            "What is the process of task breakdown?"
        ]
        
        assert parsed == expected
    
    def test_parse_queries_limit(self, processor):
        """Test query parsing with limit"""
        output = """1. Query one
2. Query two
3. Query three
4. Query four
5. Query five
6. Query six"""
        
        parsed = processor._parse_queries(output)
        
        # Should limit to num_queries (4)
        assert len(parsed) == 4
        assert parsed[0] == "Query one"
        assert parsed[3] == "Query four"
    
    def test_reciprocal_rank_fusion_basic(self, processor, sample_documents):
        """Test basic reciprocal rank fusion"""
        # Create different result lists
        results = [
            [sample_documents[0], sample_documents[1], sample_documents[2]],  # Query 1 results
            [sample_documents[1], sample_documents[0], sample_documents[3]],  # Query 2 results
            [sample_documents[0], sample_documents[3], sample_documents[4]]   # Query 3 results
        ]
        
        fused_results = processor.reciprocal_rank_fusion(results)
        
        # Should return list of (document, score) tuples
        assert len(fused_results) > 0
        assert all(isinstance(item, tuple) and len(item) == 2 for item in fused_results)
        assert all(isinstance(item[0], Document) for item in fused_results)
        assert all(isinstance(item[1], float) for item in fused_results)
        
        # Results should be sorted by score (descending)
        scores = [score for _, score in fused_results]
        assert scores == sorted(scores, reverse=True)
    
    def test_reciprocal_rank_fusion_scoring(self, processor, sample_documents):
        """Test RRF scoring logic"""
        # Document appears in first position in all lists - should get highest score
        results = [
            [sample_documents[0], sample_documents[1]],
            [sample_documents[0], sample_documents[2]],
            [sample_documents[0], sample_documents[3]]
        ]
        
        fused_results = processor.reciprocal_rank_fusion(results, k=60)
        
        # First document should have highest score
        top_doc, top_score = fused_results[0]
        assert top_doc.metadata["id"] == "1"  # sample_documents[0]
        
        # Score should be 3 * (1/(0+60)) = 3/60 = 0.05
        expected_score = 3 * (1 / (0 + 60))
        assert abs(top_score - expected_score) < 0.001
    
    def test_reciprocal_rank_fusion_empty_results(self, processor):
        """Test RRF with empty results"""
        results = [[], [], []]
        
        fused_results = processor.reciprocal_rank_fusion(results)
        
        assert fused_results == []
    
    def test_reciprocal_rank_fusion_custom_k(self, processor, sample_documents):
        """Test RRF with custom k parameter"""
        results = [[sample_documents[0]]]
        
        # Test with different k values
        result_k10 = processor.reciprocal_rank_fusion(results, k=10)
        result_k100 = processor.reciprocal_rank_fusion(results, k=100)
        
        # Higher k should result in lower scores
        score_k10 = result_k10[0][1]
        score_k100 = result_k100[0][1]
        assert score_k10 > score_k100
    
    @patch('src.rag_engine.query.rag_fusion.ChatGoogleGenerativeAI')
    def test_generate_fusion_queries_success(self, mock_llm_class):
        """Test successful fusion query generation"""
        # Setup mock
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        processor = RAGFusionProcessor()
        
        # Mock the chain invoke
        processor.generation_chain = Mock()
        processor.generation_chain.invoke.return_value = [
            "How do agents handle task decomposition?",
            "What are task breakdown methods?",
            "Explain task decomposition process"
        ]
        
        question = "What is task decomposition for LLM agents?"
        queries = processor.generate_fusion_queries(question)
        
        # Should include original + alternatives
        assert len(queries) >= 1
        assert question in queries  # Original should be included first
        assert queries[0] == question  # Original should be first
    
    @patch('src.rag_engine.query.rag_fusion.ChatGoogleGenerativeAI')
    def test_generate_fusion_queries_failure(self, mock_llm_class):
        """Test fusion query generation failure handling"""
        # Setup mock to raise exception
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        processor = RAGFusionProcessor()
        processor.generation_chain = Mock()
        processor.generation_chain.invoke.side_effect = Exception("LLM error")
        
        question = "What is task decomposition?"
        
        with pytest.raises(QueryProcessingError):
            processor.generate_fusion_queries(question)
    
    @patch('src.rag_engine.query.rag_fusion.ChatGoogleGenerativeAI')
    def test_process_with_retriever_success(self, mock_llm_class, sample_documents):
        """Test successful processing with retriever"""
        # Setup mock
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        processor = RAGFusionProcessor()
        
        # Mock generate_fusion_queries
        processor.generate_fusion_queries = Mock()
        processor.generate_fusion_queries.return_value = [
            "What is task decomposition?",
            "How do agents break down tasks?",
            "Explain task breakdown methods"
        ]
        
        # Mock retriever function
        def mock_retriever(query):
            if "decomposition" in query:
                return [sample_documents[0], sample_documents[2]]
            elif "break down" in query:
                return [sample_documents[1], sample_documents[0]]
            else:
                return [sample_documents[3], sample_documents[4]]
        
        question = "What is task decomposition?"
        results = processor.process_with_retriever(question, mock_retriever, top_k=3)
        
        # Should return list of (document, score) tuples
        assert len(results) <= 3
        assert all(isinstance(item, tuple) and len(item) == 2 for item in results)
        assert all(isinstance(item[0], Document) for item in results)
        assert all(isinstance(item[1], float) for item in results)
    
    @patch('src.rag_engine.query.rag_fusion.ChatGoogleGenerativeAI')
    def test_process_with_retriever_failure(self, mock_llm_class):
        """Test processing with retriever failure handling"""
        # Setup mock
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        processor = RAGFusionProcessor()
        processor.generate_fusion_queries = Mock()
        processor.generate_fusion_queries.side_effect = Exception("Generation failed")
        
        def mock_retriever(query):
            return []
        
        question = "What is task decomposition?"
        
        with pytest.raises(QueryProcessingError):
            processor.process_with_retriever(question, mock_retriever)
    
    @patch('src.rag_engine.query.rag_fusion.ChatGoogleGenerativeAI')
    def test_process_query_success(self, mock_llm_class):
        """Test successful query processing"""
        # Setup mock
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        processor = RAGFusionProcessor()
        
        # Mock generate_fusion_queries
        processor.generate_fusion_queries = Mock()
        processor.generate_fusion_queries.return_value = [
            "What is task decomposition for LLM agents?",
            "How do agents handle task decomposition?",
            "What are task breakdown methods?",
            "Explain task decomposition process"
        ]
        
        question = "What is task decomposition for LLM agents?"
        result = processor.process_query(question)
        
        assert isinstance(result, ProcessedQuery)
        assert result.original_query == question
        assert len(result.transformed_queries) == 4
        assert result.strategy_used == "rag_fusion"
        assert "num_fusion_queries" in result.metadata
        assert result.metadata["num_fusion_queries"] == 4
        assert result.metadata["rrf_k"] == 60
    
    @patch('src.rag_engine.query.rag_fusion.ChatGoogleGenerativeAI')
    def test_process_query_failure(self, mock_llm_class):
        """Test query processing failure handling"""
        # Setup mock
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        processor = RAGFusionProcessor()
        processor.generate_fusion_queries = Mock()
        processor.generate_fusion_queries.side_effect = Exception("Generation failed")
        
        question = "What is task decomposition?"
        
        with pytest.raises(QueryProcessingError):
            processor.process_query(question)
    
    def test_get_config(self, processor):
        """Test configuration retrieval"""
        config = processor.get_config()
        
        assert config["llm_model"] == "gemini-2.0-flash-lite"
        assert config["temperature"] == 0.0
        assert config["num_queries"] == 4
        assert config["rrf_k"] == 60
        assert config["strategy"] == "rag_fusion"
    
    def test_create_prompt_template(self, processor):
        """Test prompt template creation"""
        template = processor._create_prompt_template()
        
        # Check that template contains expected elements
        template_str = str(template)
        assert "helpful assistant" in template_str
        assert "multiple search queries" in template_str
        assert "4 queries" in template_str  # num_queries
        assert "question" in template_str
    
    def test_validate_queries_with_fallback(self, processor):
        """Test validation with fallback when no valid queries"""
        original = "What is task decomposition?"
        queries = ["", "   ", "A" * 600]  # All invalid
        
        validated = processor._validate_queries(queries, original)
        
        # Should create fallback queries
        assert len(validated) >= 1
        assert all(isinstance(q, str) and len(q) > 0 for q in validated)
    
    def test_is_too_similar(self, processor):
        """Test similarity detection"""
        query1 = "What is task decomposition for LLM agents?"
        query2 = "What is task decomposition for LLM agents?"  # Identical
        query3 = "How do agents handle planning?"  # Different
        
        assert processor._is_too_similar(query1, query2) == True
        assert processor._is_too_similar(query1, query3) == False