"""
Integration tests for the unified QueryProcessor
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.rag_engine.query.processor import QueryProcessor, QueryStrategy
from src.rag_engine.core.models import Document, ProcessedQuery
from src.rag_engine.core.exceptions import QueryProcessingError


class TestQueryProcessorIntegration:
    """Integration test cases for QueryProcessor"""
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing"""
        return [
            Document(content="Task decomposition involves breaking down complex tasks", metadata={"id": "1"}),
            Document(content="LLM agents use planning for systematic problem solving", metadata={"id": "2"}),
            Document(content="Memory systems help agents retain information", metadata={"id": "3"}),
            Document(content="Tool use enables agents to interact with external systems", metadata={"id": "4"}),
            Document(content="Hierarchical planning enables multi-step reasoning", metadata={"id": "5"})
        ]
    
    @pytest.fixture
    def processor(self):
        """Create a QueryProcessor instance for testing"""
        with patch('src.rag_engine.query.multi_query.ChatGoogleGenerativeAI'), \
             patch('src.rag_engine.query.rag_fusion.ChatGoogleGenerativeAI'), \
             patch('src.rag_engine.query.decomposition.ChatGoogleGenerativeAI'), \
             patch('src.rag_engine.query.step_back.ChatGoogleGenerativeAI'), \
             patch('src.rag_engine.query.hyde.ChatGoogleGenerativeAI'):
            return QueryProcessor(
                llm_model="gemini-2.0-flash-lite",
                temperature=0.0,
                default_strategy="multi_query"
            )
    
    def test_init_success(self):
        """Test successful QueryProcessor initialization"""
        with patch('src.rag_engine.query.multi_query.ChatGoogleGenerativeAI'), \
             patch('src.rag_engine.query.rag_fusion.ChatGoogleGenerativeAI'), \
             patch('src.rag_engine.query.decomposition.ChatGoogleGenerativeAI'), \
             patch('src.rag_engine.query.step_back.ChatGoogleGenerativeAI'), \
             patch('src.rag_engine.query.hyde.ChatGoogleGenerativeAI'):
            
            processor = QueryProcessor(
                llm_model="gemini-2.0-flash-lite",
                temperature=0.3,
                default_strategy="rag_fusion"
            )
            
            assert processor.llm_model == "gemini-2.0-flash-lite"
            assert processor.temperature == 0.3
            assert processor.default_strategy == "rag_fusion"
            
            # Check that all processors are initialized
            assert hasattr(processor, 'multi_query_generator')
            assert hasattr(processor, 'rag_fusion_processor')
            assert hasattr(processor, 'query_decomposer')
            assert hasattr(processor, 'step_back_processor')
            assert hasattr(processor, 'hyde_processor')
    
    def test_init_failure(self):
        """Test QueryProcessor initialization failure"""
        from src.rag_engine.query.multi_query import MultiQueryGenerator
        
        # Patch the MultiQueryGenerator class directly to ensure we test the failure handling
        with patch.object(MultiQueryGenerator, '__init__') as mock_multi:
            mock_multi.side_effect = Exception("Initialization failed")
            
            with pytest.raises(QueryProcessingError):
                QueryProcessor()
    
    def test_get_available_strategies(self, processor):
        """Test getting available strategies"""
        strategies = processor.get_available_strategies()
        
        expected_strategies = [
            "multi_query",
            "rag_fusion", 
            "decomposition",
            "step_back",
            "hyde",
            "basic"
        ]
        
        assert len(strategies) == len(expected_strategies)
        for strategy in expected_strategies:
            assert strategy in strategies
    
    def test_is_valid_strategy(self, processor):
        """Test strategy validation"""
        assert processor._is_valid_strategy("multi_query") == True
        assert processor._is_valid_strategy("rag_fusion") == True
        assert processor._is_valid_strategy("decomposition") == True
        assert processor._is_valid_strategy("step_back") == True
        assert processor._is_valid_strategy("hyde") == True
        assert processor._is_valid_strategy("basic") == True
        assert processor._is_valid_strategy("invalid_strategy") == False
    
    def test_configure_strategy_success(self, processor):
        """Test successful strategy configuration"""
        processor.configure_strategy("multi_query", num_queries=7, temperature=0.2)
        
        assert "multi_query" in processor.strategy_configs
        assert processor.strategy_configs["multi_query"]["num_queries"] == 7
        assert processor.strategy_configs["multi_query"]["temperature"] == 0.2
    
    def test_configure_strategy_invalid(self, processor):
        """Test configuration with invalid strategy"""
        with pytest.raises(QueryProcessingError):
            processor.configure_strategy("invalid_strategy", param=1)
    
    def test_set_default_strategy_success(self, processor):
        """Test setting default strategy"""
        processor.set_default_strategy("rag_fusion")
        assert processor.default_strategy == "rag_fusion"
    
    def test_set_default_strategy_invalid(self, processor):
        """Test setting invalid default strategy"""
        with pytest.raises(QueryProcessingError):
            processor.set_default_strategy("invalid_strategy")
    
    def test_process_multi_query_strategy(self, processor):
        """Test processing with multi-query strategy"""
        # Mock the multi-query generator
        processor.multi_query_generator.process_query = Mock()
        processor.multi_query_generator.process_query.return_value = ProcessedQuery(
            original_query="What is task decomposition?",
            transformed_queries=["What is task decomposition?", "How do agents break down tasks?"],
            strategy_used="multi_query",
            metadata={"num_generated": 1}
        )
        
        query = "What is task decomposition?"
        result = processor.process(query, "multi_query")
        
        assert isinstance(result, ProcessedQuery)
        assert result.strategy_used == "multi_query"
        assert len(result.transformed_queries) == 2
        processor.multi_query_generator.process_query.assert_called_once_with(query)
    
    def test_process_rag_fusion_strategy(self, processor):
        """Test processing with RAG-Fusion strategy"""
        # Mock the RAG-Fusion processor
        processor.rag_fusion_processor.process_query = Mock()
        processor.rag_fusion_processor.process_query.return_value = ProcessedQuery(
            original_query="What is task decomposition?",
            transformed_queries=["What is task decomposition?", "How do agents handle tasks?"],
            strategy_used="rag_fusion",
            metadata={"num_fusion_queries": 2}
        )
        
        query = "What is task decomposition?"
        result = processor.process(query, "rag_fusion")
        
        assert isinstance(result, ProcessedQuery)
        assert result.strategy_used == "rag_fusion"
        processor.rag_fusion_processor.process_query.assert_called_once_with(query)
    
    def test_process_decomposition_strategy(self, processor):
        """Test processing with decomposition strategy"""
        # Mock the query decomposer
        processor.query_decomposer.process_query = Mock()
        processor.query_decomposer.process_query.return_value = ProcessedQuery(
            original_query="What are the main components of LLM agents?",
            transformed_queries=["What are the core components?", "How do they work?"],
            strategy_used="decomposition",
            metadata={"num_sub_questions": 2}
        )
        
        query = "What are the main components of LLM agents?"
        result = processor.process(query, "decomposition")
        
        assert isinstance(result, ProcessedQuery)
        assert result.strategy_used == "decomposition"
        processor.query_decomposer.process_query.assert_called_once_with(query)
    
    def test_process_step_back_strategy(self, processor):
        """Test processing with step-back strategy"""
        # Mock the step-back processor
        processor.step_back_processor.process_query = Mock()
        processor.step_back_processor.process_query.return_value = ProcessedQuery(
            original_query="What is the specific implementation of task decomposition in GPT-4?",
            transformed_queries=["What is the specific implementation of task decomposition in GPT-4?", "What are general AI approaches?"],
            strategy_used="step_back",
            metadata={"step_back_question": "What are general AI approaches?"}
        )
        
        query = "What is the specific implementation of task decomposition in GPT-4?"
        result = processor.process(query, "step_back")
        
        assert isinstance(result, ProcessedQuery)
        assert result.strategy_used == "step_back"
        processor.step_back_processor.process_query.assert_called_once_with(query)
    
    def test_process_hyde_strategy(self, processor):
        """Test processing with HyDE strategy"""
        # Mock the HyDE processor
        processor.hyde_processor.process_query = Mock()
        processor.hyde_processor.process_query.return_value = ProcessedQuery(
            original_query="What is task decomposition?",
            transformed_queries=["Hypothetical document about task decomposition..."],
            strategy_used="hyde",
            metadata={"hypothetical_document": "Hypothetical document about task decomposition..."}
        )
        
        query = "What is task decomposition?"
        result = processor.process(query, "hyde")
        
        assert isinstance(result, ProcessedQuery)
        assert result.strategy_used == "hyde"
        processor.hyde_processor.process_query.assert_called_once_with(query)
    
    def test_process_basic_strategy(self, processor):
        """Test processing with basic strategy"""
        query = "What is task decomposition?"
        result = processor.process(query, "basic")
        
        assert isinstance(result, ProcessedQuery)
        assert result.original_query == query
        assert result.transformed_queries == [query]
        assert result.strategy_used == "basic"
        assert result.metadata["processor"] == "unified"
    
    def test_process_default_strategy(self, processor):
        """Test processing with default strategy"""
        # Mock the default strategy (multi_query)
        processor.multi_query_generator.process_query = Mock()
        processor.multi_query_generator.process_query.return_value = ProcessedQuery(
            original_query="What is task decomposition?",
            transformed_queries=["What is task decomposition?", "How do agents break down tasks?"],
            strategy_used="multi_query",
            metadata={"num_generated": 1}
        )
        
        query = "What is task decomposition?"
        result = processor.process(query, "default")  # Use "default" to trigger default strategy
        
        assert isinstance(result, ProcessedQuery)
        assert result.strategy_used == "multi_query"
        processor.multi_query_generator.process_query.assert_called_once()
    
    def test_process_invalid_strategy(self, processor):
        """Test processing with invalid strategy"""
        query = "What is task decomposition?"
        
        with pytest.raises(QueryProcessingError):
            processor.process(query, "invalid_strategy")
    
    def test_process_with_custom_config(self, processor):
        """Test processing with custom strategy configuration"""
        # Configure strategy
        processor.configure_strategy("multi_query", num_queries=7)
        
        # Mock the processor
        processor.multi_query_generator.process_query = Mock()
        processor.multi_query_generator.process_query.return_value = ProcessedQuery(
            original_query="What is task decomposition?",
            transformed_queries=["Query 1", "Query 2"],
            strategy_used="multi_query",
            metadata={}
        )
        
        query = "What is task decomposition?"
        result = processor.process(query, "multi_query")
        
        # Check that custom config was passed
        processor.multi_query_generator.process_query.assert_called_once_with(query, num_queries=7)
    
    def test_multi_query_method(self, processor):
        """Test direct multi-query method"""
        processor.multi_query_generator.generate_queries = Mock()
        processor.multi_query_generator.generate_queries.return_value = [
            "What is task decomposition?",
            "How do agents break down tasks?",
            "What are task breakdown methods?"
        ]
        
        query = "What is task decomposition?"
        result = processor.multi_query(query)
        
        assert len(result) == 3
        assert "What is task decomposition?" in result
        processor.multi_query_generator.generate_queries.assert_called_once_with(query)
    
    def test_decompose_method(self, processor):
        """Test direct decompose method"""
        processor.query_decomposer.decompose_query = Mock()
        processor.query_decomposer.decompose_query.return_value = [
            "What are the main components?",
            "How do they work?",
            "What are the benefits?"
        ]
        
        query = "What are the main components of LLM agents?"
        result = processor.decompose(query)
        
        assert len(result) == 3
        assert "What are the main components?" in result
        processor.query_decomposer.decompose_query.assert_called_once_with(query)
    
    def test_step_back_method(self, processor):
        """Test direct step-back method"""
        processor.step_back_processor.generate_step_back_question = Mock()
        processor.step_back_processor.generate_step_back_question.return_value = "What are general AI approaches?"
        
        query = "What is the specific implementation of task decomposition in GPT-4?"
        result = processor.step_back(query)
        
        assert result == "What are general AI approaches?"
        processor.step_back_processor.generate_step_back_question.assert_called_once_with(query)
    
    def test_hyde_method(self, processor):
        """Test direct HyDE method"""
        processor.hyde_processor.generate_hypothetical_document = Mock()
        processor.hyde_processor.generate_hypothetical_document.return_value = "Hypothetical document content..."
        
        query = "What is task decomposition?"
        result = processor.hyde(query)
        
        assert result == "Hypothetical document content..."
        processor.hyde_processor.generate_hypothetical_document.assert_called_once_with(query)
    
    def test_process_with_retriever_rag_fusion(self, processor, sample_documents):
        """Test processing with retriever using RAG-Fusion"""
        # Mock the RAG-Fusion processor
        processor.rag_fusion_processor.process_with_retriever = Mock()
        processor.rag_fusion_processor.process_with_retriever.return_value = [
            (sample_documents[0], 0.9),
            (sample_documents[1], 0.8)
        ]
        
        def mock_retriever(query):
            return sample_documents[:3]
        
        query = "What is task decomposition?"
        result = processor.process_with_retriever(query, "rag_fusion", mock_retriever, top_k=2)
        
        assert len(result) == 2
        assert all(isinstance(item, tuple) for item in result)
        processor.rag_fusion_processor.process_with_retriever.assert_called_once()
    
    def test_process_with_retriever_multi_query(self, processor, sample_documents):
        """Test processing with retriever using multi-query"""
        # Mock the multi-query generator
        processor.multi_query_generator.generate_queries = Mock()
        processor.multi_query_generator.generate_queries.return_value = [
            "What is task decomposition?",
            "How do agents break down tasks?"
        ]
        
        def mock_retriever(query):
            if "decomposition" in query:
                return sample_documents[:2]
            else:
                return sample_documents[2:4]
        
        query = "What is task decomposition?"
        result = processor.process_with_retriever(query, "multi_query", mock_retriever, top_k=3)
        
        assert len(result) <= 3
        assert all(isinstance(doc, Document) for doc in result)
    
    def test_get_processor_config(self, processor):
        """Test getting processor configuration"""
        # Mock configs
        processor.multi_query_generator.get_config = Mock()
        processor.multi_query_generator.get_config.return_value = {
            "llm_model": "gemini-2.0-flash-lite",
            "temperature": 0.0,
            "num_queries": 5,
            "strategy": "multi_query"
        }
        
        config = processor.get_processor_config("multi_query")
        
        assert config["strategy"] == "multi_query"
        assert config["llm_model"] == "gemini-2.0-flash-lite"
        processor.multi_query_generator.get_config.assert_called_once()
    
    def test_get_processor_config_invalid(self, processor):
        """Test getting config for invalid strategy"""
        with pytest.raises(QueryProcessingError):
            processor.get_processor_config("invalid_strategy")
    
    def test_get_all_configs(self, processor):
        """Test getting all processor configurations"""
        # Mock all configs
        processor.multi_query_generator.get_config = Mock(return_value={"strategy": "multi_query"})
        processor.rag_fusion_processor.get_config = Mock(return_value={"strategy": "rag_fusion"})
        processor.query_decomposer.get_config = Mock(return_value={"strategy": "decomposition"})
        processor.step_back_processor.get_config = Mock(return_value={"strategy": "step_back"})
        processor.hyde_processor.get_config = Mock(return_value={"strategy": "hyde"})
        
        configs = processor.get_all_configs()
        
        assert len(configs) == 6  # All strategies including basic
        assert "multi_query" in configs
        assert "rag_fusion" in configs
        assert "decomposition" in configs
        assert "step_back" in configs
        assert "hyde" in configs
        assert "basic" in configs
    
    def test_get_strategy_info(self, processor):
        """Test getting strategy information"""
        # Mock config
        processor.multi_query_generator.get_config = Mock()
        processor.multi_query_generator.get_config.return_value = {"strategy": "multi_query"}
        
        info = processor.get_strategy_info("multi_query")
        
        assert info["name"] == "multi_query"
        assert "description" in info
        assert "config" in info
        assert "custom_config" in info
        assert "supports_retriever" in info
        assert info["supports_retriever"] == True
    
    def test_get_strategy_info_invalid(self, processor):
        """Test getting info for invalid strategy"""
        with pytest.raises(QueryProcessingError):
            processor.get_strategy_info("invalid_strategy")
    
    def test_supports_retriever(self, processor):
        """Test checking retriever support"""
        assert processor._supports_retriever("multi_query") == True
        assert processor._supports_retriever("rag_fusion") == True
        assert processor._supports_retriever("decomposition") == True
        assert processor._supports_retriever("step_back") == True
        assert processor._supports_retriever("hyde") == True
        assert processor._supports_retriever("basic") == False
    
    def test_method_error_handling(self, processor):
        """Test error handling in direct methods"""
        # Test multi_query error
        processor.multi_query_generator.generate_queries = Mock()
        processor.multi_query_generator.generate_queries.side_effect = Exception("Generation failed")
        
        with pytest.raises(QueryProcessingError):
            processor.multi_query("test query")
        
        # Test decompose error
        processor.query_decomposer.decompose_query = Mock()
        processor.query_decomposer.decompose_query.side_effect = Exception("Decomposition failed")
        
        with pytest.raises(QueryProcessingError):
            processor.decompose("test query")
        
        # Test step_back error
        processor.step_back_processor.generate_step_back_question = Mock()
        processor.step_back_processor.generate_step_back_question.side_effect = Exception("Step-back failed")
        
        with pytest.raises(QueryProcessingError):
            processor.step_back("test query")
        
        # Test hyde error
        processor.hyde_processor.generate_hypothetical_document = Mock()
        processor.hyde_processor.generate_hypothetical_document.side_effect = Exception("HyDE failed")
        
        with pytest.raises(QueryProcessingError):
            processor.hyde("test query")